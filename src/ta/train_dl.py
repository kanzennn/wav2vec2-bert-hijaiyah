"""Training Wav2Vec2-BERT (fine-tune seluruh lapisan). Dipanggil dari ta.train bila model.type = w2v_bert.

Alur per sifat, diulang untuk tiap seed di cfg["seeds"]:
1. Validasi silang fold 1-5: latih di 4 fold, early stopping (macro-F1) di fold sisanya.
2. Jumlah epoch akhir = median epoch terbaik dari 5 fold CV (dipilih tanpa melihat data uji).
3. Latih ulang di seluruh split latih selama epoch akhir itu, uji SEKALI di fold 0.
4. Leave-item-group-out dengan hiperparameter dan jumlah epoch yang sama.
Fitur fbank dinormalisasi per rekaman oleh feature extractor bawaan (features/raw.py), jadi tidak ada statistik
yang dihitung dari data lain.

Supaya muat di GPU 8 GB: autocast bf16, gradient checkpointing, batch kecil + akumulasi gradien, AdamW 8-bit
(bitsandbytes). Bobot terbaik saat early stopping ditulis ke disk (berkas *.best.pt di samping checkpoint) dan
checkpoint dibaca langsung ke GPU, supaya tidak perlu menahan 2,3 GB bobot di RAM (RAM bebas bisa sempit).

Jeda/lanjut: tiap unit (sifat, seed, cv/test/ligo, fold) disimpan ke run_dir/units/ begitu selesai, dan model
yang sedang dilatih di-checkpoint tiap epoch ke run_dir/checkpoints/. `python -m ta.train --resume <run_dir>`
melewati unit yang sudah ada dan melanjutkan unit yang terpotong dari epoch terakhirnya.
"""
import gc
import json
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ta.data.labels import get_labels
from ta.data.split import cv_folds, ligo_folds, train_test_indices
from ta.evaluate import compute_metrics, confusion, error_tables, plot_confusion, subset_metrics
from ta.utils import set_seed

DEFAULT_TRAIN = dict(batch_size=8, grad_accum=4, lr=1e-3, backbone_lr=2e-5, weight_decay=0.01, optimizer="adamw8bit",
                     max_grad_norm=1.0, amp_dtype="bf16", max_epochs=10, patience=3, eval_batch=32, checkpoint=True)


def build_model(cfg):
    kind, params = cfg["model"]["type"], {**cfg["model"].get("params", {})}
    if kind == "w2v_bert":
        from ta.models.wav2vec_bert import DEFAULT_CHECKPOINT, W2VBertClassifier
        return W2VBertClassifier(checkpoint=cfg["model"].get("checkpoint", DEFAULT_CHECKPOINT), **params)
    raise ValueError(f"model deep learning belum didukung: {kind}")


class FbankData:
    """Fitur fbank (float16) di RAM, dipindah ke GPU per batch, beserta attention mask dari panjang valid."""

    def __init__(self, F, lengths, y, device):
        self.device = device
        self.X = torch.from_numpy(np.ascontiguousarray(F))
        self.lengths = torch.from_numpy(lengths.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.int64)).to(device)
        self.t = torch.arange(F.shape[1])

    def batch(self, idx):
        i = idx.cpu()
        mask = (self.t[None, :] < self.lengths[i][:, None]).long()
        return self.X[i].to(self.device).float(), mask.to(self.device), self.y[idx]


def _amp(tc, device):
    """(aktif, dtype) untuk autocast; bf16 tidak butuh GradScaler."""
    on = device.type == "cuda" and tc["amp_dtype"] in ("bf16", "fp16")
    return on, (torch.bfloat16 if tc["amp_dtype"] == "bf16" else torch.float16)


def make_optimizer(model, tc):
    groups = model.param_groups(tc) if hasattr(model, "param_groups") else model.parameters()
    if tc["optimizer"] == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(groups, lr=tc["lr"], weight_decay=tc["weight_decay"])
    return torch.optim.AdamW(groups, lr=tc["lr"], weight_decay=tc["weight_decay"])


@torch.no_grad()
def predict(model, data, idx, bs, tc):
    model.eval()
    on, dtype = _amp(tc, data.device)
    probs = []
    for c in torch.split(torch.as_tensor(np.array(idx), device=data.device), bs):
        x, m, _ = data.batch(c)
        with torch.autocast("cuda", dtype=dtype, enabled=on):
            logits = model(x, m)
        probs.append(torch.softmax(logits.float(), 1)[:, 1])
    return torch.cat(probs).cpu().numpy()


def _atomic(path, write):
    """Tulis ke berkas sementara lalu rename, supaya berkas tidak pernah setengah jadi bila proses dihentikan."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _save_checkpoint(path, st):
    """Simpan atomik. Bila disk tidak cukup untuk dua salinan, checkpoint lama dihapus dulu (kalau proses mati
    tepat saat menulis, unit itu diulang dari awal; hasil unit lain tetap aman)."""
    if path.exists() and shutil.disk_usage(path.parent).free < 2.2 * path.stat().st_size:
        path.unlink()
    _atomic(path, lambda t: torch.save(st, t))


def _rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu())
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def train_model(cfg, data, tr, seed, va=None, epochs=None, ckpt=None):
    """Latih satu model. Dengan va: early stopping di macro-F1 va, kembalikan bobot terbaik.
    Tanpa va: latih tepat `epochs` epoch. Mengembalikan (model, epoch_terpakai, riwayat per epoch).

    Bila `ckpt` (Path) diberikan, keadaan lengkap (model, optimizer, scaler, epoch, RNG) disimpan tiap akhir
    epoch dan bobot terbaik di `ckpt.best.pt`; bila berkasnya sudah ada, training dilanjutkan dari epoch berikutnya.
    Tanpa `ckpt` bobot terbaik disimpan di RAM (hanya untuk model kecil / tes)."""
    tc = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    set_seed(seed)
    dev = data.device
    model = build_model(cfg).to(dev)
    opt = make_optimizer(model, tc)
    on, dtype = _amp(tc, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=on and dtype == torch.float16)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tr_t = torch.as_tensor(np.array(tr))
    n_epochs = epochs if va is None else tc["max_epochs"]
    best, log, start = (-1.0, 0, None), [], 1
    best_path = None if ckpt is None else ckpt.with_name(ckpt.stem + ".best.pt")
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)   # langsung ke GPU, per tensor
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        gen.set_state(st["gen"].cpu())
        _set_rng_state(st["rng"])
        best, log, start = st["best"], st["log"], st["epoch"] + 1
        del st
        torch.cuda.empty_cache()
        print(f"  lanjut dari {ckpt.name} epoch {start - 1}")
    for ep in range(start, n_epochs + 1):
        t_ep = time.time()
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen)].to(dev)
        batches = torch.split(perm, tc["batch_size"])
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for bi, c in enumerate(batches):
            x, m, yb = data.batch(c)
            with torch.autocast("cuda", dtype=dtype, enabled=on):
                loss = loss_fn(model(x, m), yb)
            scaler.scale(loss / tc["grad_accum"]).backward()
            tot += loss.item() * len(c)
            if (bi + 1) % tc["grad_accum"] == 0 or bi + 1 == len(batches):
                if tc["max_grad_norm"]:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        row = {"epoch": ep, "train_loss": tot / len(tr)}
        stop = False
        if va is not None:
            p = predict(model, data, va, tc["eval_batch"], tc)
            yv = data.y[torch.as_tensor(np.array(va), device=dev)].cpu().numpy()
            row.update(compute_metrics(yv, (p >= 0.5).astype(int), p))
            if row["macro_f1"] > best[0]:
                if best_path is None:
                    best = (row["macro_f1"], ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                else:
                    _atomic(best_path, lambda t: torch.save(model.state_dict(), t))
                    best = (row["macro_f1"], ep, None)
            elif ep - best[1] >= tc["patience"]:
                stop = True
        log.append(row)
        print(f"  epoch {ep}: loss {row['train_loss']:.4f}, {time.time() - t_ep:.0f}s"
              + (f", val macro-F1 {row['macro_f1']:.4f}" if va is not None else ""), flush=True)
        if stop:
            break
        if ckpt is not None and tc["checkpoint"] and ep < n_epochs:
            _save_checkpoint(ckpt, {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                                    "scaler": scaler.state_dict(), "gen": gen.get_state(), "rng": _rng_state(),
                                    "best": best, "log": log})
    if va is not None:
        if best_path is None:
            model.load_state_dict(best[2])
        else:
            del opt                       # bebaskan state optimizer di GPU sebelum memuat bobot terbaik
            torch.cuda.empty_cache()
            model.load_state_dict(torch.load(best_path, map_location=dev))
        return model, best[1], log
    return model, n_epochs, log


def _log_memory(tag):
    """Cetak pemakaian memori proses dan sistem, untuk memantau kebocoran antar-unit."""
    try:
        import psutil
        proc, vm = psutil.Process().memory_info(), psutil.virtual_memory()
        gpu = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(f"  [mem {tag}] proses RSS {proc.rss / 2**30:.1f} GB, private {getattr(proc, 'private', 0) / 2**30:.1f} GB, "
              f"RAM bebas {vm.available / 2**30:.1f} GB, GPU {gpu:.2f} GB", flush=True)
    except Exception:
        pass


def make_pred(index, te, y, prob, seed):
    from ta.train import PRED_COLUMNS
    p = index.loc[te, PRED_COLUMNS].copy()
    p["seed"], p["y_true"], p["y_pred"], p["score"] = seed, y[te], (prob >= 0.5).astype(int), prob
    return p.reset_index(drop=True)


def run_unit(run_dir, key, fit):
    """Jalankan satu unit (satu model dilatih lalu memprediksi) atau muat hasilnya bila sudah ada.

    fit(ckpt_path) -> (prediksi, meta). Hasil selalu dibaca ulang dari disk, jadi run yang dilanjutkan dan
    run tanpa jeda memakai data yang persis sama untuk agregasi."""
    units, ckpts = run_dir / "units", run_dir / "checkpoints"
    pred_p, meta_p = units / f"{key}.csv", units / f"{key}.json"
    if not meta_p.exists():
        units.mkdir(exist_ok=True)
        ckpts.mkdir(exist_ok=True)
        ckpt = ckpts / f"{key}.pt"
        pred, meta = fit(ckpt)
        _atomic(pred_p, lambda t: pred.to_csv(t, index=False))
        _atomic(meta_p, lambda t: t.write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8"))
        ckpt.unlink(missing_ok=True)
        ckpt.with_name(ckpt.stem + ".best.pt").unlink(missing_ok=True)
    pred = pd.read_csv(pred_p, dtype={"source_id": str, "item": str, "fold": str})
    return pred, json.loads(meta_p.read_text(encoding="utf-8"))


def run_task_dl(task, cfg, index, feats, run_dir, smoke, run_id):
    from ta.train import result_rows
    tc = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    if smoke:
        tc["max_epochs"] = cfg.get("smoke", {}).get("max_epochs", 2)
    cfg = {**cfg, "train": tc}
    seeds = cfg.get("seeds", [cfg["seed"]])[: 1 if smoke else None]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    y = get_labels(task, index)
    data = FbankData(*feats, y, device)
    rows, preds, summaries = [], {"cv": [], "test": [], "ligo": []}, []

    def fit(tr, te, seed, va=None, epochs=None):
        def _fit(ckpt):
            _log_memory("mulai unit")
            model, ep, log = train_model(cfg, data, tr, seed, va=va, epochs=epochs, ckpt=ckpt)
            prob = predict(model, data, te, tc["eval_batch"], tc)
            # bebaskan model sebelum unit berikutnya memuat backbone baru (RAM/commit Windows terbatas)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return make_pred(index, te, y, prob, seed), {"epochs": ep, "history": log}
        return _fit

    for seed in seeds:
        cfg_s = {**cfg, "seed": seed}
        # 1. CV + early stopping
        t0, hist, best_epochs = time.time(), [], []
        for f, tr, va in cv_folds(index):
            p, meta = run_unit(run_dir, f"{task}_seed{seed}_cv_{f}", fit(tr, va, seed, va=va))
            hist += [{"seed": seed, "fold": f, **r} for r in meta["history"]]
            best_epochs.append(meta["epochs"])
            preds["cv"].append(p)
            rows += result_rows(task, cfg_s, "cv", f, subset_metrics(p), run_id)
        pd.DataFrame(hist).to_csv(run_dir / f"{task}_seed{seed}_cv_history.csv", index=False)
        final_epochs = max(1, int(np.median(best_epochs)))
        cv_f1 = [r["macro_f1"] for r in rows if r["seed"] == seed and r["mode"] == "cv" and r["subset"] == "all"]
        print(f"[{task} seed {seed}] CV {time.time() - t0:.0f}s, epoch terbaik {best_epochs} -> {final_epochs}; "
              f"macro-F1 {np.mean(cv_f1):.4f} +- {np.std(cv_f1, ddof=1):.4f}")

        # 2. uji akhir (fold 0), sekali per seed
        tr, te = train_test_indices(index)
        p, _ = run_unit(run_dir, f"{task}_seed{seed}_test_0", fit(tr, te, seed, epochs=final_epochs))
        preds["test"].append(p)
        test_m = subset_metrics(p)
        rows += result_rows(task, cfg_s, "test", 0, test_m, run_id)
        cm = confusion(p)
        cm.to_csv(run_dir / f"{task}_seed{seed}_cm_test.csv")
        plot_confusion(cm, f"{task} {cfg['model']['type']}+{cfg['features']['type']} seed {seed} (uji)", run_dir / f"{task}_seed{seed}_cm_test.png")

        # 3. leave-item-group-out
        lp = []
        for k, tr_k, te_k in ligo_folds(index):
            if len(te_k) == 0:
                continue
            pk, _ = run_unit(run_dir, f"{task}_seed{seed}_ligo_{k}", fit(tr_k, te_k, seed, epochs=final_epochs))
            pk = pk.assign(ligo_fold=k)
            lp.append(pk)
            rows += result_rows(task, cfg_s, "ligo", k, subset_metrics(pk), run_id)
        lp = pd.concat(lp)
        preds["ligo"].append(lp)
        ligo_m = subset_metrics(lp)
        rows += result_rows(task, cfg_s, "ligo", "pooled", ligo_m, run_id)
        confusion(lp).to_csv(run_dir / f"{task}_seed{seed}_cm_ligo.csv")
        print(f"[{task} seed {seed}] uji macro-F1 {test_m['all']['macro_f1']:.4f} (benar {test_m['benar']['macro_f1']:.4f}, "
              f"salah {test_m['salah']['macro_f1']:.4f}); LIGO {ligo_m['all']['macro_f1']:.4f}; total {time.time() - t0:.0f}s")
        summaries.append({"seed": seed, "cv_best_epochs": best_epochs, "final_epochs": final_epochs,
                          "cv_macro_f1_mean": float(np.mean(cv_f1)), "test": test_m, "ligo_pooled": ligo_m})

    for mode, ps in preds.items():
        pd.concat(ps).to_csv(run_dir / f"{task}_predictions_{mode}.csv", index=False)
    # tabel kesalahan dari gabungan semua seed (tingkat kesalahan = rata-rata antar-seed)
    for name, tab in zip(("by_item", "by_harakat", "by_item_harakat_kategori"), error_tables(pd.concat(preds["test"]))):
        tab.round(4).to_csv(run_dir / f"{task}_errors_test_{name}.csv", index=False)
    return rows, {"task": task, "train": tc, "seeds": summaries}
