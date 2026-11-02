"""Training deep learning: CNN/CRNN (log-Mel) dan wav2vec 2.0 (beku / fine-tune). Dipanggil dari ta.train.

Alur per sifat, diulang untuk tiap seed di cfg["seeds"]:
1. Validasi silang fold 1-5: latih di 4 fold, early stopping (macro-F1) di fold sisanya.
2. Jumlah epoch akhir = median epoch terbaik dari 5 fold CV (dipilih tanpa melihat data uji).
3. Latih ulang di seluruh split latih selama epoch akhir itu, uji SEKALI di fold 0.
4. Leave-item-group-out dengan hiperparameter dan jumlah epoch yang sama.
Normalisasi (per pita Mel / per dimensi embedding) dihitung dari data latih tiap run (fold / latih penuh / lipatan LIGO).
Audio mentah untuk fine-tune hanya dinormalisasi per rekaman (features/raw.py).

Jeda/lanjut: tiap unit (sifat, seed, cv/test/ligo, fold) disimpan ke run_dir/units/ begitu selesai, dan model
yang sedang dilatih di-checkpoint tiap epoch ke run_dir/checkpoints/. `python -m ta.train --resume <run_dir>`
melewati unit yang sudah ada dan melanjutkan unit yang terpotong dari epoch terakhirnya.
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ta.data.labels import get_labels
from ta.data.split import cv_folds, ligo_folds, train_test_indices
from ta.evaluate import compute_metrics, confusion, error_tables, plot_confusion, subset_metrics
from ta.utils import set_seed

DEFAULT_TRAIN = dict(batch_size=64, lr=1e-3, weight_decay=1e-4, max_epochs=40, patience=8, amp=True, eval_batch=512)


def build_model(cfg, data):
    kind, params = cfg["model"]["type"], {**cfg["model"].get("params", {})}
    n_mels = data.X.shape[1]
    if "channels" in params:
        params["channels"] = tuple(params["channels"])
    if kind == "cnn":
        from ta.models.cnn import MelCNN
        return MelCNN(n_mels=n_mels, **params)
    if kind == "crnn":
        from ta.models.crnn import MelCRNN
        return MelCRNN(n_mels=n_mels, **params)
    if kind == "w2v_frozen":
        from ta.models.wav2vec import W2VHead
        return W2VHead(n_layers=data.X.shape[1], dim=data.X.shape[2], **params)
    if kind == "w2v_finetune":
        from ta.models.wav2vec import W2VFineTune
        return W2VFineTune(checkpoint=cfg["model"].get("checkpoint", "pretrained/facebook-wav2vec2-base"), **params)
    raise ValueError(f"model deep learning belum didukung: {kind}")


class MelData:
    """Seluruh spectrogram (float16) dan label di GPU; batch dinormalisasi saat diambil."""

    def __init__(self, X, vf, y, device):
        self.device = device
        self.X = torch.from_numpy(X).to(device)
        self.vf = torch.from_numpy(vf.astype(np.int64)).to(device)
        self.y = torch.from_numpy(y.astype(np.int64)).to(device)
        self.t = torch.arange(X.shape[2], device=device)
        self.mean = self.std = None

    def fit_norm(self, idx, chunk=2048):
        """Mean/std per pita Mel dari frame valid rekaman idx (data latih saja)."""
        s = ss = n = 0.0
        for c in torch.split(torch.as_tensor(np.array(idx), device=self.X.device), chunk):
            x = self.X[c].double()
            m = (self.t[None, :] < self.vf[c][:, None]).double()[:, None, :]
            s = s + (x * m).sum((0, 2))
            ss = ss + (x * x * m).sum((0, 2))
            n = n + m.sum()
        mean = s / n
        self.mean = mean.float()[:, None]
        self.std = torch.sqrt(torch.clamp(ss / n - mean ** 2, min=1e-6)).float()[:, None]

    def batch(self, idx):
        x = (self.X[idx].float() - self.mean) / self.std
        x = x * (self.t[None, :] < self.vf[idx][:, None]).float()[:, None, :]   # padding = 0
        return x, self.vf[idx], self.y[idx]


class EmbData:
    """Embedding wav2vec beku [N, lapis, dim] di GPU; standarisasi per (lapis, dim) dari data latih."""

    def __init__(self, E, y, device):
        self.device = device
        self.X = torch.from_numpy(np.ascontiguousarray(E)).to(device)
        self.y = torch.from_numpy(y.astype(np.int64)).to(device)

    def fit_norm(self, idx):
        x = self.X[torch.as_tensor(np.array(idx), device=self.device)].double()
        self.mean = x.mean(0).float()
        self.std = torch.clamp(x.std(0), min=1e-6).float()

    def batch(self, idx):
        return (self.X[idx].float() - self.mean) / self.std, None, self.y[idx]


class RawData:
    """Audio mentah (sudah dinormalisasi per rekaman) di RAM; dipindah ke GPU per batch."""

    def __init__(self, W, valid, y, device):
        self.device = device
        self.X = torch.from_numpy(W)
        self.valid = torch.from_numpy(valid.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.int64)).to(device)

    def fit_norm(self, idx):
        pass

    def batch(self, idx):
        i = idx.cpu()
        return (self.X[i].to(self.device, non_blocking=True).float(), self.valid[i].to(self.device), self.y[idx])


def make_data(cfg, feats, y, device):
    kind = cfg["model"]["type"]
    if kind in ("cnn", "crnn"):
        return MelData(*feats, y, device)
    if kind == "w2v_frozen":
        return EmbData(feats[0], y, device)
    return RawData(*feats, y, device)


@torch.no_grad()
def predict(model, data, idx, bs):
    model.eval()
    probs = []
    for c in torch.split(torch.as_tensor(np.array(idx), device=data.device), bs):
        x, l, _ = data.batch(c)
        with torch.autocast("cuda", enabled=data.device.type == "cuda"):
            logits = model(x, l)
        probs.append(torch.softmax(logits.float(), 1)[:, 1])
    return torch.cat(probs).cpu().numpy()


def _atomic(path, write):
    """Tulis ke berkas sementara lalu rename, supaya berkas tidak pernah setengah jadi bila proses dihentikan."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


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

    Bila `ckpt` (Path) diberikan, keadaan lengkap (model, optimizer, scaler AMP, epoch, bobot terbaik, RNG)
    disimpan tiap akhir epoch; bila berkasnya sudah ada, training dilanjutkan dari epoch berikutnya."""
    tc = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    set_seed(seed)
    dev = data.device
    data.fit_norm(tr)
    model = build_model(cfg, data).to(dev)
    groups = model.param_groups(tc) if hasattr(model, "param_groups") else model.parameters()
    opt = torch.optim.Adam(groups, lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=tc["amp"] and dev.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tr_t = torch.as_tensor(np.array(tr))
    n_epochs = epochs if va is None else tc["max_epochs"]
    best, log, start = (-1.0, 0, None), [], 1
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        gen.set_state(st["gen"].cpu())
        _set_rng_state(st["rng"])
        best, log, start = st["best"], st["log"], st["epoch"] + 1
        print(f"  lanjut dari {ckpt.name} epoch {st['epoch']}")
    for ep in range(start, n_epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen)].to(dev)
        tot = 0.0
        for c in torch.split(perm, tc["batch_size"]):
            x, l, yb = data.batch(c)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = loss_fn(model(x, l), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(c)
        row = {"epoch": ep, "train_loss": tot / len(tr)}
        stop = False
        if va is not None:
            p = predict(model, data, va, tc["eval_batch"])
            yv = data.y[torch.as_tensor(np.array(va), device=dev)].cpu().numpy()
            row.update(compute_metrics(yv, (p >= 0.5).astype(int), p))
            if row["macro_f1"] > best[0]:
                best = (row["macro_f1"], ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
            elif ep - best[1] >= tc["patience"]:
                stop = True
        log.append(row)
        if stop:
            break
        if ckpt is not None and ep < n_epochs:
            st = {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                  "gen": gen.get_state(), "rng": _rng_state(), "best": best, "log": log}
            _atomic(ckpt, lambda t: torch.save(st, t))
    if va is not None:
        model.load_state_dict(best[2])
        return model, best[1], log
    return model, n_epochs, log


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
    data = make_data(cfg, feats, y, device)
    rows, preds, summaries = [], {"cv": [], "test": [], "ligo": []}, []

    def fit(tr, te, seed, va=None, epochs=None):
        def _fit(ckpt):
            model, ep, log = train_model(cfg, data, tr, seed, va=va, epochs=epochs, ckpt=ckpt)
            return make_pred(index, te, y, predict(model, data, te, tc["eval_batch"]), seed), {"epochs": ep, "history": log}
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
