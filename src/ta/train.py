"""Training: python -m ta.train --config configs/xgb_mfcc.yaml --task S1 [--smoke] [--resume RUN_DIR]

Baseline XGBoost dijalankan di sini; Wav2Vec2-BERT diteruskan ke ta.train_dl.

Alur per sifat (XGBoost):
1. Grid hiperparameter dinilai dengan validasi silang fold 1-5 (split latih); pilih macro-F1 rata-rata tertinggi.
2. Model terbaik dilatih ulang di seluruh split latih, diuji SEKALI di split uji (fold 0).
3. Leave-item-group-out dengan hiperparameter yang sama (tanpa tuning ulang).
Run --smoke memakai subset kecil dan TIDAK menulis ke reports/results.csv.
"""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier

from ta.data.index import build_index
from ta.data.labels import TASKS, get_labels
from ta.data.preprocess import DEFAULT_PREPROCESS, build_cache
from ta.data.split import cv_folds, ligo_folds, train_test_indices
from ta.evaluate import (append_results, compute_metrics, confusion, error_tables, plot_confusion,
                         subset_metrics)
from ta.features import load_features
from ta.models.xgboost_model import decision_scores, expand_grid, make_model
from ta.paths import ROOT, RUNS_DIR
from ta.utils import detect_device, git_info, set_seed

DEEP_MODELS = ("w2v_bert",)
PRED_COLUMNS = ["source_id", "category", "is_correct", "responden", "item", "huruf_no", "sub", "rep", "split", "fold"]


def fit_predict(kind, params, seed, X, y, tr, te):
    model = make_model(kind, params, seed)
    model.fit(X[tr], y[tr])
    return model.predict(X[te]), decision_scores(model, X[te])


def make_pred(index, te, y, y_pred, score):
    p = index.loc[te, PRED_COLUMNS].copy()
    p["y_true"], p["y_pred"], p["score"] = y[te], y_pred, score
    return p.reset_index(drop=True)


def smoke_subset(index, frac, seed):
    """Ambil sebagian rekaman per (split, fold, category) supaya semua fold tetap terwakili."""
    keep = index.groupby(["split", "fold", "category"], group_keys=False).sample(frac=frac, random_state=seed)
    return index.loc[sorted(keep.index)].reset_index(drop=True)


def result_rows(task, cfg, mode, fold, metrics_by_subset, run_id):
    base = {"sifat": task, "model": cfg["model"]["type"], "fitur": cfg["features"]["type"], "mode": mode,
            "seed": cfg["seed"], "fold": fold, "run_id": run_id}
    return [{**base, "subset": s, **m} for s, m in metrics_by_subset.items()]


def run_task(task, cfg, index, X, run_dir, smoke, run_id):
    kind, seed, n_jobs = cfg["model"]["type"], cfg["seed"], cfg.get("n_jobs", 4)
    y = get_labels(task, index)
    fixed = cfg["model"].get("fixed", {})
    grid = [{**fixed, **g} for g in expand_grid(cfg["model"]["grid"])]
    if smoke:
        grid = grid[: cfg.get("smoke", {}).get("max_grid", 1)]
    folds = list(cv_folds(index))

    # 1. validasi silang untuk memilih hiperparameter
    t0 = time.time()
    jobs = [(gi, f, tr, va) for gi in range(len(grid)) for f, tr, va in folds]
    outs = Parallel(n_jobs=n_jobs)(delayed(fit_predict)(kind, grid[gi], seed, X, y, tr, va) for gi, f, tr, va in jobs)
    cv_rows = []
    for (gi, f, tr, va), (yp, sc) in zip(jobs, outs):
        cv_rows.append({"grid_id": gi, **{f"param_{k}": v for k, v in grid[gi].items()}, "fold": f,
                        **compute_metrics(y[va], yp, sc)})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(run_dir / f"{task}_cv_grid.csv", index=False)
    mean = cv.groupby("grid_id").macro_f1.mean()
    best_id = int(mean.idxmax())
    best = grid[best_id]
    print(f"[{task}] CV {len(jobs)} fit dalam {time.time() - t0:.0f}s; terbaik {best} macro-F1 {mean[best_id]:.4f}")

    rows = []
    best_cv = []
    for (gi, f, tr, va), (yp, sc) in zip(jobs, outs):
        if gi == best_id:
            p = make_pred(index, va, y, yp, sc)
            best_cv.append(p)
            rows += result_rows(task, cfg, "cv", f, subset_metrics(p), run_id)
    pd.concat(best_cv).to_csv(run_dir / f"{task}_predictions_cv.csv", index=False)

    # 2. uji akhir (fold 0), sekali
    tr, te = train_test_indices(index)
    yp, sc = fit_predict(kind, best, seed, X, y, tr, te)
    pred = make_pred(index, te, y, yp, sc)
    pred.to_csv(run_dir / f"{task}_predictions_test.csv", index=False)
    test_m = subset_metrics(pred)
    rows += result_rows(task, cfg, "test", 0, test_m, run_id)
    cm = confusion(pred)
    cm.to_csv(run_dir / f"{task}_cm_test.csv")
    plot_confusion(cm, f"{task} {kind}+{cfg['features']['type']} (uji)", run_dir / f"{task}_cm_test.png")
    for name, tab in zip(("by_item", "by_harakat", "by_item_harakat_kategori"), error_tables(pred)):
        tab.round(4).to_csv(run_dir / f"{task}_errors_test_{name}.csv", index=False)

    # baseline kelas mayoritas (cek kewajaran)
    dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    dm = compute_metrics(y[te], dummy.predict(X[te]))
    rows += [{**r, "model": "majority"} for r in result_rows(task, cfg, "test", 0, {"all": dm}, run_id)]

    # 3. leave-item-group-out
    ligo_preds = []
    for k, tr_k, te_k in ligo_folds(index):
        if len(te_k) == 0:
            continue
        yp, sc = fit_predict(kind, best, seed, X, y, tr_k, te_k)
        p = make_pred(index, te_k, y, yp, sc).assign(ligo_fold=k)
        ligo_preds.append(p)
        rows += result_rows(task, cfg, "ligo", k, subset_metrics(p), run_id)
    ligo = pd.concat(ligo_preds)
    ligo.to_csv(run_dir / f"{task}_predictions_ligo.csv", index=False)
    ligo_m = subset_metrics(ligo)
    rows += result_rows(task, cfg, "ligo", "pooled", ligo_m, run_id)
    confusion(ligo).to_csv(run_dir / f"{task}_cm_ligo.csv")

    summary = {"task": task, "best_params": best, "cv_macro_f1_mean": float(mean[best_id]),
               "cv_macro_f1_std": float(cv[cv.grid_id == best_id].macro_f1.std()),
               "test": test_m, "test_majority": dm, "ligo_pooled": ligo_m}
    print(f"[{task}] uji macro-F1 {test_m['all']['macro_f1']:.4f} (benar {test_m['benar']['macro_f1']:.4f}, "
          f"salah {test_m['salah']['macro_f1']:.4f}); mayoritas {dm['macro_f1']:.4f}; LIGO {ligo_m['all']['macro_f1']:.4f}")
    return rows, summary


def load_deep_features(cfg, index):
    """Fitur model deep learning untuk semua rekaman di index (tuple array, baris = urutan index)."""
    kind = cfg["features"]["type"]
    if kind == "w2v_bert_fbank":
        from ta.features.raw import extract_fbank
        return extract_fbank(cfg, index)
    raise ValueError(f"fitur deep learning belum didukung: {kind}")


def load_resume(args):
    """Baca run_dir, run_args.json, dan config.yaml dari run yang mau dilanjutkan; tolak bila tidak cocok."""
    run_dir = Path(args.resume).resolve()
    if not run_dir.is_dir():
        run_dir = (RUNS_DIR / args.resume).resolve()
    if not (run_dir / "run_args.json").exists():
        raise SystemExit(f"{run_dir} tidak bisa dilanjutkan (run_args.json tidak ada; run dibuat sebelum fitur resume)")
    saved = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(open(run_dir / "config.yaml", encoding="utf-8"))
    if cfg["model"]["type"] not in DEEP_MODELS:
        raise SystemExit("--resume hanya untuk model deep learning")
    if args.config and yaml.safe_load(open(args.config, encoding="utf-8")) != cfg:
        raise SystemExit(f"config {args.config} berbeda dengan {run_dir / 'config.yaml'}; resume ditolak")
    if args.task and args.task != saved["task"]:
        raise SystemExit(f"--task {args.task} berbeda dengan run aslinya ({saved['task']})")
    if args.smoke and not saved["smoke"]:
        raise SystemExit("--smoke berbeda dengan run aslinya")
    return run_dir, saved, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--task", choices=[*TASKS, "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", metavar="RUN_DIR", help="lanjutkan run deep learning yang terhenti")
    args = ap.parse_args()

    git = git_info()
    if args.resume:
        run_dir, saved, cfg = load_resume(args)
        args.task, args.smoke = saved["task"], saved["smoke"]
        run_id = run_dir.name
        resumes = saved.get("resumes", []) + [{"time": f"{datetime.now():%Y%m%d-%H%M%S}", "git": git}]
        (run_dir / "run_args.json").write_text(json.dumps({**saved, "resumes": resumes}, indent=2), encoding="utf-8")
        git = saved["git"]
        print(f"lanjut {run_id}; git saat ini {resumes[-1]['git']['commit'][:8]}")
    else:
        if not args.config:
            ap.error("--config wajib (kecuali dengan --resume)")
        args.task = args.task or "all"
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['name']}_{args.task}{'_smoke' if args.smoke else ''}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True)
        shutil.copy(args.config, run_dir / "config.yaml")
        (run_dir / "run_args.json").write_text(json.dumps({"task": args.task, "smoke": args.smoke, "git": git}, indent=2),
                                               encoding="utf-8")
    set_seed(cfg["seed"])
    tasks = list(TASKS) if args.task == "all" else [args.task]

    index = build_index()
    build_cache(index, {**DEFAULT_PREPROCESS, **cfg.get("preprocess", {})}, n_jobs=cfg.get("n_jobs", 4))
    # fitur selalu dihitung untuk semua rekaman (cache lengkap), baru diambil subsetnya untuk --smoke
    deep = cfg["model"]["type"] in DEEP_MODELS
    if deep:
        from ta.train_dl import run_task_dl
        feats = load_deep_features(cfg, index)
        X, names = feats[0], [cfg["features"]["type"]]
    else:
        X, names = load_features(cfg, index)
    if args.smoke:
        sub = smoke_subset(index, cfg.get("smoke", {}).get("fraction", 0.1), cfg["seed"])
        rows = index.reset_index(drop=True).index[index.source_id.isin(set(sub.source_id))]
        index, X = sub, X[rows]
        if deep:
            feats = tuple(f[rows] for f in feats)
    print(f"run {run_id}: {len(index)} rekaman, fitur {X.shape[1:]}, device {detect_device()}, git {git['commit'][:8]}{' (dirty)' if git['dirty'] else ''}")

    all_rows, summaries = [], []
    for t in tasks:
        if deep:
            rows, s = run_task_dl(t, cfg, index, feats, run_dir, args.smoke, run_id)
        else:
            rows, s = run_task(t, cfg, index, X, run_dir, args.smoke, run_id)
        all_rows += rows
        summaries.append(s)

    pd.DataFrame(all_rows).to_csv(run_dir / "results_rows.csv", index=False)
    meta = {"run_id": run_id, "config": cfg, "git": git, "smoke": args.smoke, "n_recordings": len(index),
            "feature_names": names, "device": detect_device(), "summaries": summaries}
    (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    if not args.smoke:
        append_results(all_rows)
    print(f"selesai -> {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
