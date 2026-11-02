"""Metrik, confusion matrix, analisis kesalahan per item/harakat, dan baris reports/results.csv."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from ta.paths import REPORTS_DIR, RUNS_DIR

RESULTS_CSV = REPORTS_DIR / "results.csv"
RESULT_COLUMNS = ["sifat", "model", "fitur", "mode", "seed", "fold", "subset", "n", "accuracy", "macro_f1",
                  "precision", "recall", "f1", "auc", "run_id"]
HARAKAT = {1: "fathah", 2: "kasrah", 3: "dhammah"}


def compute_metrics(y_true, y_pred, score=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc": float("nan"),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, score)
    return {k: (round(float(v), 6) if k != "n" else v) for k, v in out.items()}


def subset_metrics(pred: pd.DataFrame) -> dict:
    """Metrik untuk semua data, subset Data Benar, dan subset Data Salah."""
    res = {"all": compute_metrics(pred.y_true, pred.y_pred, pred.score)}
    for name, flag in (("benar", 1), ("salah", 0)):
        p = pred[pred.is_correct == flag]
        if len(p):
            res[name] = compute_metrics(p.y_true, p.y_pred, p.score)
    return res


def confusion(pred: pd.DataFrame) -> pd.DataFrame:
    cm = confusion_matrix(pred.y_true, pred.y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])


def plot_confusion(cm: pd.DataFrame, title: str, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm.values, cmap="Blues")
    for i in range(2):
        for j in range(2):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.values.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_tables(pred: pd.DataFrame):
    """Tingkat kesalahan per item, per harakat, dan per (item, harakat, kategori)."""
    p = pred.assign(error=(pred.y_true != pred.y_pred).astype(int),
                    harakat=pred["sub"].map(HARAKAT), kategori=np.where(pred.is_correct == 1, "benar", "salah"))
    by_item = p.groupby(["item", "huruf_no"]).agg(n=("error", "size"), error_rate=("error", "mean")).reset_index()
    by_harakat = p.groupby("harakat").agg(n=("error", "size"), error_rate=("error", "mean")).reset_index()
    by_cell = p.groupby(["item", "harakat", "kategori"]).agg(n=("error", "size"), error_rate=("error", "mean")).reset_index()
    return (by_item.sort_values("error_rate", ascending=False), by_harakat,
            by_cell.sort_values("error_rate", ascending=False))


def append_results(rows: list):
    """Tambahkan baris ke results.csv. Baris lama dengan run_id yang sama dibuang dulu (run yang dilanjutkan)."""
    df = pd.DataFrame(rows)[RESULT_COLUMNS]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV, dtype=str)
        dup = old.run_id.isin(set(df.run_id))
        if dup.any():
            old[~dup].to_csv(RESULTS_CSV, index=False)
    df.to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per (sifat, model, fitur), hanya dari run terakhir tiap kelompok.

    CV = mean +- std antar-fold (dan antar-seed bila ada beberapa seed); uji dan LIGO-pooled = mean +- std antar-seed
    (std kosong bila hanya satu seed)."""
    r = results[results.subset == "all"]
    keys = ["sifat", "model", "fitur"]
    last_run = r.groupby(keys).run_id.transform("last")
    r = r[r.run_id == last_run]
    cv = r[r["mode"] == "cv"].groupby(keys).macro_f1.agg(["mean", "std"])
    cv.columns = ["cv_macro_f1_mean", "cv_macro_f1_std"]
    parts = [cv]
    for name, sel in (("test", r["mode"] == "test"), ("ligo", (r["mode"] == "ligo") & (r.fold.astype(str) == "pooled"))):
        g = r[sel].groupby(keys)
        parts.append(g.macro_f1.agg(["mean", "std"]).set_axis([f"{name}_macro_f1", f"{name}_macro_f1_std"], axis=1))
        parts.append(g.auc.mean().rename(f"{name}_auc"))
        parts.append(g.seed.nunique().rename(f"{name}_n_seed") if name == "test" else None)
    return pd.concat([x for x in parts if x is not None], axis=1).sort_index().round(4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ringkas reports/results.csv")
    ap.parse_args()
    print(summarize(pd.read_csv(RESULTS_CSV)).to_string())
