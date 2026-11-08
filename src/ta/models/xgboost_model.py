"""Baseline XGBoost di atas ringkasan MFCC. Model pohon tidak butuh normalisasi fitur.

Tiap model memakai 1 thread karena paralelisme ada di level fold/grid (joblib di train.py).
"""
import itertools


def make_model(kind: str, params: dict, seed: int):
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist", device="cpu",
                             objective="binary:logistic", eval_metric="logloss", **params)
    raise ValueError(f"model klasik belum didukung: {kind}")


def decision_scores(model, X):
    """Skor kontinu untuk AUC (probabilitas kelas 1)."""
    return model.predict_proba(X)[:, 1]


def expand_grid(grid: dict):
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
