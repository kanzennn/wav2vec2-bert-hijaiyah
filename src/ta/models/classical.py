"""Model klasik. Normalisasi (StandardScaler) ada di dalam pipeline, jadi selalu di-fit di data latih saja.

RF dan XGBoost tidak butuh normalisasi; tiap model memakai 1 thread karena paralelisme ada di level fold/grid.
"""
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model(kind: str, params: dict, seed: int):
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed, **params))
    if kind == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist", device="cpu",
                             objective="binary:logistic", eval_metric="logloss", **params)
    raise ValueError(f"model klasik belum didukung: {kind}")


def decision_scores(model, X):
    """Skor kontinu untuk AUC (makin besar = makin condong ke kelas 1)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def expand_grid(grid: dict):
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
