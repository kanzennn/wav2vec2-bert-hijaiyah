"""Tes opsi sensitivitas E2: buang 26 berkas `-error` dari data latih, sel hiperparameter dikunci per sifat."""
from pathlib import Path

import yaml

from ta.data.index import build_index, keep_without_flag_error
from ta.train import task_grid

ROOT = Path(__file__).resolve().parents[1]


def test_keep_without_flag_error_drops_only_26_train_rows():
    idx = build_index()
    keep = keep_without_flag_error(idx)
    dropped = idx[~keep]
    assert len(dropped) == 26
    assert set(dropped.split) == {"latih"}
    assert dropped.flag_error.astype(bool).all()
    assert set(dropped.responden) == {"benar-15"}
    # data uji utuh
    assert keep[idx.split == "uji"].all()


def test_task_grid_uses_locked_params_per_task():
    cfg = yaml.safe_load(open(ROOT / "configs/xgb_mfcc_ext_noerror.yaml", encoding="utf-8"))
    for task, n_est in (("S1", 600), ("S3", 600), ("T2", 1000)):
        g = task_grid(cfg, task)
        assert g == [{"subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": n_est,
                      "max_depth": 8, "learning_rate": 0.05}]


def test_task_grid_without_lock_is_full_grid():
    cfg = yaml.safe_load(open(ROOT / "configs/xgb_mfcc_ext.yaml", encoding="utf-8"))
    assert len(task_grid(cfg, "S1")) == 27
    assert len(task_grid(cfg, "S1", smoke=True)) == 1
