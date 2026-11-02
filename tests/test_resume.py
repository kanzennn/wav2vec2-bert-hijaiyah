"""Tes fitur jeda/lanjut: baris results.csv tidak dobel dan checkpoint melanjutkan training dengan hasil sama."""
import numpy as np
import pandas as pd
import torch

from ta import evaluate
from ta.train_dl import EmbData, train_model


def test_append_results_replaces_same_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS_CSV", tmp_path / "results.csv")
    row = {c: 0 for c in evaluate.RESULT_COLUMNS}
    evaluate.append_results([{**row, "run_id": "lama"}])
    evaluate.append_results([{**row, "run_id": "r1", "macro_f1": 0.1}])
    evaluate.append_results([{**row, "run_id": "r1", "macro_f1": 0.2}, {**row, "run_id": "r1", "macro_f1": 0.3}])
    df = pd.read_csv(tmp_path / "results.csv")
    assert list(df.run_id) == ["lama", "r1", "r1"]
    assert list(df.macro_f1) == [0, 0.2, 0.3]


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    """Latih 4 epoch tanpa jeda vs 2 epoch (checkpoint) lalu lanjut ke 4: bobot akhir harus sama (CPU)."""
    rng = np.random.default_rng(0)
    E = rng.normal(size=(200, 3, 16)).astype(np.float16)
    y = (E[:, 0, 0] > 0).astype(int)
    data = EmbData(E, y, torch.device("cpu"))
    cfg = {"model": {"type": "w2v_frozen", "params": {"hidden": 8}}, "train": {"batch_size": 32, "amp": False}}
    tr = np.arange(150)
    full, _, log_full = train_model(cfg, data, tr, seed=1, epochs=4)
    ck = tmp_path / "u.pt"
    train_model(cfg, data, tr, seed=1, epochs=4, ckpt=ck)          # menulis checkpoint epoch 1-3
    st = torch.load(ck, weights_only=False)
    assert st["epoch"] == 3
    # mundurkan ke checkpoint epoch 2 dengan melatih ulang 2 epoch saja
    ck2 = tmp_path / "v.pt"
    train_model(cfg, data, tr, seed=1, epochs=3, ckpt=ck2)
    assert torch.load(ck2, weights_only=False)["epoch"] == 2
    resumed, _, log_res = train_model(cfg, data, tr, seed=1, epochs=4, ckpt=ck2)
    assert [r["epoch"] for r in log_res] == [1, 2, 3, 4]
    assert [r["train_loss"] for r in log_res] == [r["train_loss"] for r in log_full]
    for k, v in full.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k])
