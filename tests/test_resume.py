"""Tes fitur jeda/lanjut: baris results.csv tidak dobel dan checkpoint melanjutkan training dengan hasil sama."""
import numpy as np
import pandas as pd
import torch

from ta import evaluate
from ta.train_dl import FbankData, train_model

TINY = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, output_hidden_size=32,
            feature_projection_input_dim=160, conv_depthwise_kernel_size=3)


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
    """Latih 4 epoch tanpa jeda vs berhenti setelah epoch 2 (checkpoint) lalu lanjut: bobot akhir sama (CPU)."""
    rng = np.random.default_rng(0)
    F = rng.normal(size=(48, 10, 160)).astype(np.float16)
    y = (F[:, 0, 0] > 0).astype(int)
    data = FbankData(F, rng.integers(3, 11, size=48), y, torch.device("cpu"))
    cfg = {"model": {"type": "w2v_bert", "params": {"random_init_config": TINY, "gradient_checkpointing": False}},
           "train": {"batch_size": 8, "grad_accum": 2, "optimizer": "adamw", "amp_dtype": "none"}}
    tr = np.arange(40)
    full, _, log_full = train_model(cfg, data, tr, seed=1, epochs=4)
    ck = tmp_path / "u.pt"
    train_model(cfg, data, tr, seed=1, epochs=3, ckpt=ck)   # checkpoint terakhir = epoch 2
    assert torch.load(ck, weights_only=False)["epoch"] == 2
    resumed, _, log_res = train_model(cfg, data, tr, seed=1, epochs=4, ckpt=ck)
    assert [r["epoch"] for r in log_res] == [1, 2, 3, 4]
    assert [r["train_loss"] for r in log_res] == [r["train_loss"] for r in log_full]
    for k, v in full.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k])


def test_improvement_per_fold():
    rows = []
    for f, xm, wm in (("1", 0.6, [0.7, 0.8]), ("2", 0.5, [0.6, 0.6])):
        rows.append({"sifat": "S1", "model": "xgb", "mode": "cv", "fold": f, "subset": "all", "seed": 42,
                     "macro_f1": xm, "accuracy": xm, "run_id": "x"})
        for s, w in zip((42, 43), wm):
            rows.append({"sifat": "S1", "model": "w2v_bert", "mode": "cv", "fold": f, "subset": "all", "seed": s,
                         "macro_f1": w, "accuracy": w, "run_id": "w"})
    imp = evaluate.improvement(pd.DataFrame(rows)).iloc[0]
    assert imp["n_fold"] == 2
    assert abs(imp["delta_macro_f1"] - 0.125) < 1e-9          # (0.15 + 0.10) / 2
