"""Tes komponen Wav2Vec2-BERT: fbank dari extractor bawaan, masker padding, dan bentuk keluaran."""
import numpy as np
import pytest
import torch

from ta.features.raw import fbank_max_frames
from ta.models.wav2vec_bert import W2VBertClassifier, masked_mean
from ta.paths import ROOT
from ta.train_dl import FbankData

TINY = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, output_hidden_size=32,
            feature_projection_input_dim=160, conv_depthwise_kernel_size=3)


def test_fbank_max_frames():
    assert fbank_max_frames(30400) == 188


def test_masked_mean_ignores_padding():
    x = torch.tensor([[[1.0], [3.0], [100.0]]])
    assert masked_mean(x, torch.tensor([[1, 1, 0]])).item() == pytest.approx(2.0)


def test_fbankdata_mask():
    F = np.zeros((3, 5, 160), dtype=np.float16)
    d = FbankData(F, np.array([5, 2, 1]), np.zeros(3), torch.device("cpu"))
    _, m, _ = d.batch(torch.tensor([0, 1, 2]))
    assert m.tolist() == [[1] * 5, [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]


def test_classifier_shape_and_padding_invariance():
    """Isi frame padding (di luar attention mask) tidak boleh mengubah keluaran."""
    torch.manual_seed(0)
    m = W2VBertClassifier(random_init_config=TINY, gradient_checkpointing=False).eval()
    x = torch.randn(2, 20, 160)
    mask = torch.ones(2, 20, dtype=torch.long)
    mask[1, 12:] = 0
    x2 = x.clone()
    x2[1, 12:] = torch.randn(8, 160) * 10
    with torch.no_grad():
        a, b = m(x, mask), m(x2, mask)
    assert a.shape == (2, 2)
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.skipif(not (ROOT / "pretrained/facebook-w2v-bert-2.0/preprocessor_config.json").exists(),
                    reason="checkpoint lokal belum ada")
def test_extractor_normalizes_per_utterance_and_pads():
    from ta.features.raw import load_extractor
    fe = load_extractor()
    rng = np.random.default_rng(0)
    short, long_ = rng.normal(size=4000).astype(np.float32) * 0.1, rng.normal(size=16000).astype(np.float32) * 0.1
    out = fe([short, long_], sampling_rate=16000, padding="max_length", max_length=188, truncation=True,
             return_attention_mask=True, return_tensors="np")
    assert out["input_features"].shape == (2, 94, 160)
    assert out["attention_mask"].sum(1).tolist()[0] < out["attention_mask"].sum(1).tolist()[1]
    # fitur rekaman pendek tidak berubah bila diproses sendirian (normalisasi tidak bocor antar-rekaman)
    alone = fe([short], sampling_rate=16000, padding="max_length", max_length=188, truncation=True,
               return_attention_mask=True, return_tensors="np")
    assert np.allclose(alone["input_features"][0], out["input_features"][0])
