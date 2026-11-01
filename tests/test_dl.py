"""Tes komponen CNN/CRNN: bentuk keluaran, masker padding, dan normalisasi yang hanya memakai data latih."""
import inspect

import numpy as np
import pytest
import torch

from ta.features.melspec import log_mel, valid_frames
from ta.models.cnn import MelCNN, masked_mean
from ta.models.crnn import MelCRNN
from ta.train_dl import MelData


def test_log_mel_uses_audio_only():
    assert list(inspect.signature(log_mel).parameters) == ["y", "sr", "p"]


def test_valid_frames():
    assert valid_frames(0, 160) == 1
    assert valid_frames(30400, 160) == 191


def test_masked_mean_ignores_padding():
    x = torch.tensor([[[1.0], [3.0], [100.0]]])
    assert masked_mean(x, torch.tensor([2])).item() == pytest.approx(2.0)


@pytest.mark.parametrize("cls", [MelCNN, MelCRNN])
def test_model_output_shape(cls):
    torch.manual_seed(0)
    m = cls(n_mels=64).eval()
    out = m(torch.randn(3, 64, 191), torch.tensor([191, 50, 1]))
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()


def test_crnn_ignores_frames_after_length():
    """GRU dipack, jadi isi setelah frame valid (setelah downsample) tidak boleh mengubah keluaran."""
    torch.manual_seed(0)
    m = MelCRNN(n_mels=64).eval()
    x = torch.randn(1, 64, 191)
    x2 = x.clone()
    x2[:, :, 120:] = torch.randn(1, 64, 71)
    with torch.no_grad():
        a, b = m(x, torch.tensor([100])), m(x2, torch.tensor([100]))
    assert torch.allclose(a, b, atol=1e-5)


def test_meldata_norm_from_train_only_and_padding_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4, 10)).astype(np.float16)
    X[3:] += 50                                    # "data uji" sangat berbeda
    vf = np.array([10, 5, 10, 10, 10, 10])
    d = MelData(X, vf, np.zeros(6), torch.device("cpu"))
    d.fit_norm([0, 1, 2])
    assert d.mean.abs().max() < 5                  # statistik tidak tercemar baris 3-5
    x, _, _ = d.batch(torch.tensor([1]))
    assert (x[0, :, 5:] == 0).all()
