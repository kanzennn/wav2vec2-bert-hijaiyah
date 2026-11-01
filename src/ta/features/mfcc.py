"""Ringkasan MFCC untuk model klasik: 13 statis + delta + delta-delta, diringkas mean + std (78 dimensi).

Jendela 25 ms, geser 10 ms. Statistik dihitung hanya pada bagian valid (tanpa padding nol), supaya
panjang padding tidak ikut menjadi fitur.
"""
import hashlib
import json

import librosa
import numpy as np

from ta.data.preprocess import DEFAULT_PREPROCESS, TrimmedAudio, fix_length
from ta.paths import INTERIM_DIR

DEFAULT_MFCC = dict(n_mfcc=13, n_fft=512, win_ms=25.0, hop_ms=10.0, n_mels=40, fmin=20.0, fmax=8000.0,
                    delta_width=9, min_seconds=0.1)


def feature_names(n_mfcc=13):
    base = [f"{p}mfcc{i}" for p in ("", "d_", "dd_") for i in range(1, n_mfcc + 1)]
    return [f"{b}_mean" for b in base] + [f"{b}_std" for b in base]


def mfcc_summary(y, sr, p):
    """y: audio valid (sudah dipraproses). Mengembalikan vektor 6 * n_mfcc."""
    min_len = int(round(p["min_seconds"] * sr))
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=p["n_mfcc"], n_fft=p["n_fft"],
                             win_length=int(round(sr * p["win_ms"] / 1000)), hop_length=int(round(sr * p["hop_ms"] / 1000)),
                             n_mels=p["n_mels"], fmin=p["fmin"], fmax=p["fmax"])
    d1 = librosa.feature.delta(m, width=p["delta_width"], order=1, mode="nearest")
    d2 = librosa.feature.delta(m, width=p["delta_width"], order=2, mode="nearest")
    f = np.vstack([m, d1, d2])
    return np.concatenate([f.mean(axis=1), f.std(axis=1)]).astype(np.float32)


def _params(cfg):
    pre = {**DEFAULT_PREPROCESS, **cfg.get("preprocess", {})}
    feat = {**DEFAULT_MFCC, **{k: v for k, v in cfg["features"].items() if k != "type"}}
    return pre, feat, float(cfg["target_seconds"])


def load_mfcc_features(cfg, index):
    pre, feat, target_s = _params(cfg)
    key = hashlib.md5(json.dumps([pre, feat, target_s], sort_keys=True).encode()).hexdigest()[:10]
    path = INTERIM_DIR / "features" / f"mfcc_{key}.npz"
    names = feature_names(feat["n_mfcc"])
    if path.exists():
        z = np.load(path, allow_pickle=False)
        pos = {s: i for i, s in enumerate(z["source_id"])}
        missing = [s for s in index["source_id"] if s not in pos]
        if not missing:
            return z["X"][[pos[s] for s in index["source_id"]]], names
    audio = TrimmedAudio(pre)
    n = int(round(target_s * pre["sr"]))
    X = np.empty((len(index), len(names)), dtype=np.float32)
    for i, sid in enumerate(index["source_id"]):
        y, valid = fix_length(audio.get(sid), n)
        X[i] = mfcc_summary(y[:valid], pre["sr"], feat)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, source_id=index["source_id"].to_numpy().astype(str))
    (path.with_suffix(".json")).write_text(json.dumps({"preprocess": pre, "features": feat, "target_seconds": target_s}, indent=2))
    return X, names
