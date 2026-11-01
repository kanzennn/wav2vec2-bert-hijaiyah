"""Log-Mel spectrogram (dB) untuk CNN dan CRNN.

Jendela 25 ms, geser 10 ms, 64 pita Mel. Audio di-pad/potong ke panjang tetap (fix_length), jadi semua
spectrogram berukuran sama (64 x 191 untuk 1,9 s). Jumlah frame valid (bukan padding) ikut disimpan supaya
model bisa mengabaikan bagian padding saat pooling dan normalisasi.
Normalisasi per pita TIDAK dilakukan di sini; dihitung dari data latih tiap fold di train_dl.py.
"""
import hashlib
import json

import librosa
import numpy as np
from joblib import Parallel, delayed

from ta.data.preprocess import DEFAULT_PREPROCESS, TrimmedAudio, fix_length
from ta.paths import INTERIM_DIR

DEFAULT_MEL = dict(n_mels=64, n_fft=512, win_ms=25.0, hop_ms=10.0, fmin=20.0, fmax=8000.0, amin=1e-10)


def log_mel(y, sr, p):
    """y: audio panjang tetap. Mengembalikan log-Mel (n_mels x frame) dalam dB relatif amplitudo 1."""
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=p["n_fft"], win_length=int(round(sr * p["win_ms"] / 1000)),
                                       hop_length=int(round(sr * p["hop_ms"] / 1000)), n_mels=p["n_mels"],
                                       fmin=p["fmin"], fmax=p["fmax"], power=2.0)
    return librosa.power_to_db(m, ref=1.0, amin=p["amin"], top_db=None).astype(np.float32)


def valid_frames(valid_samples, hop):
    """Frame yang pusatnya masih di dalam audio valid (center=True), minimal 1."""
    return max(1, valid_samples // hop + 1)


def _params(cfg):
    pre = {**DEFAULT_PREPROCESS, **cfg.get("preprocess", {})}
    feat = {**DEFAULT_MEL, **{k: v for k, v in cfg["features"].items() if k != "type"}}
    return pre, feat, float(cfg["target_seconds"])


def _compute_chunk(sids, pre, feat, n):
    audio = TrimmedAudio(pre)
    hop = int(round(pre["sr"] * feat["hop_ms"] / 1000))
    out, vf = [], []
    for sid in sids:
        y, valid = fix_length(audio.get(sid), n)
        out.append(log_mel(y, pre["sr"], feat).astype(np.float16))
        vf.append(valid_frames(valid, hop))
    return np.stack(out), np.array(vf, dtype=np.int32)


def load_mel_features(cfg, index, n_jobs=8):
    """Mengembalikan (X float16 [N, n_mels, T], valid_frames [N]) dengan urutan baris = urutan index."""
    pre, feat, target_s = _params(cfg)
    key = hashlib.md5(json.dumps([pre, feat, target_s], sort_keys=True).encode()).hexdigest()[:10]
    d = INTERIM_DIR / "features" / f"mel_{key}"
    if not (d / "source_id.npy").exists():
        d.mkdir(parents=True, exist_ok=True)
        sids = index["source_id"].to_numpy().astype(str)
        n = int(round(target_s * pre["sr"]))
        chunks = np.array_split(sids, max(1, len(sids) // 500))
        res = Parallel(n_jobs=n_jobs)(delayed(_compute_chunk)(c, pre, feat, n) for c in chunks)
        np.save(d / "X.npy", np.concatenate([r[0] for r in res]))
        np.save(d / "valid_frames.npy", np.concatenate([r[1] for r in res]))
        np.save(d / "source_id.npy", sids)
        (d / "config.json").write_text(json.dumps({"preprocess": pre, "features": feat, "target_seconds": target_s}, indent=2))
    sids = np.load(d / "source_id.npy")
    pos = {s: i for i, s in enumerate(sids)}
    missing = [s for s in index["source_id"] if s not in pos]
    if missing:
        raise ValueError(f"{len(missing)} rekaman tidak ada di cache Mel {d}; hapus folder cache lalu ulangi")
    rows = np.array([pos[s] for s in index["source_id"]])
    X = np.load(d / "X.npy", mmap_mode="r")
    return np.asarray(X[rows]), np.load(d / "valid_frames.npy")[rows]
