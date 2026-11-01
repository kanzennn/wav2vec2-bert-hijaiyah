"""Praproses (sama untuk semua model): mono -> 16 kHz -> trim hening (ambang energi) -> normalisasi puncak.

Hasil trim (panjang variabel) di-cache sekali ke data/interim/. Pad/potong ke panjang tetap dilakukan
belakangan (fix_length), supaya panjang target bisa diputuskan dari distribusi durasi data latih.
"""
import hashlib
import json

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

from ta.paths import INTERIM_DIR

DEFAULT_PREPROCESS = dict(
    sr=16000,
    trim_top_db=40.0,      # frame dianggap ucapan bila RMS > puncak - 40 dB
    frame_ms=25.0,
    hop_ms=10.0,
    margin_ms=50.0,        # sisakan 50 ms sebelum/sesudah ucapan supaya onset/release tidak terpotong
    peak=0.95,
)


def load_audio(path):
    """Baca WAV sebagai float32 mono (sama dengan cara dataset menghitung md5)."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def resample(y, sr, target_sr):
    if sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)


def trim_silence(y, sr, top_db, frame_ms, hop_ms, margin_ms):
    """Potong hening di awal/akhir berdasarkan RMS per frame relatif ke frame terkeras."""
    frame = int(round(sr * frame_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    if len(y) < frame:
        return y, 0, len(y)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=False)[0]
    db = 20 * np.log10(rms + 1e-10)
    voiced = np.flatnonzero(db > db.max() - top_db)
    margin = int(round(sr * margin_ms / 1000))
    start = max(0, voiced[0] * hop - margin)
    end = min(len(y), voiced[-1] * hop + frame + margin)
    return y[start:end], start, end


def normalize_peak(y, peak):
    m = np.abs(y).max() if len(y) else 0.0
    return y if m < 1e-8 else (y * (peak / m)).astype(np.float32)


def fix_length(y, n):
    """Pad nol di kanan atau potong di tengah ke n sampel. Mengembalikan (y_fixed, panjang_valid)."""
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n], n
    return np.pad(y, (0, n - len(y))), len(y)


def preprocess_one(path, cfg):
    raw, sr = load_audio(path)
    stats = {
        "md5_samples": hashlib.md5(raw.tobytes()).hexdigest(),
        "sr_read": sr,
        "n_raw": len(raw),
        "dur_raw": len(raw) / sr,
        "peak_raw": float(np.abs(raw).max()) if len(raw) else 0.0,
        "rms_dbfs_raw": float(20 * np.log10(np.sqrt(np.mean(raw.astype(np.float64) ** 2)) + 1e-10)) if len(raw) else -200.0,
        "clip_frac": float(np.mean(np.abs(raw) >= 0.999)) if len(raw) else 0.0,
    }
    with open(path, "rb") as fh:
        stats["md5_file"] = hashlib.md5(fh.read()).hexdigest()
    y = resample(raw, sr, cfg["sr"])
    y, s, e = trim_silence(y, cfg["sr"], cfg["trim_top_db"], cfg["frame_ms"], cfg["hop_ms"], cfg["margin_ms"])
    y = normalize_peak(y, cfg["peak"])
    stats.update(trim_start=s, trim_end=e, dur_trim=len(y) / cfg["sr"])
    return y.astype(np.float32), stats


def cache_key(cfg) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def cache_dir(cfg):
    return INTERIM_DIR / f"trim16k_{cache_key(cfg)}"


def build_cache(index: pd.DataFrame, cfg=None, n_jobs=-2, force=False):
    """Praproses semua rekaman di index dan simpan audio flat + offset + statistik."""
    cfg = {**DEFAULT_PREPROCESS, **(cfg or {})}
    out = cache_dir(cfg)
    if (out / "stats.csv").exists() and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)
    results = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(preprocess_one)(p, cfg) for p in index["path"])
    lengths = np.array([len(y) for y, _ in results], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    np.save(out / "audio.npy", np.concatenate([y for y, _ in results]))
    stats = pd.DataFrame([s for _, s in results])
    stats.insert(0, "source_id", index["source_id"].to_numpy())
    stats["offset"], stats["length"] = offsets, lengths
    stats.to_csv(out / "stats.csv", index=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return out


class TrimmedAudio:
    """Akses audio hasil cache per source_id."""

    def __init__(self, cfg=None):
        cfg = {**DEFAULT_PREPROCESS, **(cfg or {})}
        d = cache_dir(cfg)
        if not (d / "stats.csv").exists():
            raise FileNotFoundError(f"cache praproses belum ada: {d} (jalankan python -m ta.data.audit)")
        self.sr = cfg["sr"]
        self.audio = np.load(d / "audio.npy", mmap_mode="r")
        self.stats = pd.read_csv(d / "stats.csv").set_index("source_id")

    def get(self, source_id):
        r = self.stats.loc[source_id]
        return np.asarray(self.audio[r.offset:r.offset + r.length], dtype=np.float32)
