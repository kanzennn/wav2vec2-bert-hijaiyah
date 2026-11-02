"""Sinyal mentah 16 kHz untuk wav2vec 2.0.

Audio hasil praproses di-pad/potong ke panjang tetap, lalu dinormalisasi per rekaman (zero-mean, unit-variance
pada bagian valid saja, seperti do_normalize pada feature extractor wav2vec2-base). Normalisasi per rekaman
tidak memakai statistik data lain, jadi aman dari kebocoran. Padding tetap nol.
"""
import numpy as np

from ta.data.preprocess import DEFAULT_PREPROCESS, TrimmedAudio, fix_length


def normalize_utterance(y, valid):
    v = y[:valid]
    out = np.zeros_like(y)
    out[:valid] = (v - v.mean()) / np.sqrt(v.var() + 1e-7)
    return out


def load_raw_audio(cfg, index):
    """Mengembalikan (W float16 [N, n_sampel], valid_samples [N]) dengan urutan baris = urutan index."""
    pre = {**DEFAULT_PREPROCESS, **cfg.get("preprocess", {})}
    n = int(round(float(cfg["target_seconds"]) * pre["sr"]))
    audio = TrimmedAudio(pre)
    W = np.empty((len(index), n), dtype=np.float16)
    valid = np.empty(len(index), dtype=np.int64)
    for i, sid in enumerate(index["source_id"]):
        y, valid[i] = fix_length(audio.get(sid), n)
        W[i] = normalize_utterance(y, valid[i])
    return W, valid
