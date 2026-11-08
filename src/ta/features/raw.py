"""Masukan Wav2Vec2-BERT: sinyal 16 kHz -> log-mel filterbank lewat feature extractor bawaan checkpoint.

Audio hasil praproses dipotong ke panjang tetap (fix_length, 1,9 s) lalu bagian valid-nya (tanpa padding nol)
diberikan ke SeamlessM4TFeatureExtractor. Extractor menghitung fbank 80 pita, menormalisasi per pita mel per
rekaman (statistik rekaman itu sendiri, bukan data lain, jadi aman dari kebocoran), mem-pad dengan
padding_value bawaan, lalu menumpuk 2 frame (160 dimensi). Hasilnya di-cache sekali ke data/interim/.
"""
import hashlib
import json

import numpy as np

from ta.data.preprocess import DEFAULT_PREPROCESS, TrimmedAudio, fix_length
from ta.paths import INTERIM_DIR, ROOT

DEFAULT_CHECKPOINT = "pretrained/facebook-w2v-bert-2.0"


def fbank_max_frames(n_samples, sr=16000, win_ms=25.0, hop_ms=10.0):
    """Jumlah frame fbank (snip edges ala Kaldi) untuk n_samples."""
    win, hop = int(sr * win_ms / 1000), int(sr * hop_ms / 1000)
    return 1 + (n_samples - win) // hop


def load_extractor(checkpoint=DEFAULT_CHECKPOINT):
    from transformers import AutoFeatureExtractor
    return AutoFeatureExtractor.from_pretrained(ROOT / checkpoint)


def extract_fbank(cfg, index, batch_size=256):
    """Mengembalikan (F float16 [N, T, 160], lengths [N] = frame valid) dengan urutan baris = urutan index."""
    pre = {**DEFAULT_PREPROCESS, **cfg.get("preprocess", {})}
    ckpt = cfg["model"].get("checkpoint", DEFAULT_CHECKPOINT)
    min_s = float(cfg["features"].get("min_seconds", 0.1))
    target_s = float(cfg["target_seconds"])
    key = hashlib.md5(json.dumps([pre, ckpt, target_s, min_s], sort_keys=True).encode()).hexdigest()[:10]
    d = INTERIM_DIR / "features" / f"w2vbert_fbank_{key}"
    sids = index["source_id"].to_numpy().astype(str)
    if not (d / "source_id.npy").exists():
        fe = load_extractor(ckpt)
        audio = TrimmedAudio(pre)
        n = int(round(target_s * pre["sr"]))
        n_min = int(round(min_s * pre["sr"]))
        max_frames = fbank_max_frames(n, pre["sr"])
        feats, lens = [], []
        for i in range(0, len(sids), batch_size):
            wavs = []
            for sid in sids[i:i + batch_size]:
                y, valid = fix_length(audio.get(sid), n)
                y = y[:valid]
                wavs.append(np.pad(y, (0, max(0, n_min - len(y)))))
            out = fe(wavs, sampling_rate=pre["sr"], padding="max_length", max_length=max_frames, truncation=True,
                     return_attention_mask=True, return_tensors="np")
            feats.append(out["input_features"].astype(np.float16))
            lens.append(out["attention_mask"].sum(1).astype(np.int32))
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "F.npy", np.concatenate(feats))
        np.save(d / "lengths.npy", np.concatenate(lens))
        np.save(d / "source_id.npy", sids)
        (d / "config.json").write_text(json.dumps({"checkpoint": ckpt, "preprocess": pre, "target_seconds": target_s,
                                                   "min_seconds": min_s, "max_fbank_frames": int(max_frames)}, indent=2))
    pos = {s: i for i, s in enumerate(np.load(d / "source_id.npy"))}
    missing = [s for s in sids if s not in pos]
    if missing:
        raise ValueError(f"{len(missing)} rekaman tidak ada di cache fbank {d}; hapus folder cache lalu ulangi")
    rows = np.array([pos[s] for s in sids])
    return np.asarray(np.load(d / "F.npy", mmap_mode="r")[rows]), np.load(d / "lengths.npy")[rows]
