"""wav2vec 2.0 (facebook/wav2vec2-base, dari folder lokal pretrained/).

Dua mode:
- beku: backbone tidak dilatih. Keluaran ke-13 lapis (embedding CNN + 12 transformer) dirata-rata di frame valid
  dan di-cache sekali (extract_embeddings). Yang dilatih hanya W2VHead: bobot softmax antar-lapis + MLP kecil.
- fine-tune: W2VFineTune melatih transformer + kepala; encoder CNN (feature extractor) tetap beku.
SpecAugment bawaan checkpoint dimatikan (apply_spec_augment=False) karena baseline tanpa augmentasi.
"""
import hashlib
import json

import numpy as np
import torch
from torch import nn

from ta.models.cnn import masked_mean
from ta.paths import INTERIM_DIR, ROOT

DEFAULT_CHECKPOINT = "pretrained/facebook-wav2vec2-base"


def load_backbone(path=DEFAULT_CHECKPOINT):
    from transformers import Wav2Vec2Model
    return Wav2Vec2Model.from_pretrained(ROOT / path, apply_spec_augment=False, layerdrop=0.0)


def frame_lengths(backbone, valid_samples):
    """Jumlah frame keluaran yang berasal dari sampel valid (minimal 1)."""
    return torch.clamp(backbone._get_feat_extract_output_lengths(valid_samples), min=1)


@torch.no_grad()
def extract_embeddings(cfg, index, W, valid, batch_size=64):
    """Cache rata-rata bermasker tiap lapis: [N, 13, 768] float16. Hanya dari audio, tanpa label."""
    ckpt = cfg["model"].get("checkpoint", DEFAULT_CHECKPOINT)
    key = hashlib.md5(json.dumps([cfg.get("preprocess", {}), cfg["target_seconds"], ckpt], sort_keys=True).encode()).hexdigest()[:10]
    d = INTERIM_DIR / "features" / f"w2v_{key}"
    sids = index["source_id"].to_numpy().astype(str)
    if (d / "source_id.npy").exists():
        pos = {s: i for i, s in enumerate(np.load(d / "source_id.npy"))}
        if all(s in pos for s in sids):
            return np.load(d / "E.npy", mmap_mode="r")[[pos[s] for s in sids]]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = load_backbone(ckpt).to(dev).eval()
    out = []
    for i in range(0, len(W), batch_size):
        x = torch.from_numpy(W[i:i + batch_size].astype(np.float32)).to(dev)
        lens = frame_lengths(bb, torch.from_numpy(valid[i:i + batch_size]).to(dev))
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            hs = bb(x, output_hidden_states=True).hidden_states
        out.append(torch.stack([masked_mean(h.float(), lens) for h in hs], 1).half().cpu().numpy())
    E = np.concatenate(out)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "E.npy", E)
    np.save(d / "source_id.npy", sids)
    (d / "config.json").write_text(json.dumps({"checkpoint": ckpt, "preprocess": cfg.get("preprocess", {}),
                                               "target_seconds": cfg["target_seconds"]}, indent=2))
    return E


class W2VHead(nn.Module):
    """Kepala mode beku: x [B, L, D] (embedding per lapis, sudah dinormalisasi) -> logits."""

    def __init__(self, n_layers=13, dim=768, hidden=256, dropout=0.3, n_classes=2):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x, lengths=None):
        w = torch.softmax(self.layer_w, 0)
        return self.mlp((x * w[None, :, None]).sum(1))


class W2VFineTune(nn.Module):
    """Fine-tune: transformer dilatih, encoder CNN beku. Pooling = rata-rata bermasker lapis terakhir."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, hidden=256, dropout=0.3, n_classes=2):
        super().__init__()
        self.backbone = load_backbone(checkpoint)
        self.backbone.feature_extractor._freeze_parameters()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.config.hidden_size, hidden),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x, valid_samples):
        h = self.backbone(x).last_hidden_state
        return self.head(masked_mean(h, frame_lengths(self.backbone, valid_samples)))

    def param_groups(self, tc):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        return [{"params": bb, "lr": tc["backbone_lr"]}, {"params": self.head.parameters(), "lr": tc["lr"]}]
