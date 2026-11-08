"""Wav2Vec2-BERT (facebook/w2v-bert-2.0, dari folder lokal pretrained/) dengan kepala klasifikasi dua kelas.

Fine-tune seluruh lapisan. Representasi = rata-rata keluaran lapis terakhir pada frame valid (attention mask),
lalu dropout + linear ke 2 kelas. SpecAugment/masking bawaan dan layerdrop dimatikan karena baseline tanpa
augmentasi (masking rentang waktu masuk ablasi augmentasi). Gradient checkpointing dipakai supaya muat di 8 GB.
"""
import torch
from torch import nn

from ta.paths import ROOT

DEFAULT_CHECKPOINT = "pretrained/facebook-w2v-bert-2.0"


def masked_mean(x, mask):
    """x: [B, T, C], mask: [B, T] (1 = frame valid) -> [B, C]."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


def load_backbone(checkpoint=DEFAULT_CHECKPOINT, random_init_config=None):
    """random_init_config (dict) hanya untuk unit test: backbone kecil acak tanpa unduhan."""
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
    overrides = dict(layerdrop=0.0, apply_spec_augment=False, mask_time_prob=0.0)
    if random_init_config is not None:
        return Wav2Vec2BertModel(Wav2Vec2BertConfig(**{**random_init_config, **overrides}))
    return Wav2Vec2BertModel.from_pretrained(ROOT / checkpoint, **overrides)


class W2VBertClassifier(nn.Module):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, dropout=0.1, gradient_checkpointing=True, n_classes=2,
                 random_init_config=None):
        super().__init__()
        self.backbone = load_backbone(checkpoint, random_init_config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.config.hidden_size, n_classes))

    def forward(self, x, mask):
        """x: input_features [B, T, 160] dari feature extractor, mask: attention mask [B, T]."""
        h = self.backbone(input_features=x, attention_mask=mask).last_hidden_state
        return self.head(masked_mean(h, mask))

    def param_groups(self, tc):
        return [{"params": self.backbone.parameters(), "lr": tc["backbone_lr"]},
                {"params": self.head.parameters(), "lr": tc["lr"]}]
