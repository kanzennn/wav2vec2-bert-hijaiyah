"""CNN untuk log-Mel: blok conv-BN-ReLU-pool, lalu rata-rata frekuensi dan rata-rata waktu bermasker.

Masker waktu membuat frame padding tidak ikut dirata-rata, jadi panjang padding tidak langsung jadi fitur.
"""
import torch
from torch import nn


def downsample_lengths(lengths, factor):
    """Jumlah frame valid setelah pooling waktu dengan faktor `factor` (dibulatkan ke atas, minimal 1)."""
    return torch.clamp((lengths + factor - 1) // factor, min=1)


def masked_mean(x, lengths):
    """x: [B, T, C], lengths: [B] -> [B, C], rata-rata hanya pada t < lengths."""
    t = torch.arange(x.shape[1], device=x.device)[None, :]
    m = (t < lengths[:, None]).to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1)


def conv_block(c_in, c_out, pool):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(pool, ceil_mode=True))


class MelCNN(nn.Module):
    def __init__(self, n_mels=64, channels=(32, 64, 128, 128), dropout=0.3, n_classes=2):
        super().__init__()
        blocks, c = [], 1
        for ch in channels:
            blocks.append(conv_block(c, ch, (2, 2)))
            c = ch
        self.features = nn.Sequential(*blocks)
        self.time_factor = 2 ** len(channels)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x, lengths):
        """x: [B, n_mels, T] (sudah dinormalisasi, padding = 0), lengths: frame valid [B]."""
        h = self.features(x.unsqueeze(1))          # [B, C, F', T']
        h = h.mean(2).transpose(1, 2)              # [B, T', C]
        h = masked_mean(h, downsample_lengths(lengths, self.time_factor))
        return self.head(h)
