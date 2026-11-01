"""CRNN: blok konvolusi (pool frekuensi lebih agresif dari waktu) lalu GRU dua arah dan rata-rata waktu bermasker.

GRU memakai pack_padded_sequence, jadi frame padding tidak diproses sama sekali.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ta.models.cnn import conv_block, downsample_lengths, masked_mean


class MelCRNN(nn.Module):
    def __init__(self, n_mels=64, channels=(32, 64, 128), rnn_hidden=128, rnn_layers=1, dropout=0.3, n_classes=2):
        super().__init__()
        pools = [(2, 2), (2, 2)] + [(2, 1)] * (len(channels) - 2)
        blocks, c = [], 1
        for ch, p in zip(channels, pools):
            blocks.append(conv_block(c, ch, p))
            c = ch
        self.features = nn.Sequential(*blocks)
        self.time_factor = 1
        f = n_mels
        for pf, pt in pools:
            f, self.time_factor = -(-f // pf), self.time_factor * pt
        self.proj = nn.Sequential(nn.Linear(c * f, rnn_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.rnn = nn.GRU(rnn_hidden, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * rnn_hidden, n_classes))

    def forward(self, x, lengths):
        """x: [B, n_mels, T] (sudah dinormalisasi, padding = 0), lengths: frame valid [B]."""
        h = self.features(x.unsqueeze(1))                  # [B, C, F', T']
        b, c, f, t = h.shape
        h = self.proj(h.permute(0, 3, 1, 2).reshape(b, t, c * f))
        lens = torch.clamp(downsample_lengths(lengths, self.time_factor), max=t)
        packed = pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=t)
        return self.head(masked_mean(out, lens))
