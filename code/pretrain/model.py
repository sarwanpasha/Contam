"""Small BERT-style single-cell foundation model (Geneformer-like MLM).

Fully specified and deterministic so that matched pretraining pairs differ ONLY
in whether the target benchmark is present in the pretraining corpus.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SmallSCFM(nn.Module):
    def __init__(self, vocab_size=25426, d_model=256, n_layers=6, n_heads=8,
                 d_ff=1024, max_len=2048, dropout=0.1, pad_id=0, mask_id=1):
        super().__init__()
        self.pad_id, self.mask_id, self.vocab_size = pad_id, mask_id, vocab_size
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=dropout, batch_first=True,
                    norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, ids):
        pad = ids.eq(self.pad_id)
        p = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.drop(self.tok(ids) + self.pos(p))
        h = self.enc(h, src_key_padding_mask=pad)
        return self.norm(h), pad

    def forward(self, ids, labels=None):
        h, pad = self.encode(ids)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def cell_embedding(self, ids):
        h, pad = self.encode(ids)
        keep = (~pad).unsqueeze(-1).float()
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
