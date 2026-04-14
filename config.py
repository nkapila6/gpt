# config.py
# 14.04.2026 09:45 AM GMT+4.00
# Nikhil Kapila

from dataclasses import dataclass


@dataclass
class Config:
    emb_size: int = 768  # embedding size
    n_layers: int = 12  # number of transformer blocks
    num_heads: int = 12  # number of heads for multihead att
    seq_len: int = 512  # context length
    vocab_size: int = 500  # vocab size, TODO: base it on required
    dropout: float = 0.5  # dropout
    out_proj: int = 3072  # linear projection
