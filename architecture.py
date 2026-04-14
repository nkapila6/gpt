# architecture.py
# 14.04.2026 09:31 AM GMT+4.00
# Nikhil Kapila

import torch
import torch.nn as nn
import torch.nn.functional as F

from att import MultiHeadAtt
from config import Config


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(cfg.emb_size)

        self.first = nn.Sequential(
            MultiHeadAtt(cfg.emb_size, cfg.emb_size, cfg.num_heads, cfg.seq_len),
            nn.Dropout(cfg.dropout),
        )

        self.att_layernorm = nn.LayerNorm(cfg.emb_size)

        self.second = nn.Sequential(
            nn.Linear(cfg.emb_size, cfg.out_proj),
            nn.GELU(),
            nn.Linear(cfg.out_proj, cfg.emb_size),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        shortcut = x
        x = self.input_layernorm(x)
        x = self.att_layernorm(self.first(x) + shortcut)

        shortcut = x
        x = self.second(x)
        return shortcut + x


# Overall architecture
class GPT1(nn.Module):
    def __init__(self, cfg: Config):
        # expect a dict with named values, better than long arg list
        super().__init__()

        # cfg
        self.cfg = cfg

        # input
        self.in_token = nn.Embedding(cfg.vocab_size, cfg.emb_size)
        self.in_position = nn.Embedding(cfg.seq_len, cfg.emb_size)
        self.in_dropout = nn.Dropout(cfg.dropout)

        # transformer blocks, 12
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # layer norm
        self.out_layernorm = nn.LayerNorm(cfg.emb_size)
        self.out_head = nn.Linear(cfg.emb_size, cfg.vocab_size)

    def forward(self, x):
        # input x is just the tokens which is seq len of token ids
        N, T = x.shape
        in_embeddings = self.in_token(x)

        pos_embeds = self.in_position(torch.arange(T, device=x.device))
        x = in_embeddings + pos_embeds
        x = self.in_dropout(x)
        x = self.transformer_blocks(x)
        x = self.out_layernorm(x)
        x = self.out_head(x)  # logits here now!!
        return x

    @torch.no_grad()
    def inference(self, x, max_new_tokens):
        # x is N, T.
        for _ in range(max_new_tokens):
            x1 = x[:, -self.cfg.seq_len :]

            logits = self(x1)

            logits = logits[:, -1, :]  # N,T,din we only need last T
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1, keepdim=True)  # greedy decode
            x = torch.cat((x, pred), dim=1)

        return x  # returns token ids for tokenizer decode
