# model.py
# 07.04.2026 12:08 AM GMT+4.00
# Nikhil Kapila

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        self.din = din
        self.dout = dout

        self.wk = nn.Linear(din, dout)
        self.wq = nn.Linear(din, dout)
        self.wv = nn.Linear(din, dout)

    def forward(self, x):
        # x is N, T, din
        w_q = self.wq(x)  # N, T, dout
        w_k = self.wk(x)  # N, T, dout
        w_v = self.wv(x)  # N, T, dout

        # N,T,T = N, T, dout @ N, dout, T
        att_weights = (w_q @ w_k.transpose(1, 2)) / (self.dout**0.5)
        scores = F.softmax(att_weights, dim=-1)  # N, T, T

        # N,T,dout = N,T,T @ N,T,dout
        enriched_vector = scores @ w_v

        return enriched_vector
