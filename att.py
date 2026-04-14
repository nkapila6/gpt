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
        weight_matrix = (w_q @ w_k.transpose(1, 2)) / (self.dout**0.5)
        att_weights = F.softmax(weight_matrix, dim=-1)  # N, T, T

        # N,T,dout = N,T,T @ N,T,dout
        enriched_vector = att_weights @ w_v

        return enriched_vector


# masked self attention
class CausalAttention(nn.Module):
    def __init__(self, din, dout, max_len):
        super().__init__()
        self.din, self.dout = din, dout
        self.wk, self.wq, self.wv = (
            nn.Linear(din, dout),
            nn.Linear(din, dout),
            nn.Linear(din, dout),
        )

        # since causality needs lower trial of matrix, we init a persistent var in the buff that takes lower trial of torch.ones
        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.full((max_len, max_len), -torch.inf), diagonal=1),
        )

    def forward(self, x):
        # x is of shape N, T, din where N=batch, T=seqlen, din=input embedding size
        T = x.size(1)
        # projecting to get k,q,v
        k, q, v = self.wk(x), self.wq(x), self.wv(x)  # each of shape N, T, dout

        att_matrix = (q @ k.nT) / (self.dout**0.5)  # N,T,T = N,T,dout @ N,dout,T
        # atp, we need to apply the mask, ill use torch.where to replace the 0s with -inf for softmax saturation
        att_matrix += self.mask[:T, :T]
        weights = F.softmax(att_matrix, dim=-1)  # N,T,T

        enriched_vector = weights @ v
        return enriched_vector


# Multi Head Att Wrapper by iterating over Masked Self Att Block
class MultiHeadAttWrapper(nn.Module):
    def __init__(self, din, dout, max_len, num_heads):
        super().__init__()
        self.din, self.dout, self.max_len, self.num_heads = (
            din,
            dout,
            max_len,
            num_heads,
        )
        self.heads = [CausalAttention(din, dout, max_len) for _ in range(num_heads)]

    def forward(self, x):
        # x is N, T, din
        context_vectors = [head(x) for head in self.heads]

        # each head returns shappe N, T, dout.. we need N, T, num_heads * dout
        return torch.concat(context_vectors, dim=-1)


class BetterMultiHeadAttWrapper(nn.Module):
    def __init__(self, din, dout, max_len, num_heads):
        super().__init__()

        self.din, self.dout, self.max_len, self.num_heads = (
            din,
            dout,
            max_len,
            num_heads,
        )

        self.heads = nn.ModuleList(
            [CausalAttention(din, dout, max_len) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAtt(nn.Module):
    # H is num_heads and T is seq_len
    def __init__(self, din, dout, H, T):
        super().__init__()

        assert dout % H == 0, "dout must be divisible by number of heads."
        self.head_dim = dout // H
        self.num_heads, self.T = H, T
        self.dout = dout

        self.q, self.k, self.v = (
            nn.Linear(din, dout),
            nn.Linear(din, dout),
            nn.Linear(din, dout),
        )

        self.out_projection = nn.Linear(dout, dout)

        self.register_buffer(
            "mask", torch.triu(torch.full((T, T), -torch.inf), diagonal=1)
        )

    # N, T, E (din)
    def forward(self, x):
        N, T, _ = x.shape
        if T > self.T:
            raise ValueError("seq len exceeds configured max len")

        # each of q,k,v is not N,T,dout
        # preparing q,k,v from the input x
        q, k, v = self.q(x), self.k(x), self.v(x)

        # reshaping to split across multiple heads, change from N,T,dout to N,T,H,E
        # transposing each to form N,H,T,E
        q, k, v = (
            q.view(N, T, self.num_heads, self.head_dim).transpose(1, 2),
            k.view(N, T, self.num_heads, self.head_dim).transpose(1, 2),
            v.view(N, T, self.num_heads, self.head_dim).transpose(1, 2),
        )

        att_matrix = (q @ k.transpose(2, 3)) / (self.head_dim**0.5)  # NHTE @ NHET=NHTT
        att_matrix += self.mask[:T, :T]  # NHTT
        weights = F.softmax(att_matrix, dim=-1)  # NHTT

        # v is NHTE
        context_vec = weights @ v  # NHTT@NHTE-> NHTE
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.reshape(N, T, self.dout)
        context_vec = self.out_projection(context_vec)

        return context_vec
