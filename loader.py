# loader.py
# 03.04.2026 07:27 PM GMT+4.00
# Nikhil Kapila
# dataset construction

import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, tokens: list[int], seq_len: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def make_loaders(
    train_tokens: list[int],
    val_tokens: list[int],
    seq_len: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
