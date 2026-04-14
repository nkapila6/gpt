# train.py
# 14.04.2026 01:08 PM GMT+4.00
# Nikhil Kapila

import torch
import torch.nn as nn

from architecture import GPT1
from config import Config
from loader import make_loaders
from tokenizer import Tokenizer, Vocab
from utils import read_data

# constant
batch_size = 64
n_epochs = 10

# using mps
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"device: {device}")

# load data
data = read_data("./data/input.txt")
vocab = Vocab(data)
tokenizer = Tokenizer(vocab)
encoded = tokenizer.encode(data)

# config, only need to change vocab size since i have a custom barebones tokenizer
print(f"Vocab length is {len(vocab)}")
cfg = Config(vocab_size=len(vocab))

# splitting stuff, 90-10 split
print("making splits")
split = int(0.9 * len(encoded))
train_tokens, val_tokens = encoded[:split], encoded[split:]

print("making data loaders")
train_loader, val_loader = make_loaders(
    train_tokens, val_tokens, cfg.seq_len, batch_size
)
print(f"total batches in train_loader: {len(train_loader)}")
print(f"total batches in val_loader: {len(val_loader)}")

# create model, optimizer
print("instantiating model..")
gpt = GPT1(cfg).to(device)  # using default params defined in config.py
opt = torch.optim.Adam(gpt.parameters(), lr=1e-5)  # random lr
criterion = nn.CrossEntropyLoss()  # standard loss func to align probab dist

# training loop
print("entering training loop")
for epoch in range(1, n_epochs + 1):
    gpt.train()
    train_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = gpt(x)
        loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        opt.step()

        train_loss += loss.item()
        print(f"epoch {epoch} | batch {i} | train: {loss}")

    # val phase
    gpt.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = gpt(x)
            val_loss += criterion(logits.view(-1, cfg.vocab_size), y.view(-1)).item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(
        f"epoch {epoch:>2}/{n_epochs}  |  train loss: {avg_train:.4f}  |  val loss: {avg_val:.4f}"
    )

# saving
print("saving model")
torch.save(gpt.state_dict(), "gpt_model.pt")
print("model saved to gpt_model.pt")

# testing inference
print("testing inference")
prompt = "First Citizen"
prompt_ids = tokenizer.encode(prompt)
x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
out = gpt.inference(x, max_new_tokens=50)
print(tokenizer.decode(out[0].tolist()))
