# tokenizer.py
# 03.04.2026 06:53 PM GMT+4.00
# Nikhil Kapila
# word-level tokenizer

import re

from utils import read_data


class Vocab:
    def __init__(self, text: str):
        self.special_tokens = ["<unk>", "<bos>", "<eos>"]

        words = re.split(r'([,.:;?_!"()\']|--|--|\s)', text)
        words = [w for w in words if w and not w.isspace()]

        self.tokens = sorted(set(words) - set(self.special_tokens))
        self.tokens.extend(self.special_tokens)  # adding special tokens

        # adding vocab
        self.vocab = {w: id for id, w in enumerate(self.tokens)}
        self.inv_vocab = {id: w for id, w in enumerate(self.tokens)}
        self.vocab_len = len(self.tokens)

    def __len__(self):
        # length
        return self.vocab_len

    def update(self, text: str):
        # update with new data
        words = re.split(r'([,.:;?_!"()\']|--|--|\s)', text)
        words = [w for w in words if w and not w.isspace()]
        new_words = sorted(set(words) - set(self.tokens))

        self.tokens.extend(new_words)
        self.vocab = {w: idx for idx, w in enumerate(self.tokens)}
        self.inv_vocab = {id: w for id, w in enumerate(self.tokens)}
        self.vocab_len = len(self.tokens)


def word_level_tokenizer(path: str) -> tuple[list, dict, dict]:
    data = read_data(path)

    tokens = sorted(set(re.findall(r"\w+|[^\w\s]", data)))

    encoder = {word: idx for idx, word in enumerate(tokens)}
    decoder = {idx: word for idx, word in enumerate(tokens)}

    encoded = [encoder[t] for t in re.findall(r"\w+|[^\w\s]", data)]

    return encoded, encoder, decoder
