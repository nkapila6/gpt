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

    @property
    def word_to_idx(self):
        return self.vocab.copy()

    @property
    def idx_to_word(self):
        return self.inv_vocab.copy()

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


class Tokenizer:
    # simple word level tokenizer

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.enc = vocab.word_to_idx  # word to idx
        self.dec = vocab.idx_to_word  # idx to word

    def encode(self, text: str) -> list[int]:
        # encoder converts incoming str text to list of token ids
        preprocessed = re.split(r'([,.:;?_!"()\']|--|--|\s)', text)
        preprocessed = [w for w in preprocessed if w and not w.isspace()]
        return [self.enc.get(w, self.enc["<unk>"]) for w in preprocessed]

    def decode(self, ids: list[int]) -> str:
        # decoder converts list of token ids to str text
        text = " ".join([self.dec.get(id, "<unk>") for id in ids])
        return re.sub(r'\s([,.:;?_!"()\']|--)', r"\1", text)


def word_level_tokenizer(path: str) -> tuple[list, dict, dict]:
    data = read_data(path)

    words = re.split(r'([,.:;?_!"()\']|--|\s)', data)  # fix: consistent split logic
    words = [w for w in words if w and not w.isspace()]

    tokens = sorted(set(words))

    encoder = {word: idx for idx, word in enumerate(tokens)}
    decoder = {idx: word for idx, word in enumerate(tokens)}

    encoded = [encoder[t] for t in words]

    return encoded, encoder, decoder
