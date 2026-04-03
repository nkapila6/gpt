# tokenizer.py
# 03.04.2026 06:53 PM GMT+4.00
# Nikhil Kapila
# word-level tokenizer

import re
from utils import read_data


def word_level_tokenizer(path: str) -> tuple[list, dict, dict]:
    data = read_data(path)

    tokens = sorted(set(re.findall(r"\w+|[^\w\s]", data)))

    encoder = {word: idx for idx, word in enumerate(tokens)}
    decoder = {idx: word for idx, word in enumerate(tokens)}

    encoded = [encoder[t] for t in re.findall(r"\w+|[^\w\s]", data)]

    return encoded, encoder, decoder
