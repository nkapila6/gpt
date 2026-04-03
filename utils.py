# utils.py
# 03.04.2026 07:41 PM GMT+4.00
# Nikhil Kapila


def read_data(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
