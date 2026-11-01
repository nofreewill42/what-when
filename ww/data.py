"""PG-19 character data: memory-mapped uint8 files, a deterministic paired data order, validation windows.

Data order: the training windows for update u depend only on (data_seed, u), never on the arm or the
micro-batch size, so every arm with the same seed sees exactly the same characters in the same order.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS_FILE = os.path.join(ROOT, "configs", "paths.json")


def load_paths() -> dict:
    with open(PATHS_FILE) as f:
        paths = json.load(f)
    for key in ("WW_TRAIN", "WW_VAL"):  # environment variables override the file
        if os.environ.get(key):
            paths[key.split("_")[1].lower()] = os.environ[key]
    return paths


class CharData:
    def __init__(self, path: str, vocab: int = 128):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} (edit configs/paths.json or set WW_TRAIN / WW_VAL)")
        self.path = path
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.n = len(self.data)
        probe = np.asarray(self.data[: min(self.n, 1 << 20)])
        if probe.max() >= vocab:
            raise ValueError(f"{path}: byte {probe.max()} >= vocab {vocab}; is this the normalized file?")


def train_offsets(n: int, seq_len: int, batch: int, data_seed: int, update: int) -> np.ndarray:
    rng = np.random.default_rng([data_seed, update])
    return rng.integers(0, n - seq_len - 1, size=batch)


def windows(data: CharData, offsets, seq_len: int) -> torch.Tensor:
    """[len(offsets), seq_len + 1] int64 (inputs = [:, :-1], targets = [:, 1:])."""
    out = np.empty((len(offsets), seq_len + 1), dtype=np.int64)
    for i, o in enumerate(offsets):
        out[i] = data.data[o:o + seq_len + 1]
    return torch.from_numpy(out)


def val_offsets(n: int, seq_len: int, max_windows: int | None = None, max_chars: int | None = None):
    """Non-overlapping windows covering the validation stream from its start (every character after
    the first is a target exactly once). Truncated to max_windows or to the first max_chars."""
    limit = n if max_chars is None else min(n, max_chars)
    offs = np.arange(0, limit - seq_len - 1 + 1, seq_len, dtype=np.int64)
    if max_windows is not None:
        offs = offs[:max_windows]
    return offs
