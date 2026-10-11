from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np

def synthetic_mnist_batches(
    batch_size: int = 128, steps: int = 10, num_classes: int = 10, features: int = 784
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generates synthetic MNIST batches (x: [N, 784], y: [N]) for testing."""
    for _ in range(steps):
        x = np.random.randn(batch_size, features).astype(np.float32)
        y = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
        yield x, y

class CharTokenizer:
    """Character-level tokenizer for TinyShakespeare."""
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in s], dtype=np.int32)

    def decode(self, indices: np.ndarray) -> str:
        return "".join([self.itos[int(i)] for i in indices])
