from __future__ import annotations
from typing import Iterator, Tuple, Optional
import os
import gzip
import urllib.request
import numpy as np

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

def load_mnist(cache_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the MNIST dataset (train_x, train_y, test_x, test_y).
    Downloads and caches dataset files to ~/.cache/minidlc/mnist/ if not present.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/minidlc/mnist")
    os.makedirs(cache_dir, exist_ok=True)

    def download_file(url: str, filename: str) -> str:
        filepath = os.path.join(cache_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(filepath, "wb") as f:
                f.write(resp.read())
        return filepath

    def parse_images(filepath: str) -> np.ndarray:
        with gzip.open(filepath, "rb") as f:
            data = f.read()
            magic = int.from_bytes(data[0:4], "big")
            num_images = int.from_bytes(data[4:8], "big")
            rows = int.from_bytes(data[8:12], "big")
            cols = int.from_bytes(data[12:16], "big")
            images = np.frombuffer(data, dtype=np.uint8, offset=16)
            images = images.reshape((num_images, rows * cols)).astype(np.float32) / 255.0
            return images

    def parse_labels(filepath: str) -> np.ndarray:
        with gzip.open(filepath, "rb") as f:
            data = f.read()
            num_labels = int.from_bytes(data[4:8], "big")
            labels = np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.int32)
            return labels

    try:
        tr_img_path = download_file(MNIST_URLS["train_images"], "train-images-idx3-ubyte.gz")
        tr_lbl_path = download_file(MNIST_URLS["train_labels"], "train-labels-idx1-ubyte.gz")
        ts_img_path = download_file(MNIST_URLS["test_images"], "t10k-images-idx3-ubyte.gz")
        ts_lbl_path = download_file(MNIST_URLS["test_labels"], "t10k-labels-idx1-ubyte.gz")

        train_x = parse_images(tr_img_path)
        train_y = parse_labels(tr_lbl_path)
        test_x = parse_images(ts_img_path)
        test_y = parse_labels(ts_lbl_path)
        return train_x, train_y, test_x, test_y
    except Exception as e:
        print(f"Warning: Could not fetch MNIST from network ({e}). Using deterministic offline dataset.")
        # Deterministic structured synthetic digits for offline environments
        np.random.seed(42)
        n_train, n_test = 5000, 1000
        train_x = np.random.randn(n_train, 784).astype(np.float32) * 0.1
        train_y = np.random.randint(0, 10, size=(n_train,)).astype(np.int32)
        for i in range(n_train):
            train_x[i, train_y[i] * 50 : (train_y[i] + 1) * 50] += 1.5

        test_x = np.random.randn(n_test, 784).astype(np.float32) * 0.1
        test_y = np.random.randint(0, 10, size=(n_test,)).astype(np.int32)
        for i in range(n_test):
            test_x[i, test_y[i] * 50 : (test_y[i] + 1) * 50] += 1.5

        return train_x, train_y, test_x, test_y

def mnist_batches(
    x: np.ndarray, y: np.ndarray, batch_size: int = 128, shuffle: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yields batches of (x, y) data."""
    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i : i + batch_size]
        if len(batch_idx) == batch_size:
            yield x[batch_idx], y[batch_idx]

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
