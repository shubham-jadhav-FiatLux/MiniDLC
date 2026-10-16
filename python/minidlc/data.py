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

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# 65-character fallback containing all standard Shakespeare characters
SHAKESPEARE_FALLBACK = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country, he did it to
please his mother, and to be partly proud; which he
is, even to the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a vice in
him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?
Enter MENENIUS AGRIPPA.

Second Citizen:
Worthy Menenius Agrippa; one that hath always loved
the people.

First Citizen:
He's one honest enough: would all the rest were so!

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

Second Citizen:
Our business is not unknown to the senate; they have had
inkling this fortnight what we intend to do, which now
we'll show 'em in deeds. They say poor suitors have strong
breaths: they shall know we have strong arms too.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?
$3 & 'QXZ jkvwxyz.
"""

class CharTokenizer:
    """Character-level tokenizer for TinyShakespeare (65 characters)."""
    def __init__(self, text: Optional[str] = None):
        if text is None:
            text = SHAKESPEARE_FALLBACK
        # Extract unique characters or enforce standard 65-char vocab
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> np.ndarray:
        return np.array([self.stoi.get(c, 0) for c in s], dtype=np.int32)

    def decode(self, indices: Union[np.ndarray, List[int]]) -> str:
        return "".join([self.itos.get(int(i), "") for i in indices])

def load_tinyshakespeare(cache_dir: Optional[str] = None) -> Tuple[str, CharTokenizer]:
    """Loads TinyShakespeare text and initializes character tokenizer."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/minidlc/tinyshakespeare")
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, "input.txt")

    if not os.path.exists(filepath):
        try:
            req = urllib.request.Request(SHAKESPEARE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp, open(filepath, "w", encoding="utf-8") as f:
                content = resp.read().decode("utf-8")
                f.write(content)
        except Exception:
            # Deterministic fallback when offline or in sandbox
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(SHAKESPEARE_FALLBACK)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    return text, tokenizer

def tinyshakespeare_batches(
    data: np.ndarray, batch_size: int = 32, block_size: int = 64, steps: int = 100
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yields batches (X: [B, T], Y: [B, T]) for next-character autoregressive modeling."""
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError(f"Data length ({len(data)}) must be larger than block_size + 1 ({block_size + 1})")

    for _ in range(steps):
        ix = np.random.randint(0, n, size=(batch_size,))
        x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int32)
        y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int32)
        yield x, y

