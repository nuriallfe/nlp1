import numpy as np
from scipy.sparse import csr_matrix, vstack
from tokenizers import Tokenizer

from collections import Counter
import numpy as np


class Encoder:
    """Maps an input x to a D-dimensional vector (x is a text, or for PAWS a pair of texts)."""

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Implement me in a sub-class")

    def __call__(self, x) -> 'np.array':
        raise NotImplementedError("Implement me in a sub-class")


def encode_all(texts, encoder: Encoder):
    """
    Encode many inputs into ONE sparse (n x D) matrix.
    (Stacking dense vectors in a list would need several GB for PAWS pairs.)
    """
    return vstack([csr_matrix(encoder(t)) for t in texts], format="csr")


class TokenIndicator(Encoder):
    """Representation space 1: 1 if the token occurs in the text, 0 otherwise."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __call__(self, x: str) -> 'np.array':
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector


class BagOfWords(Encoder):
    """Representation space 2: how many times each token occurs in the text."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __call__(self, x: str) -> 'np.array':
        ids = self.tokenizer.encode(x).ids
        return np.bincount(ids, minlength=len(self)).astype(float)

class NgramBagOfWords:
    """Cuenta cuántas veces aparece cada n-grama de tokens BPE en una frase.

    n             : tamaño del n-grama (1 = BagOfWords normal, 2 = bigramas, ...)
    include_lower : si True usa todos los n-gramas de tamaño 1..n; si False solo tamaño n
    min_count     : descarta n-gramas que aparecen menos de esto en el corpus de ajuste
    max_features  : se queda con los max_features n-gramas más frecuentes
    """

    def __init__(self, tokenizer, n=2, include_lower=False, min_count=5, max_features=8000):
        self.tokenizer = tokenizer
        self.n = n
        self.include_lower = include_lower
        self.min_count = min_count
        self.max_features = max_features
        self.vocab = None  # dict: n-gram (tokens tuple) -> index

    def _ngrams(self, text):
        tokens = self.tokenizer.encode(text).tokens
        sizes = range(1, self.n + 1) if self.include_lower else [self.n]
        return [tuple(tokens[i:i + k]) for k in sizes for i in range(len(tokens) - k + 1)]

    def fit(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(self._ngrams(t))
        kept = [(g, c) for g, c in counts.items() if c >= self.min_count]
        kept.sort(key=lambda gc: -gc[1])
        kept = kept[:self.max_features]
        self.vocab = {g: i for i, (g, _) in enumerate(kept)}
        return self
    
    def __len__(self):
        return self.dim
    
    @property
    def dim(self):
        return len(self.vocab)

    def encode(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        for g in self._ngrams(text):
            j = self.vocab.get(g)
            if j is not None:        # the n-grams outside of vocab are ignored
                vec[j] += 1
        return vec

    __call__ = encode

class PairEncoder(Encoder):
    """
    Encodes a PAWS example x = (sentence1, sentence2).
    Each sentence is encoded with a base encoder (TokenIndicator or BagOfWords) into
    vectors a and b, which are then combined:

    mode="diff" (default): [|a - b|, a * b]
        |a - b| is non-zero for tokens that differ between the sentences,
        a * b is non-zero for tokens that both sentences share.
        Symmetric: swapping the two sentences gives the same vector.
    mode="concat": [a, b]  (each sentence separately, no direct comparison)
    """

    def __init__(self, base_encoder: Encoder, mode: str = "diff"):
        if mode not in ("diff", "concat"):
            raise ValueError(f"Unknown mode={mode}")
        self.base = base_encoder
        self.mode = mode

    def __len__(self) -> int:
        return 2 * len(self.base)

    def __call__(self, x) -> 'np.array':
        sentence1, sentence2 = x
        a, b = self.base(sentence1), self.base(sentence2)
        if self.mode == "concat":
            return np.concatenate([a, b])
        return np.concatenate([np.abs(a - b), a * b])
