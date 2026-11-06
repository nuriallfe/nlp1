from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, vstack
from tokenizers import Tokenizer


class Encoder:
    def __len__(self) -> int:
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


def encode_all(texts, encoder: Encoder):
    """Encode inputs as CSR. Exploit sparse_items when available."""
    if not hasattr(encoder, "sparse_items"):
        return vstack([csr_matrix(encoder(t)) for t in texts], format="csr")
    indptr = [0]
    indices = []
    data = []
    for text in texts:
        items = encoder.sparse_items(text)
        for col, value in sorted(items.items()):
            if value:
                indices.append(col)
                data.append(float(value))
        indptr.append(len(indices))
    return csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(indptr) - 1, len(encoder)),
        dtype=np.float32,
    )


class TokenIndicator(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def sparse_items(self, text):
        return {i: 1.0 for i in set(self.tokenizer.encode(text).ids)}

    def __call__(self, text):
        vec = np.zeros(len(self), dtype=np.float32)
        for idx in self.sparse_items(text):
            vec[idx] = 1.0
        return vec


class BagOfWords(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def sparse_items(self, text):
        return Counter(self.tokenizer.encode(text).ids)

    def __call__(self, text):
        return np.bincount(
            self.tokenizer.encode(text).ids, minlength=len(self)
        ).astype(np.float32)


class NgramBagOfWords(Encoder):
    def __init__(self, tokenizer, n=2, include_lower=False,
                 min_count=5, max_features=8000):
        self.tokenizer = tokenizer
        self.n = n
        self.include_lower = include_lower
        self.min_count = min_count
        self.max_features = max_features
        self.vocab = None

    def _ngrams(self, text):
        tokens = self.tokenizer.encode(text).tokens
        sizes = range(1, self.n + 1) if self.include_lower else [self.n]
        return [tuple(tokens[i:i + k])
                for k in sizes for i in range(len(tokens) - k + 1)]

    def fit(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(self._ngrams(text))
        kept = [(g, c) for g, c in counts.items() if c >= self.min_count]
        # Stable ordering in case of ties
        kept.sort(key=lambda x: (-x[1], x[0]))
        self.vocab = {g: i for i, (g, _) in enumerate(kept[:self.max_features])}
        return self

    @property
    def dim(self):
        if self.vocab is None:
            raise ValueError("NgramBagOfWords.fit() must be called first")
        return len(self.vocab)

    def __len__(self):
        return self.dim

    def sparse_items(self, text):
        if self.vocab is None:
            raise ValueError("NgramBagOfWords.fit() must be called first")
        result = Counter()
        for gram in self._ngrams(text):
            idx = self.vocab.get(gram)
            if idx is not None:
                result[idx] += 1
        return result

    def encode(self, text):
        vec = np.zeros(len(self), dtype=np.float32)
        for idx, count in self.sparse_items(text).items():
            vec[idx] = count
        return vec

    __call__ = encode


class PairEncoder(Encoder):
    """Pair features: diff=[|a-b|, a*b], concat=[a,b]."""
    def __init__(self, base_encoder, mode="diff"):
        if mode not in {"diff", "concat"}:
            raise ValueError(f"Unknown mode={mode}")
        self.base = base_encoder
        self.mode = mode

    def __len__(self):
        return 2 * len(self.base)

    def sparse_items(self, x):
        a = self.base.sparse_items(x[0])
        b = self.base.sparse_items(x[1])
        offset = len(self.base)
        if self.mode == "concat":
            return {**a, **{i + offset: value for i, value in b.items()}}
        result = {}
        for i in a.keys() | b.keys():
            v = abs(a.get(i, 0) - b.get(i, 0))
            if v:
                result[i] = v
        for i in a.keys() & b.keys():
            v = a[i] * b[i]
            if v:
                result[i + offset] = v
        return result

    def __call__(self, x):
        a, b = self.base(x[0]), self.base(x[1])
        if self.mode == "concat":
            return np.concatenate([a, b])
        return np.concatenate([np.abs(a - b), a * b])
