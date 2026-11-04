import numpy as np
from scipy.sparse import csr_matrix, vstack
from tokenizers import Tokenizer



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



class PairEncoder(Encoder):
    """
    Encodes a PAWS example x = (sentence1, sentence2).
    Each sentence is encoded with a base encoder (TokenIndicator or BagOfWords)
    and the two vectors are concatenated: [enc(sentence1), enc(sentence2)].
    """

    def __init__(self, base_encoder: Encoder):
        self.base = base_encoder

    def __len__(self) -> int:
        return 2 * len(self.base)

    def __call__(self, x) -> 'np.array':
        sentence1, sentence2 = x
        return np.concatenate([self.base(sentence1), self.base(sentence2)])
