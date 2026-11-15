import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn.utils.rnn import pad_sequence 
import time
from tqdm.auto import tqdm
from collections import Counter
from zipfile import ZipFile
from io import TextIOWrapper
from pathlib import Path
import gzip
import nltk
import re
import unicodedata
import random
import string

# ----------------------------
# Vocabulary and tokenization 
# ----------------------------

class Vocabulary:
    """
    This is a helper class that wraps around a list of symbols/words/tokens and provides 
    functions to map from str to int.
    """

    def __init__(self, init: list, default: str, frozen=False):
        if default not in init:
            raise ValueError("The default token must be in the init list")        
        self.words = list(init)
        self._str2int = {w: i for i, w in enumerate(self.words)}
        self._default_word = default
        self._default_index = self._str2int[self._default_word]        
        self._frozen = frozen

    @property
    def default_word(self):
        return self._default_word

    @property
    def default_index(self):
        return self._default_index
        
    def freeze(self):
        """After this operation add(.) no longer expands the vocabulary"""
        self._frozen = True

    def add(self, word: str) -> int:
        """
        Add a word to the vocabulary and return its id (existing words are not added again).
        If the vocabulary is frozen this adds nothing and returns the id of the default.
        """        
        if self._frozen:
            return self.index(word)
        
        i = self._str2int.get(word, -1)
        if i < 0:
            i = len(self.words)
            self.words.append(word)
            self._str2int[word] = i
        return i
            
    def index(self, word: str) -> int:
        """Convert word to 0-based integer (0-based because Python lists are 0-based)"""
        return self._str2int.get(word, self.default_index)

    def word(self, index: int) -> str:
        """Convert an 0-based integer to a word"""
        if index >= len(self):  # if the index is out of bounds, we return the default str
            return self.default_word
        else:
            return self.words[index]  

    def __contains__(self, word: str):
        return word in self._str2int
        
    def __len__(self):
        """Size of vocabulary"""
        return len(self.words)

    def __iter__(self):
        """Iterate over words"""
        return iter(self.words)

    def items(self):
        """Iterates over pairs of the kind (index, word) for all words in the vocabulary"""
        return enumerate(self.words)

    def __str__(self):
        return "\n".join(f"{i} {w}" for i, w in self.items())


class Tokenizer:
    def tokens(self, text: str) -> list:
        """Tokenize as a list of string tokens"""
        raise NotImplementedError("Implement me in subclass")
    def ids(self, text: str) -> list:
        """Tokenize as a list of token ids"""
        raise NotImplementedError("Implement me in subclass")


class BasicTokenizer(Tokenizer):
    """
    Tokenize text (str) using the call method of the object.
    
    Here are the  steps:
    1. lowercase (optional)
    2. remove punctuation (optional)
    3. NLT-tokenize 
    4. remove stopwords (optional)

    Different languages are only supported to the extent that NLTK supports them. 
    If you need something more complex, you need to write your own tokenizer.
    """

    def __init__(self, lowercase: bool, remove_punct: bool, remove_stopwords: bool, language='english', vocab: Vocabulary=None):
        """        
        lowercase: lowercase the text before tokenization
        remove_punct: remove punctuation 
        remove_stopwords: use NLKT to remove stopwords
        language: tells NLTK what language-specific rules to use
        vocab: a Vocabulary object that maintains a correspondence between string symbols and unique ids
        including a 'default' symbol for unkonwn words.
        if None, the basic tokenizer will not be able to implement the `ids` method
        """
        self.vocab = vocab
        self.language = language
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words(language))    

    def tokens(self, text) -> 'list[str]':
        """Chops text into a list of words/tokens (each a string)"""
        if self.lowercase:
            text = text.lower()
        if self.remove_punct:
            # sub space for punctuation
            text = re.sub(r'[^\w\s]', ' ', text)        
        # basic NLT tokenization
        tokens = nltk.tokenize.word_tokenize(text, language=self.language)
        if self.remove_stopwords:
            # filter stopwords
            tokens = [token for token in tokens if token not in self.stopwords]
        return tokens

    def ids(self, text) -> 'list[int]':
        """
        Chops text into a list of words/tokens ids.
        Unknown tokens get the id of the default symbol in the vocabulary.
        """
        if self.vocab is None:
            raise ValueError("I need a vocabulary")
        return [self.vocab.index(t) for t in self.tokens(text)]        
        

def read_lines_from_file(path, encoding="utf-8"):
    if path.endswith('.txt'):
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                yield line.strip()
    elif path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding=encoding) as f:
            for line in f:
                yield line.strip()
    elif path.endswith('.zip'):
        with ZipFile(path) as archive:
            members = archive.namelist()

            if len(members) != 1:
                raise ValueError(
                    f"Expected exactly one .txt file in {path}, "
                    f"found {len(members)}: {members}"
                )

            with archive.open(members[0]) as f:
                with TextIOWrapper(f, encoding=encoding) as text:
                    for line in text:
                        yield line.strip()        
    else:
        raise ValueError(f"Unknown format: {path}")

# --------------------
# Embeddings analysis
# --------------------

class PretrainedEmbeddings:
    """
    Load an embedding matrix (stored in one of the supported files, see read_lines_from_file)
    and gives structured access to it.     
    """

    def __init__(self, path, num_dims, unk_token="<unk>", pad_token="<pad>", encoding="utf-8", skip_first_line=False):
        """
        path: text/gz/zip file with embeddings
        num_dims: dimensionality of embeddings
        unk_token: special token reserve for words not in the table, we use the average of all embeddings as the unk embedding
        pad_token: special token used for batching sequences of variable length, we use a vector of zeros
        encoding: encoding of the embeddings file
        skip_first_line: some embedding files have metadata in the first line
        """
        vectors = [            
            np.zeros(num_dims, dtype=np.float32),  # pad
            np.zeros(num_dims, dtype=np.float32) # unk placeholder
        ]        
        self.vocab = Vocabulary(init=[pad_token, unk_token], default=unk_token)
        self.num_dims = num_dims        
        self.unk_token = unk_token
        self.pad_token = pad_token 
        
        for line_num, line in enumerate(read_lines_from_file(path, encoding=encoding)):            
            if skip_first_line and line_num == 0:
                continue
            parts = line.split()
            word = parts[0].lower() # as we don't know much about their preprocessing, let's lowercase words here
            values = parts[1:]            
            if len(values) != num_dims: # avoids some known problems with these files
                continue
            vector = np.array(values).astype(np.float32)
            if word in self.vocab:
                continue  # better skip clashes, if any
            self.vocab.add(word)
            vectors.append(vector) 
            
        # [num_words, num_dims]    
        vectors = np.stack(vectors, axis=0)
        # unk embedding is the mean of all others after pad (that's the "center" of the space)
        vectors[self.vocab.default_index] = np.mean(vectors[2:, :], 0)
        self.vectors = vectors
        # we freeze the vocab to avoid having it out of sync with the vectors
        self.vocab.freeze()
        assert self.vectors.shape[0] == len(self.vocab), "Mismatch between vocab and vectors"
        
    def __contains__(self, word: str):
        """Test whether or not a word (or its lowercased version) is in the vocabulary"""
        return word.lower() in self.vocab

    def index(self, word: str):        
        """Return the id of a word (or of its lowercased version) or 0 (by definition, id of unk_token)"""
        return self.vocab.index(word.lower())

    def word(self, i: int):    
        """Return the word associated with an id"""
        return self.vocab.word(i)        

    def __getitem__(self, i: int):
        """Return self.vectors[i]"""
        return self.vectors[i]  # this will raise an exceptino if i is out of bounds
        
    def embed_word(self, word: str):
        """Return the vector representation of the word (or its lowercased version, or unk)"""
        i = self.index(word)
        return self.vectors[i]
    
    def embed_sequence(self, tokens: list):        
        """Return a list of embeddings, one for each token in the input list"""
        return [self.embed_word(t) for t in tokens]
    

# ----------------------------------------
# START Shelf-implemented
def _scatter(ax, Z, mask, dim, **kw):
    """Scatter helper that works for 2D and 3D axes."""
    ax.scatter(*[Z[mask, i] for i in range(dim)], **kw)


def _new_axes(dim):
    fig = plt.figure(figsize=(8, 7))
    return fig.add_subplot(111, projection="3d") if dim == 3 else fig.add_subplot(111)


def _label_axes(ax, dim):
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if dim == 3:
        ax.set_zlabel("PC3")


def analyse_embs(trained_emb_model, class1, class2, method: str = "Dimensionality reduction", dim: int = 2):
    """
    trained_emb_model: PretrainedEmbeddings (uses `in`, `.embed_word`)
    class1, class2: lists of words (e.g. pos_words, neg_words)
    dim: 2 or 3, number of PCA components used for plotting
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    w1 = [w for w in class1 if w in trained_emb_model]
    w2 = [w for w in class2 if w in trained_emb_model]
    words = w1 + w2
    X = np.vstack([np.asarray(trained_emb_model.embed_word(w)) for w in words])  # (N, D)
    y = np.array([0] * len(w1) + [1] * len(w2))

    pca = PCA(n_components=dim).fit(X)
    Z = pca.transform(X)
    var = pca.explained_variance_ratio_.sum()

    if method == "Dimensionality reduction":
        ax = _new_axes(dim)
        for label, name, color in [(0, "class 1", "tab:blue"), (1, "class 2", "tab:red")]:
            _scatter(ax, Z, y == label, dim, c=color, label=name, alpha=0.5, s=12)
        ax.set_title(f"PCA {dim}D ({var:.1%} of variance explained)")
        _label_axes(ax, dim)
        ax.legend()
        plt.show()

    elif method == "k-means":
        # Clustering is done in the full embedding space, PCA is only for display
        pred = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)

        ari = adjusted_rand_score(y, pred)
        sil = silhouette_samples(X, pred)
        purity = sum(np.bincount(y[pred == k], minlength=2).max() for k in range(2)) / len(y)
        print(f"Adjusted Rand Index: {ari:.3f}  (1 = perfect, ~0 = random)")
        print(f"Purity:              {purity:.3f}  (chance ≈ {max(np.mean(y), 1 - np.mean(y)):.3f})")
        print(f"Silhouette:          {sil:.3f}")

        ax = _new_axes(dim)
        for k, color in [(0, "tab:green"), (1, "tab:orange")]:
            for label, marker in [(0, "o"), (1, "s")]:
                _scatter(ax, Z, (pred == k) & (y == label), dim, c=color, marker=marker,
                        alpha=0.5, s=12, label=f"cluster {k} / class {label + 1}")
        ax.set_title(f"k-means (k=2) vs. true classes, PCA {dim}D ({var:.1%} variance)")
        _label_axes(ax, dim)
        ax.legend()
        plt.show()

        return {"ari": ari, "purity": purity, "silhouette": sil}

    else:
        raise ValueError("Method not known")

def cos_similarity(u, v):  
    """
    Return the cosine similarity (a real number) of the two words in the embedding space of our model.
    """  
    eps = 1e-12
    cos_sim = np.inner(u/(np.linalg.norm(u)+eps), v/(np.linalg.norm(v)+eps))    
    return cos_sim

def topk_ids(v, E, k=10):    
    """
    word embedding: a D-dimensional vector
    E: our model's embedding matrix as a numpy array of shape [V, D]
    k: how many nearest neighbours we want to find

    Return a list with the k word ids that are nearest to the input word embedding v
    in embedding space according to cosine similarity. 
    """   
    sims = np.array([cos_similarity(v, E[idx, :]) for idx in range(E.shape[0])])
    # Give the idx that order the array in a way that the smallest k values are given first without an order
    top = np.argpartition(-sims, k)[:k]
    # Order the idx vector given in order with the smallest similarity first
    top = top[np.argsort(-sims[top])]
    return top.tolist()

_CHARS = str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ð": "d", "→": ":", "_": " ",
                        "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'"})
_MULTI = {"ß": "ss", "æ": "ae", "œ": "oe", "þ": "th"}
BUCKETS = {"freq = 1": (1, 1), "freq 2-5": (2, 5), "freq 6-50": (6, 50), "freq > 50": (51, 10**9)}
PAD, UNK = "<pad>", "<unk>"
def normalize_text(text):
    for k, v in _MULTI.items():
        text = text.replace(k, v)
    text = text.translate(_CHARS)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

class CleanTokenizer(BasicTokenizer):
    def tokens(self, text):
        return super().tokens(normalize_text(text))


def build_embedding_matrix(vocab, pretrained, counts, seed=0):
    """
    Build the embedding matrix aligned with `vocab` and compute OOV stats.

    Args:
        vocab: Vocabulary (uses iteration and .index())
        pretrained: PretrainedEmbeddings (uses `in`, .embed_word() and .num_dims)
        counts: Counter with word frequencies in the train set
        seed: seed for the random vectors of OOV words

    Returns:
        torch.Tensor [len(vocab), D] and a dict with OOV stats
        (types, tokens, per frequency bucket, top OOV words)
    """
    rng = np.random.default_rng(seed)
    # small-variance normal init for OOV words (breaks symmetry, similar scale to pretrained vectors)
    E = rng.normal(0, 0.1, (len(vocab), pretrained.num_dims)).astype(np.float32)
    E[vocab.index(PAD)] = 0.0

    words = [w for w in vocab if w not in (PAD, UNK)]
    oov = []
    for w in words:
        if w in pretrained:
            E[vocab.index(w)] = pretrained.embed_word(w)
        else:
            oov.append(w)

    oov_set = set(oov)
    stats = {
        "OOV types": len(oov) / len(words),
        "OOV tokens": sum(counts[w] for w in oov) / sum(counts[w] for w in words),
    }
    for b, (lo, hi) in BUCKETS.items():
        ws = [w for w in words if lo <= counts[w] <= hi]
        stats[b] = sum(w in oov_set for w in ws) / max(len(ws), 1)
    stats["top_oov"] = sorted(oov, key=lambda w: -counts[w])[:10]
    return torch.tensor(E), stats


def build_vocab(sentences, tokenizer, min_freq = 1):
    """
    Create the vocabulary with the training corpus with nltk tokenizer
    """
    counts = Counter(t for s in sentences for t in tokenizer.tokens(s))
    vocab = Vocabulary([PAD, UNK], default= UNK) # Giving pad = 0 and unk = 1
    for w, c in counts.items(): 
        if c >= min_freq:
            vocab.add(w)
    vocab.freeze()
    return vocab, counts
# END Shelf-implemented
#---------------------------------------------

# ---------------------------------
# Compositional encoder functions
# ---------------------------------
# padding batchs for sentences of different length 
def collate_sequences(sequences, padding_value=0):    
    """
    This is a helper to batch tokenized sequences of variable length into a torch tensor with shape (num_sequences, longest_seq_length)
    sequences: list of tokenized texts (each text is a list of token ids)
    padding_value: the id of the pad symbol
    """
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    x = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in sequences], batch_first=True, padding_value=padding_value)
    return x


class EmbedAndPoolEncoder(nn.Module):
    """
    This encoder embeds a sequence of token ids and then pools the result along sequence length using
    sum or mean.
    """
    def __init__(self, embedding_matrix, average_pool=True, padding_idx=0, freeze_embeddings=True):
        """
        embedding_matrix: a torch tensor of shape (vocab_size, emb_dim)
        average_pool: use True for mean and False for sum
        padding_idx: the id of the pad token (this is for batched computation)
            suppose we are batching three sequences [1, 2, 3], [6, 2] and [2, 1, 4, 5]
            as they differ in length, we use the following trick to batch them into a single tensor
            [
                [1, 2, 3, 0],
                [6, 2, 0, 0],
                [2, 1, 4, 5]
            ]
            that is we extend the shorter sequences with 0s until they match the longest sequence
        freeze_embeddings: tells torch that we are not updating the parameters in the embedding matrix
        """
        super().__init__()
        self.average_pool = average_pool
        # Pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, 
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )       

    @property
    def enc_dim(self):
        return self.embedding.embedding_dim

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):     
        """
        Encodes each sequence in the batch as a single vector of size emb_dim.
        
        x: token ids with shape (batch_size, max_seq_length, emb_dim)        
        """
        # [..., max_seq_length, emb_dim]
        e = self.embedding(x)  
        # Normalization for pooling
        if self.average_pool:
            # Total non-padding tokens (min of 1 to avoid problems when averaging the sum of embs)
            # [..., max_seq_length, 1]  # keepdim=True keeps the last axis, this is needed for division later on
            den = torch.sum(x.ne(self.embedding.padding_idx), dim=-1, keepdim=True).clamp_min(1)
        else:  # sum pooling
            den = 1
        # [..., emb_dim]
        h = torch.sum(e, -2) / den  # see that we reduce-sum the sequence dimension, not the representation dimension
        return h    

        
class RNNEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_matrix, padding_idx=0, freeze_embeddings=True):
        """        
        hidden_size: the size of the recurrent state

        To simplify implementation, we treat the embedding layer as part of the encoder. 
        
        embedding_matrix: word vectors            
        padding_idx: the id of the pad token
        freeze_embeddings: treat the vectors as frozen or not
        """
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.emb_dim = embedding_dim
        self.hidden_size = hidden_size
        # Pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, 
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )
        # Here's the design of the update
        # see that we can achieve the same parameterisation as explained above
        # via h[i] = tanh(linear(concat(e, h[i-1])))
        # so the key trainable component of the simplest recurrent encoder is this one non-linear transformation with D+H inputs and H outputs
        self.elman_net = nn.Sequential(
            nn.Linear(embedding_dim + hidden_size, hidden_size),
            nn.Tanh(),            
        )  

    @property
    def enc_dim(self):
        return self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        """
        Encode each sequence in the batch into a single H-dimensional vector.
        
        x: a batch of sequences of token ids
        """
        
        embeddings = self.embedding(x)  # (batch_size, max_len, emb_dim)
        # intial state
        # zeros on the same device and with the same dtype as embeddings.
        h = torch.zeros(x.shape[:-1] + (self.hidden_size,), dtype=embeddings.dtype, device=embeddings.device)
        max_len = x.shape[-1]
        for t in range(max_len):
            e_t = embeddings[..., t, :]  # (..., emb_dim)            
            # Candidate state for every example in the batch.
            # (..., hidden_dim)
            h_candidate = torch.tanh(self.elman_net(torch.cat([e_t, h], -1)))             
            # Update valid positions; retain the old state at padding positions.
            # (..., 1)            
            valid = x[..., t].ne(self.embedding.padding_idx).unsqueeze(-1)  
            # (..., hidden_dim)
            h = torch.where(valid, h_candidate, h)
        return h        

class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_matrix, padding_idx=0, freeze_embeddings=True, word_dropout=0.):
        """
        hidden_size: size of the recurrent state
        embedding_matrix: word vectors
        padding_idx: id of the pad token
        freeze_embeddings: treat embeddings as frozen
        """
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        # Pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, 
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size,
            num_layers=1, # this could be turned into an argument 
            batch_first=True  # important, in NLP we treat the first axis of the tensor as the batch dimension, and the second as the sequence dimension
        )

    @property
    def enc_dim(self):
        return self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        """
        Encode each sequence in the batch into a single H-dimensional vector.
        
        x: a batch of sequences of token ids
        """
        # [...,max_seq_len, emb_dim]
        embedded = self.embedding(x)   
        valid = x.ne(self.embedding.padding_idx)

        # Word dropout is a regularisation technique for recurrent architectures
        # If you are curious why we have to drop words in order to regularise RNNs, 
        # here's the paper that figured it out: https://proceedings.neurips.cc/paper_files/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
        # In a nutshell, if we apply dropout before linear layers inside the RNN cell
        # as sequences grow long, information is completely destroyed.
        # That paper builds on a previous paper that interpreted dropout as a specific kind of approximate inference (for a specific kind of approximate Bayesian model)
        # and then establishes that to achieve the same in a recurrent architecture, one can drop words as we do here
        # (By dropping a word, we mean 0-out its embedding)
        if self.training and self.word_dropout > 0:
            # Independently zero each non-padding token with probability word_dropout.
            # [..., max_seq_len]
            drop_mask = (
                torch.rand(x.shape, device=embedded.device) < self.word_dropout # at a certain pre-specified rate
            ) & valid
            # we 0-out the embedding of a valid (not pad) position
            embedded = embedded.masked_fill(drop_mask.unsqueeze(-1), 0.0)
                
        # [..., max_seq_len] this is necessary for packing
        lengths = torch.sum(valid, dim=-1)
        # for efficiency, we let torch do the for loop, and for that we need to 'pack' our sentences
        # using a torch helper
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False) # sorting leads to even more efficient processing, but it is cumbersome and we skip it
        output, (hidden, cell) = self.lstm(packed)
        final_hidden = hidden[-1]
        return final_hidden        

class SoftmaxRegressor(nn.Module):
    """
    This is a softmax regressor, 
    it works with encoded text h(x) as input.
    The map from h(x) to outputs uses an MLP.

    One important detail: for numerical stability the output is not softmax-activitate, 
    rather, it is logsoftmax-activated. This means the outputs are log prob masses, not prob masses.
    """
    def __init__(self, hidden_size1, hidden_size2, num_classes, encoder: nn.Module, dropout=0.):
        """
        hidden_size1: size of first hidden layer (in the output transformation)
        hidden_size2: size of second hidden layer
        num_classes: number of possible classes
        encoder: an nn.Module that maps each x to a single vector of size encoder.enc_dim
        """
        super().__init__()
        self.encoder = encoder      
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout), # dropout is a regularisation technique (whereby at some rate, we drop/0-out inputs to a linear layer)
            nn.Linear(encoder.enc_dim, hidden_size1),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size2, num_classes),
            nn.LogSoftmax(dim=-1) # for numerical stability, it is common to use log softmax, rather than softmax
        )

    def forward(self, x):  
        # [..., enc_dim]
        h = self.encoder(x)
        # [...,num_classes]
        logprobs = self.output_layer(h)
        return logprobs            
