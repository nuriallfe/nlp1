from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace


def train_tokenizer(corpus, vocab_size, min_frequency, pretokenizer=Metaspace(), unk_token="[UNK]"):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization
    
        You probably don't need to change the pretokenizer, unless you are working with
        languages that really don't use any form of blank space 
        (then you could, for example, pretokenize at character level)
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token)) 

    tokenizer.pre_tokenizer = pretokenizer

    trainer = BpeTrainer(
        vocab_size=vocab_size, # The size of the vocabulary to learn.
        min_frequency=min_frequency, # Only perform a merge if the pair occurs at least min_frequency times in the training corpus.
        special_tokens=[unk_token]
    )

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return tokenizer    


def tokenizer_stats(tokenizer, texts, unk_token="[UNK]"):
    """Avg tokens per sentence and fraction of [UNK] tokens."""
    encs = tokenizer.encode_batch(texts)
    n_tok = sum(len(e.tokens) for e in encs)
    n_unk = sum(e.tokens.count(unk_token) for e in encs)
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "avg_tokens_per_sentence": n_tok / len(texts),
        "unk_rate": n_unk / n_tok,
    }
