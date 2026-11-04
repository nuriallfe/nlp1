from datasets import load_dataset


def load_paws(config: str = "labeled_final"):
    """Load PAWS (train/validation/test). Each example has sentence1, sentence2, label."""
    return load_dataset("google-research-datasets/paws", config)


def tokenizer_corpus(dataset):
    """Both sentences of every TRAIN pair (no validation/test data, to avoid leakage)."""
    return list(dataset["train"]["sentence1"]) + list(dataset["train"]["sentence2"])


def paws_pairs(split):
    """
    Turn a PAWS split into a list of (x, label), where x = (sentence1, sentence2).
    This is the (input, label) format the cross-validation and training code expects.
    """
    return list(zip(zip(split["sentence1"], split["sentence2"]), split["label"]))
