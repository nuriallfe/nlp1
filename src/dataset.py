from datasets import load_dataset


def load_paws(config: str = "labeled_final"):
    """Load PAWS (train/validation/test). Each example has sentence1, sentence2, label."""
    return load_dataset("google-research-datasets/paws", config)


def tokenizer_corpus(dataset):
    """Both sentences of every TRAIN pair (no validation/test data, to avoid leakage)."""
    return list(dataset["train"]["sentence1"]) + list(dataset["train"]["sentence2"])
