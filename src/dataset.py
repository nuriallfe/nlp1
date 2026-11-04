from datasets import load_dataset
from collections import defaultdict

def remove_conflicting_pairs(train):
    """Remove all training examples with conflicting paraphrase labels."""

    # Collect labels assigned to each unordered sentence pair
    pair_labels = defaultdict(set)

    for example in train:
        pair = tuple(sorted((
            example["sentence1"],
            example["sentence2"]
        )))
        pair_labels[pair].add(example["label"])

    # Identify pairs assigned both labels
    conflicting_pairs = {
        pair
        for pair, labels in pair_labels.items()
        if len(labels) > 1
    }

    # Keep only examples without conflicting annotations
    keep_indices = [
        i
        for i, example in enumerate(train)
        if tuple(sorted((
            example["sentence1"],
            example["sentence2"]
        ))) not in conflicting_pairs
    ]

    cleaned_train = train.select(keep_indices)

    print(
        f"Removed {len(train) - len(cleaned_train)} examples "
        f"from {len(conflicting_pairs)} conflicting pairs."
    )
    print(f"Training examples remaining: {len(cleaned_train):,}")

    return cleaned_train


def load_paws(config: str = "labeled_final", remove_conflicts: bool = True):
    """Load PAWS (train/validation/test). Each example has sentence1, sentence2, label."""

    dataset = load_dataset("google-research-datasets/paws", config)

    if remove_conflicts:
        dataset["train"] = remove_conflicting_pairs(dataset["train"])
        
    return dataset


def tokenizer_corpus(dataset):
    """Both sentences of every TRAIN pair (no validation/test data, to avoid leakage)."""
    return list(dataset["train"]["sentence1"]) + list(dataset["train"]["sentence2"])


def paws_pairs(split):
    """
    Turn a PAWS split into a list of (x, label), where x = (sentence1, sentence2).
    This is the (input, label) format the cross-validation and training code expects.
    """
    return list(zip(zip(split["sentence1"], split["sentence2"]), split["label"]))
