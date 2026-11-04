import numpy as np


class NamedClasses:
    """Wraps a list of class names and maps names (str) to 0-based integers and back."""

    def __init__(self, names: list):
        self.names = tuple(names)
        self._str2int = {name: k for k, name in enumerate(names)}

    def index(self, name: str):
        return self._str2int[name]

    def name(self, index: int):
        return self.names[index]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def items(self):
        return enumerate(self.names)

    def __str__(self):
        return ", ".join(f"{i}: {n}" for i, n in self.items())


class ProbabilisticModel:
    """Maps an input x to a probability mass function."""

    def __call__(self, x) -> 'np.array':
        raise NotImplementedError("Implement me in a sub-class")


class DecisionRule:
    """Chooses an outcome given a pmf."""

    def __call__(self, pmf: 'np.array') -> int:
        raise NotImplementedError("Implement me in a sub-class")


class MostProbableClass(DecisionRule):
    def __call__(self, pmf: 'np.array') -> int:
        return np.argmax(pmf)


class MajorityVoteClassifier:
    """Ensemble: each model votes, ties are broken at random."""

    def __init__(self, outcome_space: NamedClasses, models: list, decoder: DecisionRule, seed=0):
        self.outcome_space = outcome_space
        self.models = models
        self.decoder = decoder
        self.rng = np.random.RandomState(seed)

    def __call__(self, x) -> str:
        candidates = [self.decoder(model(x)) for model in self.models]
        values, counts = np.unique(candidates, return_counts=True)
        index = self.rng.choice(values[counts == counts.max()])
        return self.outcome_space.name(index)
