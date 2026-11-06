import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from .encoders import encode_all


def fit_features(X_train, y_train, save_to=None, max_iter=2000,
                 C=1.0, penalty="l2", random_state=0):
    """Train on a precomputed sparse feature matrix.

    liblinear supports l1/l2 for binary classification; PAWS is binary.
    Convergence warnings are intentionally *not* suppressed.
    """
    kwargs = dict(solver="liblinear", max_iter=max_iter,
                  C=C, random_state=random_state)
    if penalty != "l2":
        kwargs["penalty"] = penalty
    model = LogisticRegression(**kwargs)
    model.fit(csr_matrix(X_train), np.asarray(y_train, dtype=int))
    if save_to is not None:
        path = Path(f"{save_to}.pickle")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return model


def fit(training_set, encoder, save_to=None, max_iter=2000, C=1.0,
        penalty="l2", random_state=0):
    X = encode_all([x for x, _ in training_set], encoder)
    y = [y for _, y in training_set]
    return fit_features(X, y, save_to=save_to, max_iter=max_iter, C=C,
                        penalty=penalty, random_state=random_state)


def predict_features(X, model):
    return model.predict(csr_matrix(X))


def predict_probs_features(X, model):
    return model.predict_proba(csr_matrix(X))


def predict(texts, model, encoder):
    return predict_features(encode_all(texts, encoder), model)


def predict_probs(texts, model, encoder):
    return predict_probs_features(encode_all(texts, encoder), model)


def load(path):
    with open(f"{path}.pickle", "rb") as f:
        return pickle.load(f)
