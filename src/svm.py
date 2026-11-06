
import pickle
import numpy as np

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC

from .encoders import encode_all


def fit_features(
    X_train,
    y_train,
    save_to=None,
    C=1.0,
    max_iter=5000
):
    model = LinearSVC(
        C=C,
        dual="auto",
        max_iter=max_iter,
        random_state=0
    )

    model.fit(
        csr_matrix(X_train),
        np.asarray(y_train, dtype=int)
    )

    if save_to is not None:
        path = Path(f"{save_to}.pickle")
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(model, f)

    return model


def predict_features(X, model):
    return model.predict(csr_matrix(X))


def fit(training_set, encoder, save_to=None, C=1.0, max_iter=5000):
    X = encode_all([x for x, _ in training_set], encoder)
    y = [y for _, y in training_set]

    return fit_features(
        X, y,
        save_to=save_to,
        C=C,
        max_iter=max_iter
    )


def predict(texts, model, encoder):
    X = encode_all(texts, encoder)
    return predict_features(X, model)


def decision_scores(texts, model, encoder):
    X = encode_all(texts, encoder)
    return model.decision_function(X)


def load(path):
    with open(f"{path}.pickle", "rb") as f:
        return pickle.load(f)
