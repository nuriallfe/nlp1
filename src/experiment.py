import hashlib
import json
import pickle
from collections import OrderedDict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from tqdm.auto import tqdm

from . import clslib, loglin, smlp, svm
from .encoders import BagOfWords, NgramBagOfWords, PairEncoder, TokenIndicator, encode_all
from .tokenization import train_tokenizer


REPRESENTATIONS = ("TokenIndicator", "BagOfWords", "NGram")
CACHE_VERSION = 3


def pair_key(x):
    """Unordered PAWS pair: (A,B) and (B,A) share a group."""
    s1, s2 = x
    return tuple(sorted((s1, s2)))


def group_ids_from_pairs(pairs):
    group_map = {}
    ids = []
    for x, _ in pairs:
        key = pair_key(x)
        if key not in group_map:
            group_map[key] = len(group_map)
        ids.append(group_map[key])
    return np.asarray(ids, dtype=np.int32)


def cross_validation_split(pairs, k=5, seed=0):
    """Stratified group CV over unordered *pairs*, not shared sentences."""
    pairs = list(pairs)
    if k < 2 or not pairs:
        raise ValueError("Need at least two folds and nonempty data")
    y = np.asarray([label for _, label in pairs], dtype=int)
    groups = group_ids_from_pairs(pairs)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = OrderedDict()
    for i, (tr, va) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        train_keys = {pair_key(pairs[j][0]) for j in tr}
        val_keys = {pair_key(pairs[j][0]) for j in va}
        if not train_keys.isdisjoint(val_keys):
            raise AssertionError("An unordered pair appears in both partitions")
        folds[i] = {
            "training": [pairs[j] for j in tr],
            "validation": [pairs[j] for j in va],
            "training_ids": tr.tolist(),
            "validation_ids": va.tolist(),
        }
    return folds


def fold_diagnostics(data_splits):
    rows = []
    for fold, split in data_splits.items():
        tr = split["training"]
        va = split["validation"]
        train_pairs = {pair_key(x) for x, _ in tr}
        val_pairs = {pair_key(x) for x, _ in va}
        train_sentences = {s for (s1, s2), _ in tr for s in (s1, s2)}
        shared_sentences = sum(any(s in train_sentences for s in x) for x, _ in va)
        counts = np.bincount([int(y) for _, y in va], minlength=2)
        rows.append({
            "fold": fold, "train": len(tr), "validation": len(va),
            "label0": int(counts[0]), "label1": int(counts[1]),
            "label1_ratio": counts[1] / len(va),
            "overlapping_pairs": len(train_pairs & val_pairs),
            "validation_pairs_with_shared_sentence": shared_sentences,
        })
    return pd.DataFrame(rows)


def _fold_signature(split, options):
    h = hashlib.sha256(json.dumps(options, sort_keys=True).encode("utf8"))
    for partition in ("training", "validation"):
        for (s1, s2), label in split[partition]:
            for item in (s1, s2, int(label)):
                payload = str(item).encode("utf8")
                h.update(len(payload).to_bytes(8, "little"))
                h.update(payload)
    return h.hexdigest()


def prepare_cv_features(data_splits, cache_dir="results_strict/feature_cache",
                        vocab_size=8000, min_frequency=2,
                        ngram_n=2, ngram_include_lower=True,
                        ngram_min_count=5, ngram_max_features=8000,
                        pair_mode="diff"):
    """Fit BPE/NGram once per training fold; cache reusable sparse matrices.

    Returned dict:
        prepared[fold][representation] = {
            X_train, X_val, y_train, y_val, train_groups
        }
    """
    options = dict(
        cache_version=CACHE_VERSION, vocab_size=vocab_size,
        min_frequency=min_frequency, ngram_n=ngram_n,
        ngram_include_lower=ngram_include_lower,
        ngram_min_count=ngram_min_count,
        ngram_max_features=ngram_max_features, pair_mode=pair_mode,
    )
    prepared = OrderedDict()
    cache_dir = Path(cache_dir)
    for fold, split in tqdm(data_splits.items(), desc="Prepare CV features"):
        folder = cache_dir / f"fold-{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        paths = {
            name: (folder / f"{name}_train.npz", folder / f"{name}_val.npz")
            for name in REPRESENTATIONS
        }
        meta_path = folder / "metadata.json"
        signature = _fold_signature(split, options)
        cached = (meta_path.exists() and
                  json.loads(meta_path.read_text()).get("signature") == signature and
                  all(a.exists() and b.exists() for a, b in paths.values()))
        y_train = np.asarray([y for _, y in split["training"]], dtype=np.int64)
        y_val = np.asarray([y for _, y in split["validation"]], dtype=np.int64)
        train_groups = group_ids_from_pairs(split["training"])
        if cached:
            print(f"Fold {fold}: loading cached features")
            prepared[fold] = {
                name: dict(X_train=load_npz(a), X_val=load_npz(b),
                           y_train=y_train, y_val=y_val,
                           train_groups=train_groups, preprocess_signature=signature)
                for name, (a, b) in paths.items()
            }
            continue

        print(f"Fold {fold}: train BPE only on {len(y_train):,} training pairs")
        train_texts = [s for (s1, s2), _ in split["training"] for s in (s1, s2)]
        tokenizer = train_tokenizer(train_texts, vocab_size, min_frequency)
        tokenizer.save(str(folder / "tokenizer.json"))
        base_encoders = {
            "TokenIndicator": TokenIndicator(tokenizer),
            "BagOfWords": BagOfWords(tokenizer),
            "NGram": NgramBagOfWords(
                tokenizer, n=ngram_n, include_lower=ngram_include_lower,
                min_count=ngram_min_count, max_features=ngram_max_features,
            ).fit(train_texts),
        }
        # Keep the learned NGram vocabulary so the fold can be reconstructed.
        with (folder / "ngram_vocab.pkl").open("wb") as f:
            pickle.dump(base_encoders["NGram"].vocab, f)

        prepared[fold] = {}
        for name, base in base_encoders.items():
            encoder = PairEncoder(base, mode=pair_mode)
            X_train = encode_all([x for x, _ in split["training"]], encoder)
            X_val = encode_all([x for x, _ in split["validation"]], encoder)
            save_npz(paths[name][0], X_train)
            save_npz(paths[name][1], X_val)
            prepared[fold][name] = dict(
                X_train=X_train, X_val=X_val, y_train=y_train,
                y_val=y_val, train_groups=train_groups, preprocess_signature=signature,
            )
            print(f"  {name}: X_train={X_train.shape}, X_val={X_val.shape}")
        meta_path.write_text(json.dumps({"signature": signature, **options}, indent=2))
    return prepared


def cross_validation_experiment(data_splits, outcome_space,
                                encoder=None, model_option="logistic",
                                save_to="tmp", loglin_args=None,
                                smlp_args=None, svm_args=None,
                                prepared_features=None, representation=None):
    """Run one configuration on the same held-out folds; use cached matrices.

    For reproducible final comparisons, provide prepared_features + representation.
    Legacy encoder argument is allowed for simple demonstrations only.
    """
    loglin_args = {} if loglin_args is None else dict(loglin_args)
    smlp_args = {} if smlp_args is None else dict(smlp_args)
    svm_args = {} if svm_args is None else dict(svm_args)
    models, predictions, accuracies, macro_f1 = (OrderedDict() for _ in range(4))
    for fold, split in tqdm(data_splits.items(), desc=f"{model_option}/{representation}"):
        folder = Path(save_to) / f"fold-{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        path = str(folder / "model")
        if prepared_features is not None:
            if representation not in prepared_features[fold]:
                raise KeyError(f"Feature representation {representation} missing in fold {fold}")
            d = prepared_features[fold][representation]
            Xtr, Xva = d["X_train"], d["X_val"]
            ytr, yva = d["y_train"], d["y_val"]
            if model_option == "logistic":
                model = loglin.fit_features(Xtr, ytr, save_to=path, **loglin_args)
                preds = loglin.predict_features(Xva, model)
            elif model_option == "svm":
                model = svm.fit_features(Xtr, ytr, save_to=path, **svm_args)
                preds = svm.predict_features(Xva, model)
            elif model_option == "mlp":
                _, _, preds = smlp.train_features(
                    Xtr, ytr, Xva, yva, len(outcome_space),
                    save_to=path, train_groups=d["train_groups"], **smlp_args)
            else:
                raise ValueError(f"Unknown model_option={model_option}")
        else:
            if encoder is None:
                raise ValueError("Supply prepared_features and representation for strict CV")
            # Legacy path: caller is responsible for fitting encoder without leakage.
            tr, va = split["training"], split["validation"]
            yva = np.asarray([y for _, y in va])
            if model_option == "logistic":
                model = loglin.fit(tr, encoder, save_to=path, **loglin_args)
                preds = loglin.predict([x for x, _ in va], model, encoder)
            elif model_option == "svm":
                model = svm.fit(tr, encoder, save_to=path, **svm_args)
                preds = svm.predict([x for x, _ in va], model, encoder)
            elif model_option == "mlp":
                raise ValueError("Use prepared_features for MLP to avoid using outer validation for early stopping")
            else:
                raise ValueError(f"Unknown model_option={model_option}")
        models[fold] = path
        predictions[fold] = np.asarray(preds, dtype=int)
        accuracies[fold] = accuracy_score(yva, preds)
        macro_f1[fold] = f1_score(yva, preds, average="macro", zero_division=0)
    return dict(models=models, predictions=predictions,
                accuracies=accuracies, macro_f1=macro_f1)


def run_grid_search(data_splits, outcome_space, prepared_features,
                    search_specs, results_dir="results_strict/models",
                    save_summary_to=None):
    """Automated small grid. Saves every configuration to disk for resuming.

    search_specs example:
      {'family':'logistic', 'model_option':'logistic',
       'representations':['BagOfWords'], 'base_args':{'max_iter':2000},
       'grid':{'C':[0.1,1.0]}}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = OrderedDict()
    rows = []
    preprocessing_id = {
        str(fold): prepared_features[fold][REPRESENTATIONS[0]]["preprocess_signature"]
        for fold in sorted(prepared_features)
    }
    model_arg_key = {"logistic": "loglin_args", "svm": "svm_args", "mlp": "smlp_args"}
    for spec in search_specs:
        model_option = spec["model_option"]
        reps = spec.get("representations", list(REPRESENTATIONS))
        for rep in reps:
            for params in ParameterGrid(spec["grid"]):
                args = {**spec.get("base_args", {}), **params}
                suffix = "_".join(f"{key}{value}" for key, value in sorted(params.items()))
                name = f"{spec['family']}_{rep}_{suffix}"
                folder = results_dir / name
                folder.mkdir(exist_ok=True)
                summary_path = folder / "cv.pkl"
                config_path = folder / "config.json"
                signature = {"model": model_option, "representation": rep, "args": args, "preprocessing": preprocessing_id}
                if summary_path.exists() and config_path.exists() and json.loads(config_path.read_text()) == signature:
                    with summary_path.open("rb") as f:
                        result = pickle.load(f)
                    print(f"Reusing CV result: {name}")
                else:
                    result = cross_validation_experiment(
                        data_splits=data_splits, outcome_space=outcome_space,
                        model_option=model_option, prepared_features=prepared_features,
                        representation=rep, save_to=str(folder),
                        **{model_arg_key[model_option]: args},
                    )
                    with summary_path.open("wb") as f:
                        pickle.dump(result, f)
                    config_path.write_text(json.dumps(signature, sort_keys=True, indent=2))
                all_results[name] = [result]  # compatible with analysis.make_data_frame
                rows.append({
                    "model": name, "family": spec["family"],
                    "representation": rep, "params": json.dumps(params, sort_keys=True),
                    "mean_accuracy": np.mean(list(result["accuracies"].values())),
                    "std_accuracy": np.std(list(result["accuracies"].values()), ddof=1),
                    "mean_macro_f1": np.mean(list(result["macro_f1"].values())),
                })
                if save_summary_to:
                    Path(save_summary_to).parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False).to_csv(save_summary_to, index=False)
    leaderboard = pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)
    return all_results, leaderboard


class WrapperForLoglinearModel(clslib.ProbabilisticModel):
    def __init__(self, encoder, model):
        self.encoder, self.model = encoder, model
    def __call__(self, x):
        return loglin.predict_probs([x], self.model, self.encoder)[0]


class WrapperForNeuralModel(clslib.ProbabilisticModel):
    def __init__(self, encoder, model):
        self.encoder, self.model = encoder, model
    def __call__(self, x):
        return smlp.predict_probs([x], self.model, self.encoder, batch_size=1)[0]
