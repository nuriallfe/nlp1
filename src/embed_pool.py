"""Embed-and-pool classifiers for PAWS (Part 2, exercise 3).

A pair of sentences (s1, s2) is classified as follows:

    s1 -> token ids -> embeddings -> pool -> u
    s2 -> token ids -> embeddings -> pool -> v      (same encoder, shared weights)
    combine(u, v) -> MLP -> log p(label | s1, s2)

The encoder and the training loop follow the course tutorial (part2/clslib.py and
part2/experiment.py), adapted to *pairs* of sentences, and they reuse the
cross-validation folds of Part 1 (src/experiment.py), so that every number here
can be compared fold by fold with the Part 1 models.
"""
import json
import pickle
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .experiment import group_ids_from_pairs


# ---------------------------------------------------------------------------
# Data: token ids for every pair
# ---------------------------------------------------------------------------

def tokenize_pairs(pairs, tokenizer):
    """[((s1, s2), y), ...] -> [((ids1, ids2), y), ...] using tokenizer.ids()."""
    return [((tokenizer.ids(s1), tokenizer.ids(s2)), int(y)) for (s1, s2), y in pairs]


class PairCorpus(Dataset):
    def __init__(self, id_pairs):
        self.x1 = [torch.tensor(a, dtype=torch.long) for (a, _), _ in id_pairs]
        self.x2 = [torch.tensor(b, dtype=torch.long) for (_, b), _ in id_pairs]
        self.y = torch.tensor([y for _, y in id_pairs], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x1[i], self.x2[i], self.y[i]


def make_collate(pad_id):
    def collate(batch):
        x1, x2, y = zip(*batch)
        return (pad_sequence(x1, batch_first=True, padding_value=pad_id),
                pad_sequence(x2, batch_first=True, padding_value=pad_id),
                torch.stack(y))
    return collate


def bag_key(ids):
    """Order-free key of a token sequence (its multiset of tokens)."""
    return tuple(sorted(Counter(ids).items()))


def bag_of_words_ceiling(id_pairs):
    """Best accuracy any model can reach if it only sees the two *bags* of tokens.

    Embed-and-pool (with mean, sum or max pooling) is invariant to word order, so
    two pairs whose sentences have the same bags of tokens receive exactly the
    same input and therefore the same prediction. Grouping the pairs by
    (bag(s1), bag(s2)) and answering the majority label of each group is the best
    such a model can do on this data.

    Returns a dict with the ceiling and the share of pairs whose two sentences
    have identical bags (word-swapped pairs), per label.
    """
    groups = defaultdict(Counter)
    same_bag = []
    for (a, b), y in id_pairs:
        ka, kb = bag_key(a), bag_key(b)
        groups[(ka, kb)][y] += 1
        same_bag.append((ka == kb, y))
    n = len(id_pairs)
    ceiling = sum(max(c.values()) for c in groups.values()) / n
    same = np.array([s for s, _ in same_bag])
    ys = np.array([y for _, y in same_bag])
    return {
        "pairs": n,
        "bag_of_words_ceiling": ceiling,
        "same_bag_share": same.mean(),
        "same_bag_share_label0": same[ys == 0].mean(),
        "same_bag_share_label1": same[ys == 1].mean(),
        "paraphrase_rate_among_same_bag": ys[same].mean() if same.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Embedding matrices used in the experiments
# ---------------------------------------------------------------------------

def random_embeddings(num_rows, dim, std, pad_id=0, seed=0):
    """Random N(0, std^2) matrix (no pretraining). The pad row is zero."""
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(num_rows, dim, generator=g) * std
    E[pad_id] = 0.0
    return E


def pca_embeddings(E, dim, pad_id=0, unk_id=1):
    """Reduce a pretrained matrix to `dim` dimensions with PCA.

    PCA is fitted on the real word rows (pad and unk excluded). It keeps the
    directions with most variance, so it is the most faithful way to obtain a
    smaller version of the same pretrained space.
    """
    rows = [i for i in range(E.shape[0]) if i not in (pad_id, unk_id)]
    pca = PCA(n_components=dim, random_state=0).fit(E[rows].numpy())
    R = torch.tensor(pca.transform(E.numpy()), dtype=torch.float32)
    R[pad_id] = 0.0
    return R, float(pca.explained_variance_ratio_.sum())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PoolEncoder(nn.Module):
    """Embed a batch of token ids and pool over the valid (non-pad) positions.

    Same idea as the tutorial's EmbedAndPoolEncoder, with max pooling added.
    pooling: "mean", "sum" or "max".
    """

    def __init__(self, embedding_matrix, pooling="mean", freeze=True, pad_id=0):
        super().__init__()
        if pooling not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling
        self.pad_id = pad_id
        # clone: from_pretrained shares memory with the input tensor, so without a copy
        # fine-tuning would modify the caller's matrix (E_ft, E_glove, ...) in place,
        # and the next fold or experiment would start from already fine-tuned vectors.
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone().float(), freeze=freeze, padding_idx=pad_id)

    @property
    def enc_dim(self):
        return self.embedding.embedding_dim

    def forward(self, x):
        e = self.embedding(x)                      # [B, L, D]
        valid = x.ne(self.pad_id).unsqueeze(-1)    # [B, L, 1]
        if self.pooling == "max":
            e = e.masked_fill(~valid, float("-inf"))
            h = e.max(dim=1).values
            return torch.where(torch.isinf(h), torch.zeros_like(h), h)  # empty sentence
        h = (e * valid).sum(dim=1)
        if self.pooling == "mean":
            h = h / valid.sum(dim=1).clamp_min(1)
        return h


PAIR_MODES = {
    # name: (feature builder, number of D-sized blocks)
    "concat":    (lambda u, v: torch.cat([u, v], -1), 2),
    "diff_prod": (lambda u, v: torch.cat([(u - v).abs(), u * v], -1), 2),
    "full":      (lambda u, v: torch.cat([u, v, (u - v).abs(), u * v], -1), 4),
}


class PairEmbedPoolClassifier(nn.Module):
    """Siamese embed-and-pool encoder + MLP that outputs log-probabilities.

    hidden_sizes: () gives a log-linear model on the pooled features,
                  (128,) an MLP with one hidden layer, (128, 128) two layers, ...
    """

    def __init__(self, embedding_matrix, num_classes=2, pooling="mean", freeze=True,
                 pair_mode="full", hidden_sizes=(128,), dropout=0.1,
                 activation="relu", pad_id=0):
        super().__init__()
        self.encoder = PoolEncoder(embedding_matrix, pooling, freeze, pad_id)
        self.combine, blocks = PAIR_MODES[pair_mode]
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers, d_in = [], blocks * self.encoder.enc_dim
        for h in hidden_sizes:
            layers += [nn.Dropout(dropout), nn.Linear(d_in, h), act()]
            d_in = h
        layers += [nn.Dropout(dropout), nn.Linear(d_in, num_classes), nn.LogSoftmax(-1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x1, x2):
        return self.head(self.combine(self.encoder(x1), self.encoder(x2)))


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def inner_split(id_pairs, raw_pairs, dev_fraction=0.1, seed=0):
    """Split the training part of a fold into train / early-stopping dev.

    Groups are unordered sentence pairs (as in Part 1), so (A, B) and (B, A)
    always land on the same side. The validation fold is never used for
    early stopping, so its accuracy is an honest estimate.
    """
    groups = group_ids_from_pairs(raw_pairs)
    gss = GroupShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=seed)
    tr, dev = next(gss.split(np.zeros(len(groups)), groups=groups))
    return [id_pairs[i] for i in tr], [id_pairs[i] for i in dev]


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu", bar=None):
    training = optimizer is not None
    model.train(training)
    total_loss, logprobs_all, labels_all = 0.0, [], []
    with torch.set_grad_enabled(training):
        for step, (x1, x2, y) in enumerate(loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logprobs = model(x1, x2)
            loss = criterion(logprobs, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            logprobs_all.append(logprobs.detach().cpu())
            labels_all.append(y.cpu())
            if bar is not None:
                bar.update(1)
                if step % 50 == 0:
                    bar.set_postfix(train_loss=f"{loss.item():.4f}")
    logprobs_all = torch.cat(logprobs_all)
    labels_all = torch.cat(labels_all)
    preds = logprobs_all.argmax(-1)
    return {
        "loss": total_loss / len(labels_all),
        "accuracy": (preds == labels_all).float().mean().item(),
        "logprobs": logprobs_all,
    }


def train_model(train_ids, dev_ids, embedding_matrix, model_kwargs, opt_kwargs,
                pad_id=0, seed=0, device="cpu", verbose=True, desc=""):
    """Train one model with early stopping on `dev_ids`. Returns (model, history).

    verbose: show a live progress bar (batches of the current epoch) and one line
    per epoch with train / dev loss and accuracy; '*' marks a new best epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PairEmbedPoolClassifier(embedding_matrix, pad_id=pad_id, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_kwargs["learning_rate"], weight_decay=opt_kwargs["weight_decay"])
    criterion = nn.NLLLoss()
    collate = make_collate(pad_id)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(PairCorpus(train_ids), batch_size=opt_kwargs["batch_size"],
                              shuffle=True, collate_fn=collate, generator=g)
    dev_loader = DataLoader(PairCorpus(dev_ids), batch_size=512, collate_fn=collate)

    history, best_loss, best_state, bad_epochs = [], float("inf"), None, 0
    max_epochs = opt_kwargs["num_epochs"]
    for epoch in range(max_epochs):
        t0 = time.time()
        bar = tqdm(total=len(train_loader), desc=f"{desc} epoch {epoch + 1}/{max_epochs}",
                   leave=False, disable=not verbose)
        tr = _run_epoch(model, train_loader, criterion, optimizer, device, bar=bar)
        bar.close()
        dv = _run_epoch(model, dev_loader, criterion, device=device)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
                        "dev_loss": dv["loss"], "dev_acc": dv["accuracy"]})
        improved = dv["loss"] < best_loss - 1e-4
        if improved:
            best_loss, bad_epochs = dv["loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if verbose:
            print(f"    {desc} epoch {epoch + 1:2d}/{max_epochs}  "
                  f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
                  f"dev loss {dv['loss']:.4f} acc {dv['accuracy']:.4f}  "
                  f"({time.time() - t0:.0f}s){'  *' if improved else ''}")
        if bad_epochs >= opt_kwargs["patience"]:
            if verbose:
                print(f"    {desc} early stop: no dev improvement for "
                      f"{opt_kwargs['patience']} epochs")
            break
    model.load_state_dict(best_state)
    return model, history


def predict(model, id_pairs, pad_id=0, device="cpu", batch_size=512):
    loader = DataLoader(PairCorpus(id_pairs), batch_size=batch_size,
                        collate_fn=make_collate(pad_id))
    out = _run_epoch(model, loader, nn.NLLLoss(), device=device)
    return out["logprobs"].exp().numpy()


DEFAULT_OPT = {"learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 64,
               "num_epochs": 30, "patience": 3, "train_fraction": 1.0}


def run_cv(name, folds, fold_ids, embedding_matrix, model_kwargs, opt_kwargs=None,
           seed=0, pad_id=0, device="cpu", results_dir="results/embed_pool",
           extra_config=None, verbose=True):
    """Cross-validate one configuration over the Part 1 folds.

    folds    : output of src.experiment.cross_validation_split (raw text pairs)
    fold_ids : same structure with token ids (tokenize_pairs applied per fold)

    Results are saved in results_dir/<name>/ (config.json + cv.pkl with the same
    keys as Part 1: accuracies, macro_f1, predictions, plus probs and history).
    If cv.pkl already exists it is loaded and nothing is trained.
    """
    opt = {**DEFAULT_OPT, **(opt_kwargs or {})}
    out_dir = Path(results_dir) / name
    cv_file = out_dir / "cv.pkl"
    if cv_file.exists():
        saved = json.loads((out_dir / "config.json").read_text())
        requested = json.loads(json.dumps({"model": model_kwargs, "opt": opt, "seed": seed,
                                           **(extra_config or {})}))
        if any(saved.get(k) != v for k, v in requested.items()):
            raise ValueError(f"{out_dir} holds results of a different configuration. "
                             f"Delete the folder or use another name.")
        with cv_file.open("rb") as f:
            result = pickle.load(f)
        if verbose:
            accs = list(result["accuracies"].values())
            print(f"✓ {name}: loaded from disk (mean acc {np.mean(accs):.4f})")
        return result

    t0 = time.time()
    if verbose:
        print(f"▶ {name}: training {len(folds)} folds | model {model_kwargs} | "
              f"lr {opt['learning_rate']}, batch {opt['batch_size']}, "
              f"max {opt['num_epochs']} epochs, seed {seed}")
    result = {"accuracies": OrderedDict(), "macro_f1": OrderedDict(),
              "predictions": OrderedDict(), "probs": OrderedDict(),
              "history": OrderedDict(), "best_epoch": OrderedDict()}
    for fold, split in folds.items():
        train_ids, dev_ids = inner_split(fold_ids[fold]["training"], split["training"], seed=seed)
        if opt["train_fraction"] < 1.0:      # learning-curve experiments
            rng = np.random.default_rng(seed)
            keep = rng.permutation(len(train_ids))[: int(opt["train_fraction"] * len(train_ids))]
            train_ids = [train_ids[i] for i in sorted(keep)]
        model, history = train_model(train_ids, dev_ids, embedding_matrix, model_kwargs,
                                     opt, pad_id=pad_id, seed=seed, device=device,
                                     verbose=verbose, desc=f"[{name} | fold {fold + 1}/{len(folds)}]")
        probs = predict(model, fold_ids[fold]["validation"], pad_id, device)
        y_true = np.array([y for _, y in fold_ids[fold]["validation"]])
        y_pred = probs.argmax(-1)
        result["accuracies"][fold] = accuracy_score(y_true, y_pred)
        result["macro_f1"][fold] = f1_score(y_true, y_pred, average="macro")
        result["predictions"][fold] = y_pred
        result["probs"][fold] = probs
        result["history"][fold] = history
        result["best_epoch"][fold] = int(np.argmin([h["dev_loss"] for h in history]))
        if verbose:
            elapsed = (time.time() - t0) / 60
            remaining = elapsed / (fold + 1) * (len(folds) - fold - 1)
            print(f"  → fold {fold + 1}/{len(folds)}: validation acc {result['accuracies'][fold]:.4f} "
                  f"(best epoch {result['best_epoch'][fold] + 1} of {len(history)}) | "
                  f"{elapsed:.1f} min so far, ~{remaining:.1f} min left for {name}")

    trainable, total = count_parameters(model)
    config = {"name": name, "seed": seed, "model": model_kwargs, "opt": opt,
              "trainable_params": trainable, "total_params": total,
              "minutes": round((time.time() - t0) / 60, 2), **(extra_config or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    with cv_file.open("wb") as f:
        pickle.dump(result, f)
    if verbose:
        accs = list(result["accuracies"].values())
        print(f"[{name}] mean acc = {np.mean(accs):.4f} ± {np.std(accs):.4f} "
              f"({config['minutes']} min)")
    return result


# ---------------------------------------------------------------------------
# Controlled comparison: the Part 1 MLP, fed with pooled embeddings
# ---------------------------------------------------------------------------

def pool_sentences(seqs, embedding_matrix, pooling="mean", pad_id=0, batch_size=2048):
    """Pooled vector of every token-id sequence (fixed encoder, no training)."""
    encoder = PoolEncoder(embedding_matrix, pooling, freeze=True, pad_id=pad_id)
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            x = pad_sequence([torch.tensor(s, dtype=torch.long) for s in seqs[i:i + batch_size]],
                             batch_first=True, padding_value=pad_id)
            out.append(encoder(x))
    return torch.cat(out).numpy()


def pair_features(U, V):
    """[|u - v|, u * v]: the pair representation used in Part 1 (pair_mode='diff')."""
    return np.hstack([np.abs(U - V), U * V]).astype(np.float32)


def _part1_mlp(input_dim, hidden_dim, dropout, num_hidden_layers=1, activation="relu"):
    """Same architecture as Part 1's SparseMLP (src/smlp.py), for dense inputs.

    Linear -> act -> dropout (x num_hidden_layers) -> Linear -> log-softmax,
    with Xavier-uniform initialisation of the first layer, as in Part 1.
    """
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
    layers = [nn.Linear(input_dim, hidden_dim), act(), nn.Dropout(dropout)]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act(), nn.Dropout(dropout)]
    layers += [nn.Linear(hidden_dim, 2), nn.LogSoftmax(-1)]
    model = nn.Sequential(*layers)
    nn.init.xavier_uniform_(model[0].weight)
    nn.init.zeros_(model[0].bias)
    return model


def run_cv_part1_mlp(name, folds, X, y, mlp_args, results_dir="results/embed_pool",
                     extra_config=None, verbose=True, standardize=False):
    """Cross-validate the Part 1 MLP (architecture and hyper-parameters) on features X.

    X[i] are the features of train pair i (the order of the pairs given to
    cross_validation_split), so each fold takes its rows with training_ids and
    validation_ids.

    Same network, optimiser (AdamW) and hyper-parameters as the best Part 1 MLP.
    Training follows the protocol of this notebook (stages A-F): train once on
    90% of the fold, early stopping on a 10% inner dev split grouped by
    unordered pair, keep the best dev epoch, evaluate once on the validation fold.
    (Part 1 additionally refitted on 100% of the fold; we skip it for consistency
    with the other Part 2 models and to save time.)

    standardize=True rescales every feature to mean 0 and std 1 with the
    statistics of the training part of each fold only (no information from
    the validation fold).
    """
    out_dir = Path(results_dir) / name
    cv_file = out_dir / "cv.pkl"
    config = {"name": name, "classifier": "Part 1 MLP architecture, dense input",
              "procedure": "single training, best inner-dev epoch (no refit)",
              "standardize": standardize,
              "mlp_args": mlp_args, "input_dim": int(X.shape[1]), **(extra_config or {})}
    if cv_file.exists():
        saved = json.loads((out_dir / "config.json").read_text())
        if saved != json.loads(json.dumps(config)):
            raise ValueError(f"{out_dir} holds results of a different configuration. "
                             f"Delete the folder or use another name.")
        with cv_file.open("rb") as f:
            result = pickle.load(f)
        print(f"✓ {name}: loaded from disk (mean acc {np.mean(list(result['accuracies'].values())):.4f})")
        return result

    a = mlp_args
    seed, device = a.get("seed", 0), a.get("device", "cpu")
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    criterion = nn.NLLLoss()
    t0 = time.time()
    if verbose:
        print(f"▶ {name}: Part 1 MLP on {len(folds)} folds ({X.shape[1]} input features)")
    result = {"accuracies": OrderedDict(), "macro_f1": OrderedDict(),
              "predictions": OrderedDict(), "probs": OrderedDict(),
              "selected_epochs": OrderedDict(), "history": OrderedDict()}

    for fold, split in folds.items():
        tr_all = np.asarray(split["training_ids"])
        va = np.asarray(split["validation_ids"])
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        tr_pos, dev_pos = next(gss.split(np.zeros(len(tr_all)),
                                         groups=group_ids_from_pairs(split["training"])))
        tr, dev = tr_all[tr_pos], tr_all[dev_pos]
        Xf = X
        if standardize:   # statistics of the training part of the fold only
            mu, sd = X[tr_all].mean(0), X[tr_all].std(0) + 1e-8
            Xf = (X - mu) / sd

        torch.manual_seed(seed)
        model = _part1_mlp(X.shape[1], a["hidden_dim"], a["dropout"],
                           a.get("num_hidden_layers", 1), a.get("activation", "relu")).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a["learning_rate"],
                                weight_decay=a["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        history, best_loss, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
        for epoch in range(a["num_epochs"]):
            model.train()
            perm = tr[torch.randperm(len(tr), generator=g).numpy()]
            total = 0.0
            for i in range(0, len(perm), a["batch_size"]):
                idx = perm[i:i + a["batch_size"]]
                xb, yb = Xf[idx].to(device), y[idx].to(device)
                loss = criterion(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            model.eval()
            with torch.no_grad():
                dev_lp = model(Xf[dev].to(device)).cpu()
            dev_loss = criterion(dev_lp, y[dev]).item()
            dev_acc = (dev_lp.argmax(-1) == y[dev]).float().mean().item()
            history.append({"epoch": epoch, "train_loss": total / len(tr),
                            "dev_loss": dev_loss, "dev_acc": dev_acc})
            improved = dev_loss < best_loss
            if improved:
                best_loss, best_epoch, bad = dev_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if verbose:
                print(f"    [{name} | fold {fold + 1}/{len(folds)}] epoch {epoch + 1:2d}/{a['num_epochs']}  "
                      f"train loss {total / len(tr):.4f} | dev loss {dev_loss:.4f} acc {dev_acc:.4f}"
                      f"{'  *' if improved else ''}")
            if a.get("patience") is not None and bad >= a["patience"]:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probs = model(Xf[va].to(device)).exp().cpu().numpy()
        preds = probs.argmax(-1)
        yv = y[va].numpy()
        result["accuracies"][fold] = accuracy_score(yv, preds)
        result["macro_f1"][fold] = f1_score(yv, preds, average="macro")
        result["predictions"][fold] = preds
        result["probs"][fold] = probs
        result["selected_epochs"][fold] = best_epoch + 1
        result["history"][fold] = history
        if verbose:
            print(f"  → fold {fold + 1}/{len(folds)}: validation acc {result['accuracies'][fold]:.4f} "
                  f"(best epoch {best_epoch + 1}) | {(time.time() - t0) / 60:.1f} min so far")

    if verbose:
        print(f"[{name}] mean acc = {np.mean(list(result['accuracies'].values())):.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    with cv_file.open("wb") as f:
        pickle.dump(result, f)
    return result


def load_results(results_dir):
    """Load every finished configuration: {name: (cv_result, config)}."""
    out = {}
    for p in sorted(Path(results_dir).glob("*/cv.pkl")):
        with p.open("rb") as f:
            res = pickle.load(f)
        out[p.parent.name] = (res, json.loads((p.parent / "config.json").read_text()))
    return out
