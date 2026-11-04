"""Bayesian model comparison with a ROPE.

Bayesian correlated t-test (Corani & Benavoli, 2015) on cross-validation scores.
Designed for the DataFrame returned by `analysis.make_data_frame`, which has the
columns `model`, `accuracy` and `run`.

Convention: the difference is always score(A) - score(B).
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def scores_by_model(df, model_col="model", score_col="accuracy",
                    run_col="run", fold_col=None):
    """Long DataFrame -> wide table: one row per (run, fold), one column per model.

    If there is no fold column, folds are numbered by order of appearance within
    each (model, run). Since all models use the same folds, row i of one model
    corresponds to the same fold as row i of the others.
    """
    d = df.copy()
    if fold_col is None:
        d["_fold"] = d.groupby([model_col, run_col]).cumcount()
        fold_col = "_fold"
    wide = d.pivot_table(index=[run_col, fold_col], columns=model_col, values=score_col)
    complete = wide.dropna()
    if len(complete) < len(wide):
        print(f"Warning: {len(wide) - len(complete)} (run, fold) rows dropped "
                f"because some model has no score.")
    return complete


def difference_posterior(diff, k):
    """Posterior of the mean performance difference (correlated t-test).

    diff : per-fold differences (and per-run, if there are several), score_A - score_B
    k    : number of cross-validation folds (rho = 1/k)
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    rho = 1.0 / k
    var = diff.var(ddof=1)
    # Folds share training data, so they are not independent: the rho/(1-rho)
    # term inflates the variance (Nadeau & Bengio correction).
    scale = np.sqrt((1.0 / n + rho / (1.0 - rho)) * var)
    return stats.t(df=n - 1, loc=diff.mean(), scale=max(scale, 1e-12))


def compare_models(wide, k, rope, models=None, pairs=None, cred=0.95):
    """Compare pairs of models. Returns (table, posteriors).

    rope  : ROPE half-width (e.g. 0.01 = 1 accuracy point)
    models: models to include (all by default); every pair is compared
    pairs : list of specific (A, B) tuples, if you don't want every pair
    cred  : HDI mass (the posterior is a symmetric t, so HDI = central interval)
    """
    models = list(models) if models is not None else list(wide.columns)
    pairs = pairs if pairs is not None else list(combinations(models, 2))
    rows, posteriors = [], {}
    for a, b in pairs:
        post = difference_posterior(wide[a] - wide[b], k)
        lo, hi = post.interval(cred)
        p_b = post.cdf(-rope)                      # P(B better): difference < -rope
        p_rope = post.cdf(rope) - post.cdf(-rope)  # P(equivalent): inside the ROPE
        p_a = post.sf(rope)                        # P(A better): difference > +rope

        if -rope <= lo and hi <= rope:
            decision = "equivalent"
        elif lo > rope:
            decision = f"{a} better"
        elif hi < -rope:
            decision = f"{b} better"
        else:
            decision = "undecided"

        rows.append({
            "A": a, "B": b,
            "mean_diff": post.mean(),
            f"HDI{int(cred * 100)}_low": lo,
            f"HDI{int(cred * 100)}_high": hi,
            "P(A better)": p_a, "P(ROPE)": p_rope, "P(B better)": p_b,
            "decision": decision,
        })
        posteriors[(a, b)] = post
    return pd.DataFrame(rows), posteriors


def plot_posteriors(posteriors, rope, cred=0.95, ncols=3):
    """One plot per pair: posterior density, shaded HDI and the ROPE in gray."""
    n = len(posteriors)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.2 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, ((a, b), post) in zip(flat, posteriors.items()):
        lo99, hi99 = post.interval(0.999)
        x = np.linspace(min(lo99, -1.5 * rope), max(hi99, 1.5 * rope), 600)
        ax.plot(x, post.pdf(x), color="C0")
        ax.axvspan(-rope, rope, color="gray", alpha=0.25, label="ROPE")
        lo, hi = post.interval(cred)
        xs = x[(x >= lo) & (x <= hi)]
        ax.fill_between(xs, post.pdf(xs), color="C0", alpha=0.35, label=f"HDI {int(cred * 100)}%")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_title(f"{a} − {b}", fontsize=9)
        ax.set_xlabel("accuracy difference (A − B)")
        ax.set_yticks([])
    for ax in flat[n:]:
        ax.axis("off")
    flat[0].legend(fontsize=8)
    fig.tight_layout()
    return fig