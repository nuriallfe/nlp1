# Paraphrase detection on PAWS-Wiki

Practical 1 of the NLP course. We classify pairs of sentences from PAWS-Wiki as paraphrase or not. The project covers the whole pipeline: a BPE tokenizer trained on the data, three ways of turning a pair of sentences into a vector, three kinds of classifiers (logistic regression, an MLP with one hidden layer and an MLP with two hidden layers, plus a linear SVM as an extra baseline), cross-validation with a Bayesian comparison of the models, and a final evaluation on the test set with an analysis of the mistakes.

The report is in the blog post. This repository contains the code and the experiments behind it.

## Main results

| System | Accuracy |
|---|---|
| Always answering the most common class | 0.558 |
| Best MLP in cross-validation (one hidden layer, n-gram features) | 0.585 (5-fold CV) |
| **Final system: logistic regression on bigram features, on the test set** | **0.583** |


## Repository layout

```
data_exploration.ipynb          Notebook 0: statistics of the dataset
1_tokenizer_and_features.ipynb  Notebook 1: BPE tokenizer and feature representations
2_models_and_evaluation.ipynb   Notebook 2: models, cross-validation, comparison, test set
src/                            Code used by the notebooks (see below)
results_strict/                 Results of notebook 2 (summaries, caches and trained models)
images/, tables/                Figures and tables used in the blog post and the slides
tokenizer_bpe_8000.json         Tokenizer from notebook 1 (used only for the exploration)
requirements.txt                Python packages
```

Modules in `src/`:

| File | Content |
|---|---|
| `dataset.py` | Loads PAWS (`labeled_final`) and removes the training examples with conflicting labels |
| `tokenization.py` | Trains the BPE tokenizer and computes its statistics |
| `encoders.py` | Feature representations: token indicators, token counts, n-gram counts, and how a pair of sentences becomes a vector |
| `experiment.py` | Cross-validation folds, features for each fold, and the search over configurations |
| `loglin.py`, `svm.py`, `smlp.py` | Logistic regression, linear SVM and the MLP (the MLP code starts from the course tutorial) |
| `bayes_rope.py` | Bayesian correlated t-test with a ROPE |
| `analysis.py`, `clslib.py`, `figures.py` | Helpers: result tables and plots, class names, saving figures |

## Setup

We used Python 3.12 and ran everything on a CPU.

```
python -m venv .venv
.venv\Scripts\activate          # on Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is downloaded the first time with the `datasets` library (`google-research-datasets/paws`, configuration `labeled_final`).

## How to run

Run the notebooks in this order, from the repository folder.

1. **`data_exploration.ipynb`** (a few minutes). It only describes the data.
2. **`1_tokenizer_and_features.ipynb`** (minutes). It trains the tokenizers, compares vocabulary sizes and explains the representations. It does not train any classifier.
3. **`2_models_and_evaluation.ipynb`**. It does not need the previous notebooks.
   - Training all the configurations of the cross-validation takes **several hours on a CPU**. Every configuration is saved in `results_strict/` when it finishes, so if you stop the notebook and run it again, finished work is reused. Section 6 loads all the saved results after a restart.
   - The final system (section 9) trains one model on the whole train split and takes a few minutes. Run it once: it is the only place where the test split is used.

## Notes

- **Preprocessing inside each fold.** The BPE tokenizer and the n-gram vocabulary are trained again for every fold, using only its training part. They are saved in `results_strict/feature_cache_*`.
- **Generated files.** `results_strict/` contains matrices and models that can be recreated by running notebook 2. Summaries (`*.csv`) and `final_test/test_predictions.csv` are the files to look at.

