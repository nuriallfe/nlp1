import numpy as np
import itertools
import random
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from . import loglin
from . import smlp
from . import clslib


def sentence_groups(pairs):
    """
    Groups the indices of PAWS pairs that are connected through a shared sentence.

    In PAWS the same sentence appears in several pairs (often with different labels). Two pairs are in the
    same group if they share a sentence, directly or through other pairs (connected components, union-find).
    Returns a list of groups, each a list of indices into `pairs`.
    """
    parent = {}

    def find(s):
        parent.setdefault(s, s)
        while parent[s] != s:
            parent[s] = parent[parent[s]]  # path compression
            s = parent[s]
        return s

    for (sentence1, sentence2), label in pairs:
        parent[find(sentence1)] = find(sentence2)

    groups = {}
    for i, ((sentence1, sentence2), label) in enumerate(pairs):
        groups.setdefault(find(sentence1), []).append(i)
    return list(groups.values())


def cross_validation_split(pairs, k, seed=0, group_by_sentence=True):
    """
    Shuffles a list of labelled texts and splits the result into k folds.

    pairs: lists of tuples (text, label)
    k: number of folds
    seed: for reproducible shuffling
    group_by_sentence: if True (default) and the texts are PAWS pairs (sentence1, sentence2), pairs that share
        a sentence always fall in the same fold, so no sentence of a validation fold is seen in training.
        (A random split leaks: ~95% of the validation pairs have a sentence that is also in training,
        and its training label tends to be the opposite one.)
    """

    pairs = list(pairs)

    if group_by_sentence:
        # Assign whole groups to folds: biggest groups first (random order among equal sizes),
        # each one to the fold that currently has fewer pairs, so the folds are balanced.
        rng = random.Random(seed)
        groups = sentence_groups(pairs)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        parts = [[] for _ in range(k)]
        for group in groups:
            min(parts, key=len).extend(group)
        ids_parts = [np.array(sorted(part), dtype=object) for part in parts]
    else:
        # Shuffle the pairs
        random.seed(seed)
        ids = list(range(len(pairs)))
        random.shuffle(ids)

        # Split without coercing (text, int) tuples into a string array
        ids_parts = np.array_split(np.array(ids, dtype=object), k)
    
    # Create a dictionary to store the training and validation sets for each fold
    cross_validation_sets = {}
    for i in range(k):
        validation_set = [pairs[j] for j in ids_parts[i]] 
        training_ids = list(itertools.chain.from_iterable(part for k, part in enumerate(ids_parts) if k != i))
        training_set = [pairs[j] for j in training_ids]
        cross_validation_sets[i] = {
            "training": training_set, 
            "validation": validation_set, 
            'training_ids': training_ids,   # these are not often needed, but we return it just in case
            'validation_ids': ids_parts[i]  # these are not often needed, but we return it just in case
        }

    return cross_validation_sets

def cross_validation_experiment(data_splits, outcome_space, encoder: 'Encoder', model_option:str, save_to="tmp", loglin_args=dict(), smlp_args=dict()):
    """
    Trains and evaluates k models using a k-fold data split for cross validation.
    This function returns a dictionary with
        - models: the paths to the k models trained (the model objects aren't returned, they are stored in disk)
        - predictions: the predictions by each of the k models
        - accuracies: the accuracy of each trained model on its corresponding validation partition

    data_splits: the training/validation folds
    encoder: an instance of Encoder
    model_option: for now this only supports 'logistic'
        which tells this function to use train_logistic_model (defined earlier in this notebook)
    save_to: folder where to store models
    smlp_args: dict with training arguments for smlp.train_neural_network_model
    """

    models, predictions, accuracies = OrderedDict(), OrderedDict(), OrderedDict()



    for fold in tqdm(data_splits.keys()):
        training_set = data_splits[fold]["training"]
        validation_set = data_splits[fold]["validation"]

        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True)

        if model_option == 'logistic':
            llmodel = loglin.fit(training_set, encoder, save_to=f'{fold_folder}/model', **loglin_args) 
            models[fold] = f'{fold_folder}/model'
            
            predictions[fold] = loglin.predict([text for text, gold_label in validation_set], llmodel, encoder)    
            gold_labels_validation = [gold_label for text, gold_label in validation_set]
            accuracies[fold] = accuracy_score(gold_labels_validation, predictions[fold])
        elif model_option == 'mlp':            
            metrics, best_epoch, predictions[fold] = smlp.train_neural_network_model(
                training_set, validation_set, len(outcome_space), encoder, save_to=f'{fold_folder}/model', **smlp_args
            )
            models[fold] = f'{fold_folder}/model'
            accuracies[fold] = metrics[best_epoch]['val_metrics']['accuracy']            
        else:
            raise ValueError(f"Unkonwn model_option={model_option}")

    return {'models': models, 'predictions': predictions, 'accuracies': accuracies}


class WrapperForLoglinearModel(clslib.ProbabilisticModel):
    """
    Wrapper to treat an instance of sklearn LogisticRegression in the API of a ProbabilisticModel
    """
    def __init__(self, encoder, model):  
        """
        text encoder used for training and the trained model
        """
        self.encoder = encoder
        self.model = model
        
    def __call__(self, x):
        """Predicts a pmf for x"""
        r = loglin.predict_probs([x], self.model, self.encoder)
        return r[0]


def ensemble_loglin(outcome_space, encoder, decision_rule, model_family, seeds: list, num_folds: int, results_folder):
    """
    Ensemble via MajorityVoteClassifier a number of instances of logistic regressors, all stored in the results folder.
    
    outcome_space: NamedClasses for the dataset
    encoder: text encoder used in training
    decision_rule: how to map 1 pmf to 1 label
    model_family: the string you used to name a subfolder under the results folder
    seeds: a list of seeds (they name subfolders under the model_family folder)
    num_folds: number of folds in cross validation (these name subfolders under the seed folder)
    results_folder: where everything is saved
    """
    models = []
    for seed in seeds:
        for f in range(num_folds):
            models.append(WrapperForLoglinearModel(encoder, loglin.load(f"{results_folder}/{model_family}/seed-{seed}/fold-{f}/model")))
    return clslib.MajorityVoteClassifier(outcome_space, models, decision_rule)

class WrapperForNeuralModel(clslib.ProbabilisticModel):
    """
    Wrapper to treat an instance of our torch softmax regressor in the API of a ProbabilisticModel
    """
    def __init__(self, encoder, model):        
        """text encoder used for training and trained model"""
        self.encoder = encoder
        self.model = model
        
    def __call__(self, x):
        """Predicts a pmf for x"""
        r = smlp.predict_probs([x], self.model, self.encoder, batch_size=1) # of course, batching multiple texts is faster for a large dataset, but this is more didactic 
        return r[0]

def ensemble_neural(outcome_space, encoder, decision_rule, model_family, seeds: list, num_folds: int, results_folder):
    """
    Ensemble via MajorityVoteClassifier a number of instances of neural softmax regressors, all stored in the results folder.
    
    outcome_space: NamedClasses for the dataset
    encoder: text encoder used in training
    decision_rule: how to map 1 pmf to 1 label
    model_family: the string you used to name a subfolder under the results folder
    seeds: a list of seeds (they name subfolders under the model_family folder)
    num_folds: number of folds in cross validation (these name subfolders under the seed folder)
    results_folder: where everything is saved
    """
    models = []
    for seed in seeds:
        for f in range(num_folds):
            models.append(WrapperForNeuralModel(encoder, smlp.load(f"{results_folder}/{model_family}/seed-{seed}/fold-{f}/model")))
    return clslib.MajorityVoteClassifier(outcome_space, models, decision_rule)            
            