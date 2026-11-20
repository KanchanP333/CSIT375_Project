"""
Shared scoring code, so every model is cross-validated and reported the same way.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data import TYPE_NAMES

SEED = 42
TASKS = ("binary", "family")


def make_cv():
    """The 5 folds every model uses."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def class_names(task):
    if task == "binary":
        return {0: "Benign", 1: "Malware"}
    return {k: v for k, v in TYPE_NAMES.items() if k != 0}


def cross_validate_model(model, X, y, names, n_jobs=None):
    """Predict every training row from a model that never saw it, then score."""
    cv = make_cv()
    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs)
    labels = sorted(names)
    per_fold_f1 = [f1_score(y.iloc[te], pred[te], average="macro") for _, te in cv.split(X, y)]
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "macro_f1_per_fold": per_fold_f1,
        "macro_f1_std": float(np.std(per_fold_f1)),
        "per_class": classification_report(
            y, pred, labels=labels, target_names=[names[i] for i in labels],
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "labels": [names[i] for i in labels],
    }


def print_scores(title, scores):
    print(f"{title}  accuracy {scores['accuracy']:.4f}  macro-F1 {scores['macro_f1']:.4f} "
          f"(+/- {scores['macro_f1_std']:.4f} across folds)")


def print_per_class(scores):
    print(f"{'class':16}{'precision':>10}{'recall':>8}{'f1':>8}{'files':>8}")
    for name in scores["labels"]:
        c = scores["per_class"][name]
        print(f"{name:16}{c['precision']:>10.3f}{c['recall']:>8.3f}{c['f1-score']:>8.3f}{int(c['support']):>8}")
    print("confusion matrix (rows = true class, columns = predicted):")
    print(np.array(scores["confusion_matrix"]))
