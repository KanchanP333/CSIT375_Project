"""
Random Forest tuning: try 30 random combinations of settings per task and
score each with the same 5-fold cross-validation as the baseline.

Run from the project root (after src/random_forest/baseline.py):
    python src/random_forest/tune.py

Writes, in results/random_forest/:
    tuning_<task>.csv    every combination tried, best first
    tuned_params.json    the winning settings and scores per task
The test set is not used here.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import load_data  # noqa: E402

from evaluation import SEED, make_cv  # noqa: E402
N_CANDIDATES = 30
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "random_forest"

# the settings to choose between
SEARCH_SPACE = {
    "n_estimators": [300, 500],                  # number of trees
    "max_depth": [None, 20, 30, 40],             # how many questions deep a tree may go
    "min_samples_leaf": [1, 2, 4],               # smallest group a tree may end on
    "min_samples_split": [2, 5, 10],             # smallest group a tree may split further
    "max_features": ["sqrt", "log2", 0.1, 0.2],  # features each split may choose from
    "criterion": ["gini", "entropy"],            # how a split's quality is measured
    "class_weight": ["balanced", "balanced_subsample", None],  # extra weight for rare classes
}


def main():
    baseline = json.loads((OUT_DIR / "baseline_cv.json").read_text())
    cv = make_cv()
    summary = {}

    for task in ("binary", "family"):
        X, _, y, _ = load_data(task)
        start = time.time()
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            SEARCH_SPACE, n_iter=N_CANDIDATES, scoring="f1_macro", cv=cv,
            random_state=SEED, n_jobs=-1, refit=False)
        search.fit(X, y)
        seconds = round(time.time() - start, 1)

        table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        keep = ["rank_test_score", "mean_test_score", "std_test_score"] + \
               [c for c in table.columns if c.startswith("param_")]
        table[keep].to_csv(OUT_DIR / f"tuning_{task}.csv", index=False)

        base_f1 = float(np.mean(baseline[task]["random_forest"]["macro_f1_per_fold"]))
        best = table.iloc[0]
        summary[task] = {
            "best_params": search.best_params_,
            "tuned_cv_macro_f1": float(best.mean_test_score),
            "tuned_cv_macro_f1_std": float(best.std_test_score),
            "baseline_cv_macro_f1": base_f1,
            "worst_candidate_macro_f1": float(table.mean_test_score.min()),
            "candidates": N_CANDIDATES,
            "seconds": seconds,
        }

        print(f"\n=== {task}: {N_CANDIDATES} combinations x 5 folds, {seconds}s ===")
        print(f"baseline macro-F1 {base_f1:.4f}")
        print(f"best     macro-F1 {best.mean_test_score:.4f} (+/- {best.std_test_score:.4f})")
        print(f"worst    macro-F1 {table.mean_test_score.min():.4f}")
        print("best settings:", search.best_params_)
        print("top 5:")
        print(table[keep].head(5).to_string(index=False))

    (OUT_DIR / "tuned_params.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR / 'tuned_params.json'}")


if __name__ == "__main__":
    main()
