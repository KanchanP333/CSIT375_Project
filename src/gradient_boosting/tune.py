"""
Gradient Boosting tuning: try 30 random hyperparameter combinations per task
and score each with the same 5-fold cross-validation as the baseline.

Run from the project root (after src/gradient_boosting/baseline.py):
    python src/gradient_boosting/tune.py

Writes, in results/gradient_boosting/:
    tuning_<task>.csv    every combination tried, best first
    tuned_params.json    the winning settings and scores per task
The test set is not used here.
"""
import os

# one thread per model, so that many models can be trained side by side
# without competing for the same cores (must be set before scikit-learn loads)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import loguniform  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import load_data  # noqa: E402
from evaluation import SEED, TASKS, make_cv  # noqa: E402

N_CANDIDATES = 30
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "gradient_boosting"

SEARCH_SPACE = {
    "learning_rate": loguniform(0.02, 0.3),   # how much each new tree corrects the previous ones
    "max_iter": [100, 200, 400, 600],         # number of boosting iterations
    "max_leaf_nodes": [15, 31, 63, 127],      # size of each tree
    "min_samples_leaf": [5, 10, 20, 40],      # smallest group of files a leaf may hold
    "l2_regularization": [0, 0.1, 1, 10],     # penalty on large leaf values
    "max_features": [0.2, 0.5, 1.0],          # share of features each split may choose from
    "class_weight": ["balanced", None],       # extra weight for rare classes
}


def workers():
    """Models to train at once: one per core, but no more than the free
    memory allows (about 0.5 GB per worker, keeping 2 GB spare)."""
    free_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    return max(1, min(os.cpu_count(), int((free_gb - 2) / 0.5)))


def main():
    baseline = json.loads((OUT_DIR / "baseline_cv.json").read_text())
    summary = {}
    n_jobs = workers()
    print(f"training {n_jobs} models at a time")

    for task in TASKS:
        X, _, y, _ = load_data(task)
        start = time.time()
        search = RandomizedSearchCV(
            HistGradientBoostingClassifier(early_stopping=False, random_state=SEED),
            SEARCH_SPACE, n_iter=N_CANDIDATES, scoring="f1_macro", cv=make_cv(),
            random_state=SEED, n_jobs=n_jobs, refit=False)
        search.fit(X, y)
        seconds = round(time.time() - start, 1)

        table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        keep = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time"] + \
               [c for c in table.columns if c.startswith("param_")]
        table[keep].to_csv(OUT_DIR / f"tuning_{task}.csv", index=False)

        base_f1 = float(np.mean(baseline[task]["gradient_boosting"]["macro_f1_per_fold"]))
        best = table.iloc[0]
        best_params = {k: (v.item() if hasattr(v, "item") else v)
                       for k, v in search.best_params_.items()}
        summary[task] = {
            "best_params": best_params,
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
        print("best settings:", best_params)
        print("top 8:")
        print(table[keep].head(8).to_string(index=False))

    (OUT_DIR / "tuned_params.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR / 'tuned_params.json'}")


if __name__ == "__main__":
    main()
