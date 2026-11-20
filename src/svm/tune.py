"""
SVM tuning: try 30 random combinations of settings per task and score each
with the same 5-fold cross-validation as the baseline.

Run from the project root (after src/svm/baseline.py):
    python src/svm/tune.py

Writes, in results/svm/:
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
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from data import load_data  # noqa: E402
from evaluation import SEED, TASKS, make_cv  # noqa: E402
from pipeline import make_svm  # noqa: E402

N_CANDIDATES = 30
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "svm"

# the settings to choose between; C and gamma are drawn at random from a range,
# evenly on a log scale (as likely between 0.1 and 1 as between 10 and 100)
SEARCH_SPACE = {
    "svm__C": loguniform(0.1, 300),         # how hard the SVM tries to fit every training file
    "svm__gamma": loguniform(1e-4, 1e-1),   # how far the influence of one training file reaches
    "svm__class_weight": ["balanced", None],  # extra weight for rare classes
}


def main():
    baseline = json.loads((OUT_DIR / "baseline_cv.json").read_text())
    summary = {}

    for task in TASKS:
        X, _, y, _ = load_data(task)
        start = time.time()
        search = RandomizedSearchCV(
            make_svm(X.columns, cache_size=1000), SEARCH_SPACE, n_iter=N_CANDIDATES,
            scoring="f1_macro", cv=make_cv(), random_state=SEED, n_jobs=-1, refit=False)
        search.fit(X, y)
        seconds = round(time.time() - start, 1)

        table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        keep = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time"] + \
               [c for c in table.columns if c.startswith("param_")]
        table[keep].to_csv(OUT_DIR / f"tuning_{task}.csv", index=False)

        base_f1 = float(np.mean(baseline[task]["svm"]["macro_f1_per_fold"]))
        best = table.iloc[0]
        best_params = {k.replace("svm__", ""): v for k, v in search.best_params_.items()}
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
