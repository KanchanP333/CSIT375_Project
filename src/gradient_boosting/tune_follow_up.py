"""
Follow-up to the Gradient Boosting tuning. In the random search, the winning
max_features (0.2) was the smallest value offered for both tasks, and the
winning max_iter for `binary` (600) was the largest offered. This grid search
checks values beyond those edges, with every other hyperparameter held at the
random-search winner.

Run from the project root (after src/gradient_boosting/tune.py):
    python src/gradient_boosting/tune_follow_up.py

Writes results/gradient_boosting/follow_up_grid.csv and updates
tuned_params.json if a combination beats the earlier best. The test set is
not used here.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per model; see tune.py

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from data import load_data  # noqa: E402
from evaluation import SEED, TASKS, make_cv  # noqa: E402
from tune import workers  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "gradient_boosting"
GRIDS = {
    "binary": {"max_features": [0.1, 0.2], "max_iter": [600, 1000, 1500]},
    "family": {"max_features": [0.1, 0.2], "max_iter": [100, 200]},
}


def main():
    params_file = OUT_DIR / "tuned_params.json"
    tuned = json.loads(params_file.read_text())
    rows = []

    for task in TASKS:
        X, _, y, _ = load_data(task)
        fixed = {k: v for k, v in tuned[task]["best_params"].items() if k not in GRIDS[task]}
        start = time.time()
        search = GridSearchCV(
            HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **fixed),
            GRIDS[task], scoring="f1_macro", cv=make_cv(), n_jobs=workers(), refit=False)
        search.fit(X, y)
        seconds = round(time.time() - start, 1)

        r = search.cv_results_
        previous = tuned[task]["tuned_cv_macro_f1"]
        print(f"\n=== {task}: {len(r['params'])} combinations x 5 folds, {seconds}s ===")
        print(f"previous best macro-F1 {previous:.4f}")
        for p, mean, std in zip(r["params"], r["mean_test_score"], r["std_test_score"]):
            rows.append({"task": task, **p, "cv_macro_f1": mean, "cv_macro_f1_std": std})
            print(f"max_features {p['max_features']}  max_iter {p['max_iter']:>5}  "
                  f"macro-F1 {mean:.4f} (+/- {std:.4f})")

        if search.best_score_ > previous:
            tuned[task]["best_params"].update(search.best_params_)
            tuned[task]["tuned_cv_macro_f1"] = float(search.best_score_)
            tuned[task]["tuned_cv_macro_f1_std"] = float(r["std_test_score"][search.best_index_])
            print("new best:", tuned[task]["best_params"])
        else:
            print("no improvement; earlier settings kept")

    pd.DataFrame(rows).to_csv(OUT_DIR / "follow_up_grid.csv", index=False)
    params_file.write_text(json.dumps(tuned, indent=2))
    print(f"\nsaved {OUT_DIR / 'follow_up_grid.csv'} and {params_file.name}")


if __name__ == "__main__":
    main()
