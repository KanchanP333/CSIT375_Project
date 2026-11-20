"""
Follow-up to the SVM tuning: the best C values in the random search were near
the top of the range searched (up to 300), so check larger ones, with gamma
values around the best found. class_weight stays at None, which won both tasks.

Run from the project root (after src/svm/tune.py):
    python src/svm/tune_large_c.py

Writes results/svm/large_c_grid.csv and updates tuned_params.json if a
combination beats the earlier best. The test set is not used here.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from data import load_data  # noqa: E402
from evaluation import TASKS, make_cv  # noqa: E402
from pipeline import make_svm  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "svm"
GRID = {"svm__C": [300, 1000, 3000, 10000, 30000], "svm__gamma": [0.002, 0.005, 0.01]}


def main():
    params_file = OUT_DIR / "tuned_params.json"
    tuned = json.loads(params_file.read_text())
    rows = []

    for task in TASKS:
        X, _, y, _ = load_data(task)
        start = time.time()
        search = GridSearchCV(make_svm(X.columns, class_weight=None, cache_size=1000), GRID,
                              scoring="f1_macro", cv=make_cv(), n_jobs=-1, refit=False)
        search.fit(X, y)
        seconds = round(time.time() - start, 1)

        r = search.cv_results_
        previous = tuned[task]["tuned_cv_macro_f1"]
        print(f"\n=== {task}: {len(r['params'])} combinations x 5 folds, {seconds}s ===")
        print(f"previous best macro-F1 {previous:.4f}  (C={tuned[task]['best_params']['C']:.0f}, "
              f"gamma={tuned[task]['best_params']['gamma']:.4f})")
        for p, mean, std, fit in zip(r["params"], r["mean_test_score"], r["std_test_score"],
                                     r["mean_fit_time"]):
            rows.append({"task": task, "C": p["svm__C"], "gamma": p["svm__gamma"],
                         "cv_macro_f1": mean, "cv_macro_f1_std": std, "mean_fit_seconds": fit})
            print(f"C {p['svm__C']:>5}  gamma {p['svm__gamma']:<6}  macro-F1 {mean:.4f} "
                  f"(+/- {std:.4f})  {fit:.0f}s per fit")

        if search.best_score_ > previous:
            tuned[task]["best_params"] = {"C": search.best_params_["svm__C"],
                                          "gamma": search.best_params_["svm__gamma"],
                                          "class_weight": None}
            tuned[task]["tuned_cv_macro_f1"] = float(search.best_score_)
            tuned[task]["tuned_cv_macro_f1_std"] = float(r["std_test_score"][search.best_index_])
            print("new best:", tuned[task]["best_params"])
        else:
            print("no improvement; earlier settings kept")

    pd.DataFrame(rows).to_csv(OUT_DIR / "large_c_grid.csv", index=False)
    params_file.write_text(json.dumps(tuned, indent=2))
    print(f"\nsaved {OUT_DIR / 'large_c_grid.csv'} and {params_file.name}")


if __name__ == "__main__":
    main()
