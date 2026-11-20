"""
Follow-up to the tuning: the best max_features (0.2) was the largest value
tried, so check larger ones. All other settings stay at the tuned values.

Run from the project root (after src/random_forest/tune.py):
    python src/random_forest/max_features.py

Writes results/random_forest/max_features_curve.csv and updates
tuned_params.json with the best value per task. The test set is not used here.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import load_data  # noqa: E402

from evaluation import SEED, make_cv  # noqa: E402
VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "random_forest"


def main():
    params_file = OUT_DIR / "tuned_params.json"
    tuned = json.loads(params_file.read_text())
    cv = make_cv()
    rows = []

    for task in ("binary", "family"):
        X, _, y, _ = load_data(task)
        fixed = {k: v for k, v in tuned[task]["best_params"].items() if k != "max_features"}
        search = GridSearchCV(
            RandomForestClassifier(random_state=SEED, n_jobs=1, **fixed),
            {"max_features": VALUES}, scoring="f1_macro", cv=cv, n_jobs=-1, refit=False)
        search.fit(X, y)

        print(f"\n=== {task} ===")
        for value, mean, std in zip(VALUES, search.cv_results_["mean_test_score"],
                                    search.cv_results_["std_test_score"]):
            rows.append({"task": task, "max_features": value,
                         "cv_macro_f1": mean, "cv_macro_f1_std": std})
            print(f"max_features {value:.1f}  macro-F1 {mean:.4f} (+/- {std:.4f})")

        tuned[task]["best_params"]["max_features"] = search.best_params_["max_features"]
        tuned[task]["tuned_cv_macro_f1"] = float(search.best_score_)
        tuned[task]["tuned_cv_macro_f1_std"] = float(
            search.cv_results_["std_test_score"][search.best_index_])
        print("best:", search.best_params_["max_features"])

    pd.DataFrame(rows).to_csv(OUT_DIR / "max_features_curve.csv", index=False)
    params_file.write_text(json.dumps(tuned, indent=2))
    print(f"\nsaved {OUT_DIR / 'max_features_curve.csv'} and updated {params_file.name}")


if __name__ == "__main__":
    main()
