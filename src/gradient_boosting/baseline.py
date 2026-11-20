"""
Gradient Boosting baseline: 5-fold cross-validation on the training set only,
with scikit-learn's default settings (100 boosting iterations, learning rate
0.1) plus class_weight="balanced".

The model is HistGradientBoostingClassifier, scikit-learn's histogram-based
gradient boosting (the same algorithm family as LightGBM and XGBoost's "hist"
method). Trees need no feature scaling, so there is no preprocessing step.

Two tasks:
    binary  benign vs malware
    family  which of the 6 malware types (malware files only)

Run from the project root:
    python src/gradient_boosting/baseline.py

Writes results/gradient_boosting/baseline_cv.json. The test set is not used here.
"""
import json
import sys
import time
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import load_data  # noqa: E402
from evaluation import (SEED, TASKS, class_names, cross_validate_model,  # noqa: E402
                        print_per_class, print_scores)

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "gradient_boosting"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for task in TASKS:
        X, _, y, _ = load_data(task)
        names = class_names(task)

        # early stopping is switched off so that every fold trains the same
        # fixed number of iterations
        model = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                               random_state=SEED)
        start = time.time()
        scores = cross_validate_model(model, X, y, names)
        scores["seconds"] = round(time.time() - start, 1)
        results[task] = {"rows": len(X), "features": X.shape[1], "gradient_boosting": scores}

        print(f"\n=== {task}: {len(X)} files, {X.shape[1]} features ===")
        print_scores(f"gradient boosting ({scores['seconds']}s)", scores)
        print_per_class(scores)

    (OUT_DIR / "baseline_cv.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_DIR / 'baseline_cv.json'}")


if __name__ == "__main__":
    main()
