"""
Random Forest baseline: 5-fold cross-validation on the training set only.

Two tasks:
    binary  benign vs malware
    family  which of the 6 malware types (malware files only)

Run from the project root:
    python src/random_forest/baseline.py

Writes results/random_forest/baseline_cv.json. The test set is not used here.
"""
import json
import sys
import time
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import load_data  # noqa: E402
from evaluation import (SEED, TASKS, class_names, cross_validate_model,  # noqa: E402
                        print_per_class, print_scores)

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "random_forest"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for task in TASKS:
        X, _, y, _ = load_data(task)
        names = class_names(task)

        # "always guess the most common class": the score any real model must beat
        dummy = cross_validate_model(DummyClassifier(strategy="most_frequent"), X, y, names)

        start = time.time()
        forest = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                        random_state=SEED, n_jobs=-1)
        rf = cross_validate_model(forest, X, y, names)
        rf["seconds"] = round(time.time() - start, 1)

        results[task] = {"rows": len(X), "features": X.shape[1],
                         "dummy": dummy, "random_forest": rf}

        print(f"\n=== {task}: {len(X)} files, {X.shape[1]} features ===")
        print_scores("always-guess-biggest-class", dummy)
        print_scores(f"random forest ({rf['seconds']}s)", rf)
        print_per_class(rf)

    (OUT_DIR / "baseline_cv.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_DIR / 'baseline_cv.json'}")


if __name__ == "__main__":
    main()
