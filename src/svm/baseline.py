"""
SVM baseline: 5-fold cross-validation on the training set only, with default
SVM settings (RBF kernel, C=1, gamma="scale").

Two tasks:
    binary  benign vs malware
    family  which of the 6 malware types (malware files only)

Run from the project root:
    python src/svm/baseline.py

Writes results/svm/baseline_cv.json. The test set is not used here.
"""
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from data import load_data  # noqa: E402
from evaluation import (TASKS, class_names, cross_validate_model,  # noqa: E402
                        print_per_class, print_scores)
from pipeline import make_svm  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "svm"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for task in TASKS:
        X, _, y, _ = load_data(task)
        names = class_names(task)

        start = time.time()
        svm = cross_validate_model(make_svm(X.columns), X, y, names, n_jobs=5)
        svm["seconds"] = round(time.time() - start, 1)
        results[task] = {"rows": len(X), "features": X.shape[1], "svm": svm}

        print(f"\n=== {task}: {len(X)} files, {X.shape[1]} features ===")
        print_scores(f"svm ({svm['seconds']}s)", svm)
        print_per_class(svm)

    (OUT_DIR / "baseline_cv.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_DIR / 'baseline_cv.json'}")


if __name__ == "__main__":
    main()
