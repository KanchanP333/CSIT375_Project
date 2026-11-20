"""
Gradient Boosting final evaluation: train once on the whole training set with the
tuned settings, then score once on the held-out test set.

Run from the project root (after the tuning scripts):
    python src/gradient_boosting/final.py

Writes:
    models/gradient_boosting/<task>.joblib              the trained model
    results/gradient_boosting/test_results.json          test metrics per task
    results/gradient_boosting/test_predictions_<task>.csv  one row per test file
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import PROCESSED, TYPE_NAMES, load_data  # noqa: E402

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "gradient_boosting"
MODEL_DIR = ROOT / "models" / "gradient_boosting"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tuned = json.loads((OUT_DIR / "tuned_params.json").read_text())
    test_ids = pd.read_csv(PROCESSED / "test.csv", usecols=["SHA256", "is_malware"])
    results = {}

    for task in ("binary", "family"):
        X_train, X_test, y_train, y_test = load_data(task)
        names = ({0: "Benign", 1: "Malware"} if task == "binary"
                 else {k: v for k, v in TYPE_NAMES.items() if k != 0})
        labels = sorted(names)

        model = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False,
                                               **tuned[task]["best_params"])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        report = classification_report(y_test, pred, labels=labels,
                                       target_names=[names[i] for i in labels],
                                       output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, pred, labels=labels)
        results[task] = {
            "train_rows": len(X_train), "test_rows": len(X_test),
            "params": tuned[task]["best_params"],
            "cv_macro_f1": tuned[task]["tuned_cv_macro_f1"],
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "per_class": report,
            "confusion_matrix": matrix.tolist(),
            "labels": [names[i] for i in labels],
        }
        if task == "binary":
            results[task]["roc_auc"] = roc_auc_score(y_test, proba[:, 1])

        # one row per test file, with its ID, so any prediction can be traced
        ids = test_ids if task == "binary" else test_ids[test_ids.is_malware == 1]
        out = pd.DataFrame({"SHA256": ids["SHA256"].values,
                            "true": [names[v] for v in y_test],
                            "predicted": [names[v] for v in pred],
                            "confidence": proba.max(axis=1).round(4)})
        out["correct"] = out.true == out.predicted
        out.to_csv(OUT_DIR / f"test_predictions_{task}.csv", index=False)
        joblib.dump(model, MODEL_DIR / f"{task}.joblib", compress=3)

        r = results[task]
        print(f"\n=== {task}: trained on {len(X_train)}, tested on {len(X_test)} ===")
        print(f"accuracy {r['accuracy']:.4f}   macro-F1 {r['macro_f1']:.4f}   "
              f"(cross-validation macro-F1 was {r['cv_macro_f1']:.4f})")
        if task == "binary":
            print(f"ROC-AUC {r['roc_auc']:.4f}")
        print(f"{'class':16}{'precision':>10}{'recall':>8}{'f1':>8}{'files':>8}")
        for name in r["labels"]:
            c = report[name]
            print(f"{name:16}{c['precision']:>10.3f}{c['recall']:>8.3f}{c['f1-score']:>8.3f}{int(c['support']):>8}")
        print("confusion matrix (rows = true class, columns = predicted):")
        print(np.array(matrix))

    (OUT_DIR / "test_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_DIR / 'test_results.json'}")


if __name__ == "__main__":
    main()
