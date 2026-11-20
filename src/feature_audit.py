"""
Feature audit: look for features that, like the compile date, let a model
score well for reasons unrelated to what a file does.

Two checks, both with 5-fold cross-validation on the training set only:

1. Single-feature score: for each of the 299 features, how well a small
   decision tree does using that feature alone. A feature that scores high on
   its own deserves a look at what it means.
2. Leave-one-out: retrain the Random Forest without one header field at a
   time (all 30), and without the version-stamp fields as a group, and
   measure the change in macro-F1.

The forest uses the tuned settings but 100 trees instead of 300, to keep the
run short; scores are compared with a 100-tree reference, not with the final
300-tree results.

Run from the project root (after src/random_forest/tune.py):
    python src/feature_audit.py

Writes results/feature_audit/single_feature.csv and leave_one_out.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from data import load_data  # noqa: E402
from evaluation import SEED, TASKS, make_cv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "feature_audit"

# header fields that record the build toolchain or a version number chosen by
# the developer, rather than the structure or behaviour of the program
VERSION_STAMPS = ["hdr_MajorLinkerVersion", "hdr_MinorLinkerVersion",
                  "hdr_MajorOperatingSystemVersion", "hdr_MinorOperatingSystemVersion",
                  "hdr_MajorSubsystemVersion", "hdr_MinorSubsystemVersion",
                  "hdr_MajorImageVersion", "hdr_MinorImageVersion"]


def cv_f1(model, X, y, n_jobs=None):
    scores = cross_val_score(model, X, y, scoring="f1_macro", cv=make_cv(), n_jobs=n_jobs)
    return scores.mean(), scores.std()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tuned = json.loads((ROOT / "results" / "random_forest" / "tuned_params.json").read_text())
    single_rows, loo_rows = [], []

    for task in TASKS:
        X, _, y, _ = load_data(task)

        print(f"\n=== {task}: single-feature score (top 15 of {X.shape[1]}) ===")
        rows = []
        for col in X.columns:
            tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED)
            mean, std = cv_f1(tree, X[[col]], y, n_jobs=5)
            rows.append({"task": task, "feature": col, "cv_macro_f1_alone": mean, "std": std})
        table = pd.DataFrame(rows).sort_values("cv_macro_f1_alone", ascending=False)
        single_rows.append(table)
        print(table.head(15)[["feature", "cv_macro_f1_alone"]].to_string(index=False))

        params = dict(tuned[task]["best_params"], n_estimators=100)

        def forest_score(columns):
            return cv_f1(RandomForestClassifier(random_state=SEED, n_jobs=-1, **params),
                         X[columns], y)

        cols = list(X.columns)
        ref, ref_std = forest_score(cols)
        print(f"\n=== {task}: leave-one-out, reference macro-F1 {ref:.4f} (+/- {ref_std:.4f}) ===")
        variants = {f"without {c}": [k for k in cols if k != c]
                    for c in cols if c.startswith("hdr_")}
        variants["without the 8 version stamps"] = [c for c in cols if c not in VERSION_STAMPS]
        variants["without version stamps and hdr_CheckSum"] = [
            c for c in cols if c not in VERSION_STAMPS + ["hdr_CheckSum"]]
        for label, keep in variants.items():
            mean, std = forest_score(keep)
            loo_rows.append({"task": task, "variant": label, "cv_macro_f1": mean, "std": std,
                             "change": mean - ref, "reference": ref, "reference_std": ref_std})
        out = pd.DataFrame([r for r in loo_rows if r["task"] == task]).sort_values("change")
        print(out[["variant", "cv_macro_f1", "std", "change"]].head(12).to_string(index=False))
        print("...")
        print(out[["variant", "cv_macro_f1", "std", "change"]].tail(3).to_string(index=False))

    pd.concat(single_rows).to_csv(OUT_DIR / "single_feature.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(OUT_DIR / "leave_one_out.csv", index=False)
    print(f"\nsaved single_feature.csv and leave_one_out.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
