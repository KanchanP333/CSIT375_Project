"""
Random Forest analysis: which features the model relies on, and how the score
changes when groups of features are removed.

Run from the project root (after src/random_forest/final.py):
    python src/random_forest/analysis.py

Writes, in results/random_forest/:
    feature_importance_<task>.csv   every feature, most important first
    importance_by_group.csv         importance summed per feature group
    ablation.csv                    CV score with feature groups removed

Feature importance uses the saved final models. Permutation importance is
measured on the test set; it only describes the finished model and changes no
setting. The ablation retrains with 5-fold cross-validation on the training
set only.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import load_data  # noqa: E402
from evaluation import SEED, TASKS, class_names, cross_validate_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "random_forest"
MODEL_DIR = ROOT / "models" / "random_forest"
GROUPS = {"hdr": "PE header", "sec": "PE sections", "dll": "Imported DLLs", "api": "API calls"}


def group_of(feature):
    return GROUPS[feature.split("_")[0]]


def importance(task):
    """Two views of importance for the final model.

    MDI: how much each feature helped the trees split the data during training.
    Permutation: how much test macro-F1 drops when that feature's values are
    shuffled, so the model can no longer use it.
    """
    _, X_test, _, y_test = load_data(task)
    model = joblib.load(MODEL_DIR / f"{task}.joblib")
    perm = permutation_importance(model, X_test, y_test, scoring="f1_macro",
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    table = pd.DataFrame({
        "feature": X_test.columns,
        "group": [group_of(c) for c in X_test.columns],
        "mdi": model.feature_importances_,
        "permutation": perm.importances_mean,
        "permutation_std": perm.importances_std,
    }).sort_values("mdi", ascending=False).reset_index(drop=True)
    table.insert(0, "mdi_rank", table.index + 1)
    table.to_csv(OUT_DIR / f"feature_importance_{task}.csv", index=False)
    return table


def ablation(task, ranked_features, params):
    """Retrain with groups of features removed and compare CV macro-F1."""
    X, _, y, _ = load_data(task)
    names = class_names(task)
    cols = list(X.columns)
    variants = {
        f"All {len(cols)} features": cols,
        "Without top 10 (MDI)": [c for c in cols if c not in ranked_features[:10]],
        "Without top 20 (MDI)": [c for c in cols if c not in ranked_features[:20]],
        "Without top 50 (MDI)": [c for c in cols if c not in ranked_features[:50]],
        "Header + sections only": [c for c in cols if c.startswith(("hdr_", "sec_"))],
        "API calls + DLLs only": [c for c in cols if c.startswith(("api_", "dll_"))],
    }
    rows = []
    for label, keep in variants.items():
        forest = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
        s = cross_validate_model(forest, X[keep], y, names)
        row = {"task": task, "variant": label, "n_features": len(keep),
               "cv_macro_f1": s["macro_f1"], "cv_macro_f1_std": s["macro_f1_std"],
               "cv_accuracy": s["accuracy"]}
        if task == "binary":
            row["benign_recall"] = s["per_class"]["Benign"]["recall"]
            row["malware_recall"] = s["per_class"]["Malware"]["recall"]
        rows.append(row)
        print(f"{label:28}{len(keep):>5} features  macro-F1 {s['macro_f1']:.4f} "
              f"(+/- {s['macro_f1_std']:.4f})")
    return rows


def main():
    tuned = json.loads((OUT_DIR / "tuned_params.json").read_text())
    ablation_rows, group_rows = [], []

    for task in TASKS:
        print(f"\n=== {task}: feature importance ===")
        table = importance(task)
        print(table.head(15)[["mdi_rank", "feature", "mdi", "permutation"]].to_string(index=False))

        by_group = table.groupby("group").agg(
            features=("feature", "size"), mdi_share=("mdi", "sum")).reset_index()
        by_group.insert(0, "task", task)
        group_rows.append(by_group)
        print(by_group.to_string(index=False))

        print(f"\n=== {task}: ablation (5-fold CV on the training set) ===")
        ablation_rows += ablation(task, list(table.feature), tuned[task]["best_params"])

    pd.concat(group_rows).to_csv(OUT_DIR / "importance_by_group.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "ablation.csv", index=False)
    print(f"\nsaved feature_importance_*.csv, importance_by_group.csv and ablation.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
