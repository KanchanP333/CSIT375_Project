"""
Gradient Boosting analysis: which features the model relies on, and how the score changes
when groups of features are removed.

Run from the project root (after src/gradient_boosting/final.py):
    python src/gradient_boosting/analysis.py

Writes, in results/gradient_boosting/:
    feature_importance_<task>.csv   every feature, most important first
    importance_by_group.csv         score drop when a whole feature group is shuffled
    ablation.csv                    CV score with feature groups removed

scikit-learn's HistGradientBoostingClassifier does not provide an impurity-
based importance (no equivalent of the Random Forest's MDI), so permutation
importance is the only measure here. It uses the saved final models and the
test set; it describes the finished model and changes no setting. The ablation retrains with 5-fold cross-validation on the
training set only, with the tuned settings.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per model; see tune.py

import json  # noqa: E402
import sys  # noqa: E402
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import load_data  # noqa: E402
from evaluation import SEED, TASKS, make_cv  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "gradient_boosting"
MODEL_DIR = ROOT / "models" / "gradient_boosting"
GROUPS = {"hdr": "PE header", "sec": "PE sections", "dll": "Imported DLLs", "api": "API calls"}
N_REPEATS = 5


def group_of(feature):
    return GROUPS[feature.split("_")[0]]


def importance(task):
    """Permutation importance: the fall in test macro-F1 when one feature's
    values are shuffled between files, so the model can no longer use it."""
    _, X_test, _, y_test = load_data(task)
    model = joblib.load(MODEL_DIR / f"{task}.joblib")
    perm = permutation_importance(model, X_test, y_test, scoring="f1_macro",
                                  n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1)
    table = pd.DataFrame({
        "feature": X_test.columns,
        "group": [group_of(c) for c in X_test.columns],
        "permutation": perm.importances_mean,
        "permutation_std": perm.importances_std,
    }).sort_values("permutation", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    table.to_csv(OUT_DIR / f"feature_importance_{task}.csv", index=False)
    return table


def group_importance(task):
    """Shuffle all the features of one group together (same row order for every
    column in the group) and measure the fall in test macro-F1."""
    _, X_test, _, y_test = load_data(task)
    model = joblib.load(MODEL_DIR / f"{task}.joblib")
    reference = f1_score(y_test, model.predict(X_test), average="macro")
    rng = np.random.RandomState(SEED)
    rows = []
    for prefix, name in GROUPS.items():
        cols = [c for c in X_test.columns if c.startswith(prefix + "_")]
        drops = []
        for _ in range(N_REPEATS):
            shuffled = X_test.copy()
            shuffled[cols] = X_test[cols].values[rng.permutation(len(X_test))]
            drops.append(reference - f1_score(y_test, model.predict(shuffled), average="macro"))
        rows.append({"task": task, "group": name, "features": len(cols),
                     "test_macro_f1": reference, "drop_when_shuffled": float(np.mean(drops)),
                     "drop_std": float(np.std(drops))})
    return rows


def workers():
    """Models to train at once: one per core, but no more than the free
    memory allows (0.5 GB per worker, keeping 2 GB spare)."""
    free_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    return max(1, min(os.cpu_count(), int((free_gb - 2) / 0.5)))


def fit_fold(X, y, keep, train_idx, test_idx, params):
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **params)
    model.fit(X.iloc[train_idx][keep], y.iloc[train_idx])
    return test_idx, model.predict(X.iloc[test_idx][keep])


def ablation(task, ranked_features, params):
    """Retrain with groups of features removed and compare CV macro-F1.
    The settings are the tuned ones and are not re-tuned for each variant."""
    X, _, y, _ = load_data(task)
    cols = list(X.columns)
    variants = {
        f"All {len(cols)} features": cols,
        "Without top 10 (permutation)": [c for c in cols if c not in ranked_features[:10]],
        "Without top 20 (permutation)": [c for c in cols if c not in ranked_features[:20]],
        "Without top 50 (permutation)": [c for c in cols if c not in ranked_features[:50]],
        "Header + sections only": [c for c in cols if c.startswith(("hdr_", "sec_"))],
        "API calls + DLLs only": [c for c in cols if c.startswith(("api_", "dll_"))],
    }
    folds = list(make_cv().split(X, y))
    jobs = [(label, i) for label in variants for i in range(len(folds))]
    out = Parallel(n_jobs=workers())(
        delayed(fit_fold)(X, y, variants[label], *folds[i], params) for label, i in jobs)

    rows = []
    for label, keep in variants.items():
        parts = [r for (lab, _), r in zip(jobs, out) if lab == label]
        pred = np.empty(len(y), dtype=y.dtype)
        fold_f1 = []
        for test_idx, p in parts:
            pred[test_idx] = p
            fold_f1.append(f1_score(y.iloc[test_idx], p, average="macro"))
        row = {"task": task, "variant": label, "n_features": len(keep),
               "cv_macro_f1": float(np.mean(fold_f1)), "cv_macro_f1_std": float(np.std(fold_f1)),
               "cv_accuracy": accuracy_score(y, pred)}
        if task == "binary":
            row["benign_recall"] = recall_score(y, pred, pos_label=0)
            row["malware_recall"] = recall_score(y, pred, pos_label=1)
        rows.append(row)
        print(f"{label:30}{len(keep):>5} features  macro-F1 {row['cv_macro_f1']:.4f} "
              f"(+/- {row['cv_macro_f1_std']:.4f})")
    return rows


def main():
    tuned = json.loads((OUT_DIR / "tuned_params.json").read_text())
    ablation_rows, group_rows = [], []

    for task in TASKS:
        start = time.time()
        print(f"\n=== {task}: permutation importance on the test set ===")
        table = importance(task)
        print(table.head(15)[["rank", "feature", "permutation", "permutation_std"]].to_string(index=False))
        print(f"features with importance above 0.001: {(table.permutation > 0.001).sum()} of {len(table)}")

        rows = group_importance(task)
        group_rows += rows
        print(pd.DataFrame(rows)[["group", "features", "drop_when_shuffled", "drop_std"]].to_string(index=False))
        print(f"({time.time() - start:.0f}s)")

        start = time.time()
        print(f"\n=== {task}: ablation (5-fold CV on the training set) ===")
        ablation_rows += ablation(task, list(table.feature), tuned[task]["best_params"])
        print(f"({time.time() - start:.0f}s)")

    pd.DataFrame(group_rows).to_csv(OUT_DIR / "importance_by_group.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "ablation.csv", index=False)
    print(f"\nsaved feature_importance_*.csv, importance_by_group.csv and ablation.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
