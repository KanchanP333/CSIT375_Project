"""
Comparison of the three models: Random Forest, SVM and Gradient Boosting.

Reads the saved results and models of each model (it trains nothing) and
produces the tables and figures for the Results section.

Run from the project root, after each model's final.py:
    python src/compare.py

Writes, in results/comparison/:
    metrics_table.csv / .md     headline metrics per task and model
    per_class_f1.csv            test F1 per class, task and model
    significance.csv            McNemar's exact test for each pair of models
    two_stage.json              the full system: detection, then type for files flagged as malware
    speed.csv                   prediction time and model file size
and four figures in figures/comparison/.

All test-set numbers here come from the single final test of each model; the
test set is only re-scored with the saved models, never used to choose anything.
"""
import os

# time every model on one CPU thread, so the speed figures are comparable
# (must be set before scikit-learn loads)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, roc_curve)

sys.path.append(str(Path(__file__).resolve().parent))
from data import TYPE_NAMES, load_data  # noqa: E402
from evaluation import TASKS  # noqa: E402
from plot_style import (BLUE_RAMP, INK, INK_MUTED, INK_SECONDARY, MODEL_COLORS,  # noqa: E402
                        SURFACE, apply_style, plt, value_grid)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "comparison"
FIG_DIR = ROOT / "figures" / "comparison"
MODELS = {"random_forest": "Random Forest", "svm": "SVM", "gradient_boosting": "Gradient Boosting"}
TITLES = {"binary": "Benign vs malware", "family": "Malware type"}


def load_json(model, name):
    return json.loads((ROOT / "results" / model / name).read_text())


def malware_score(model, X):
    """A score that is higher for files the model considers more likely malware:
    the predicted probability of malware for the tree models, and the signed
    distance from the decision boundary for the SVM."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# ---------------------------------------------------------------- tables

def metrics_table():
    rows, class_rows = [], []
    for key, name in MODELS.items():
        test = load_json(key, "test_results.json")
        tuned = load_json(key, "tuned_params.json")
        for task in TASKS:
            r = test[task]
            row = {"task": task, "model": name,
                   "cv_macro_f1": tuned[task]["tuned_cv_macro_f1"],
                   "cv_macro_f1_std": tuned[task]["tuned_cv_macro_f1_std"],
                   "test_accuracy": r["accuracy"],
                   "test_macro_precision": r["per_class"]["macro avg"]["precision"],
                   "test_macro_recall": r["per_class"]["macro avg"]["recall"],
                   "test_macro_f1": r["macro_f1"]}
            if task == "binary":
                m = np.array(r["confusion_matrix"])
                row.update({"test_roc_auc": r["roc_auc"],
                            "benign_recall": r["per_class"]["Benign"]["recall"],
                            "benign_wrongly_flagged": int(m[0, 1]),
                            "malware_missed": int(m[1, 0])})
            rows.append(row)
            for cls in r["labels"]:
                c = r["per_class"][cls]
                class_rows.append({"task": task, "model": name, "class": cls,
                                   "precision": c["precision"], "recall": c["recall"],
                                   "f1": c["f1-score"], "files": int(c["support"])})
    return pd.DataFrame(rows), pd.DataFrame(class_rows)


def significance():
    """McNemar's exact test on the test set for each pair of models.

    Only the files on which the two models disagree count: b files that model A
    got right and model B got wrong, c files the other way round. If the two
    models were equally accurate, b and c would differ only by chance; the
    p-value is the probability of a split at least this uneven under that
    assumption (exact two-sided binomial test)."""
    rows = []
    keys = list(MODELS)
    for task in TASKS:
        correct = {k: pd.read_csv(ROOT / "results" / k / f"test_predictions_{task}.csv")["correct"].values
                   for k in keys}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                only_a = int((correct[a] & ~correct[b]).sum())
                only_b = int((~correct[a] & correct[b]).sum())
                p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
                rows.append({"task": task, "model_a": MODELS[a], "model_b": MODELS[b],
                             "only_a_correct": only_a, "only_b_correct": only_b, "p_value": p})
    return pd.DataFrame(rows)


def two_stage():
    """The system as it would be used: the detection model decides benign or
    malware for every test file, and the type model then names the type of
    every file flagged as malware. Scored against the 7 true labels
    (Benign plus the 6 types)."""
    _, X_test, _, y_type = load_data("type")
    labels = sorted(TYPE_NAMES)
    names = [TYPE_NAMES[i] for i in labels]
    out = {}
    for key, name in MODELS.items():
        detector = joblib.load(ROOT / "models" / key / "binary.joblib")
        typer = joblib.load(ROOT / "models" / key / "family.joblib")
        pred = np.zeros(len(X_test), dtype=int)
        flagged = detector.predict(X_test) == 1
        pred[flagged] = typer.predict(X_test[flagged])
        report = classification_report(y_type, pred, labels=labels, target_names=names,
                                       output_dict=True, zero_division=0)
        out[name] = {"test_files": len(X_test),
                     "accuracy": accuracy_score(y_type, pred),
                     "macro_f1": f1_score(y_type, pred, average="macro"),
                     "per_class": report,
                     "confusion_matrix": confusion_matrix(y_type, pred, labels=labels).tolist(),
                     "labels": names}
    return out


def speed():
    """Prediction time on this machine, for the detection and type models."""
    rows = []
    for task in TASKS:
        _, X_test, _, _ = load_data(task)
        for key, name in MODELS.items():
            path = ROOT / "models" / key / f"{task}.joblib"
            model = joblib.load(path)
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1          # time one file the way a single request would run
            model.predict(X_test.iloc[:10])   # warm-up
            start = time.perf_counter()
            model.predict(X_test)
            batch = time.perf_counter() - start
            singles = []
            for i in range(30):
                row = X_test.iloc[[i]]
                start = time.perf_counter()
                model.predict(row)
                singles.append(time.perf_counter() - start)
            rows.append({"task": task, "model": name, "test_files": len(X_test),
                         "batch_seconds": batch,
                         "ms_per_file_in_batch": 1000 * batch / len(X_test),
                         "ms_single_file_median": 1000 * float(np.median(singles)),
                         "model_file_mb": path.stat().st_size / 1e6})
    return pd.DataFrame(rows)


def markdown_table(table):
    """The headline table in Markdown, for pasting into the report."""
    lines = []
    for task in TASKS:
        t = table[table.task == task]
        head = ["Model", "CV macro-F1", "Test accuracy", "Test macro-precision",
                "Test macro-recall", "Test macro-F1"]
        if task == "binary":
            head += ["ROC-AUC", "Benign recall", "Benign wrongly flagged (of 375)",
                     "Malware missed (of 4,318)"]
        lines += [f"### {TITLES[task]}", "", "| " + " | ".join(head) + " |",
                  "|" + "---|" * len(head)]
        for _, r in t.iterrows():
            cells = [r.model, f"{r.cv_macro_f1:.4f} (+/- {r.cv_macro_f1_std:.4f})",
                     f"{r.test_accuracy:.4f}", f"{r.test_macro_precision:.4f}",
                     f"{r.test_macro_recall:.4f}", f"**{r.test_macro_f1:.4f}**"]
            if task == "binary":
                cells += [f"{r.test_roc_auc:.4f}", f"{r.benign_recall:.3f}",
                          f"{int(r.benign_wrongly_flagged)}", f"{int(r.malware_missed)}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- figures

def save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print("saved", FIG_DIR / name)


def fig_macro_f1(table):
    """Test macro-F1 (filled dot) and the cross-validation estimate with its
    standard deviation across folds (hollow dot and line), per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 2.9))
    for ax, task in zip(axes, TASKS):
        t = table[table.task == task].reset_index(drop=True)
        for y, r in t.iterrows():
            color = MODEL_COLORS[r.model]
            ax.errorbar(r.cv_macro_f1, y + 0.15, xerr=r.cv_macro_f1_std, fmt="o", color=color,
                        markerfacecolor=SURFACE, markersize=7, markeredgewidth=2,
                        elinewidth=2, capsize=0)
            ax.plot(r.test_macro_f1, y - 0.15, "o", color=color, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{r.test_macro_f1:.4f}", (r.test_macro_f1, y - 0.15),
                        textcoords="offset points", xytext=(8, 0), va="center",
                        fontsize=9, color=INK)
        ax.set_yticks(range(len(t)), t.model)
        ax.set_ylim(len(t) - 0.5, -0.5)
        lo = min(t.test_macro_f1.min(), (t.cv_macro_f1 - t.cv_macro_f1_std).min())
        hi = max(t.test_macro_f1.max(), (t.cv_macro_f1 + t.cv_macro_f1_std).max())
        ax.set_xlim(lo - 0.01, hi + 0.012)
        ax.set_xlabel("Macro-F1")
        ax.set_title(TITLES[task])
        value_grid(ax, "x")
    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=INK_SECONDARY,
                          markersize=8, label="Test set"),
               plt.Line2D([], [], marker="o", linestyle="-", color=INK_SECONDARY,
                          markerfacecolor=SURFACE, markersize=7, markeredgewidth=2,
                          label="Cross-validation estimate (line = standard deviation across folds)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Macro-F1 of the three models", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "cmp1_macro_f1.png")


def fig_per_class(classes):
    """Test F1 per malware type, one dot per model."""
    t = classes[classes.task == "family"]
    order = (t.groupby("class").f1.mean().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    offsets = {"Random Forest": 0.22, "SVM": 0.0, "Gradient Boosting": -0.22}
    for name, off in offsets.items():
        m = t[t.model == name].set_index("class").loc[order]
        ax.plot(m.f1, np.arange(len(order)) + off, "o", color=MODEL_COLORS[name],
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("F1 on the test set")
    ax.set_title("F1 per malware type: the same three types are hardest for every model")
    value_grid(ax, "x")
    ax.legend(loc="lower right")
    save(fig, "cmp2_f1_per_malware_type.png")


def fig_roc():
    """ROC curves for detection, full range and a close-up of the top-left corner."""
    _, X_test, _, y_test = load_data("binary")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    test = {k: load_json(k, "test_results.json") for k in MODELS}
    for key, name in MODELS.items():
        model = joblib.load(ROOT / "models" / key / "binary.joblib")
        fpr, tpr, _ = roc_curve(y_test, malware_score(model, X_test))
        auc = test[key]["binary"]["roc_auc"]
        for ax in axes:
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2,
                    label=f"{name} (AUC {auc:.4f})")
        # the model's actual operating point: its default decision threshold
        m = np.array(test[key]["binary"]["confusion_matrix"])
        point = (m[0, 1] / m[0].sum(), m[1, 1] / m[1].sum())
        for ax in axes:
            ax.plot(*point, "o", color=MODEL_COLORS[name], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    axes[0].plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle="--")
    axes[0].text(0.55, 0.47, "random guessing (AUC 0.5)", color=INK_SECONDARY, fontsize=9,
                 rotation=33)
    axes[0].set_title("Full curve")
    axes[1].set_xlim(0, 0.4)
    axes[1].set_ylim(0.9, 1.0)
    axes[1].set_title("Close-up of the top-left corner")
    for ax in axes:
        ax.set_xlabel("False positive rate (share of benign files flagged as malware)")
        ax.set_ylabel("True positive rate\n(share of malware detected)")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].legend(loc="lower right")
    fig.suptitle("ROC curves for benign vs malware (test set; dot = the model's default threshold)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "cmp3_roc_curves.png")


def fig_two_stage(result, name):
    r = result[name]
    m = np.array(r["confusion_matrix"])
    share = m / m.sum(axis=1, keepdims=True)
    labels = r["labels"]
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if share[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Final prediction of the system")
    ax.set_ylabel("True class")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"Full system ({name}): detection, then malware type\n"
                 f"{r['test_files']:,} test files, accuracy {r['accuracy']:.4f}, "
                 f"macro-F1 {r['macro_f1']:.4f}", loc="left")
    fig.tight_layout()
    save(fig, "cmp4_two_stage_confusion.png")


# ---------------------------------------------------------------- main

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    table, classes = metrics_table()
    table.to_csv(OUT_DIR / "metrics_table.csv", index=False)
    classes.to_csv(OUT_DIR / "per_class_f1.csv", index=False)
    md = markdown_table(table)
    (OUT_DIR / "metrics_table.md").write_text(md)
    print(md)

    sig = significance()
    sig.to_csv(OUT_DIR / "significance.csv", index=False)
    print("McNemar's exact test on the test set (p < 0.05: the difference is unlikely to be chance)")
    print(sig.to_string(index=False))

    stages = two_stage()
    (OUT_DIR / "two_stage.json").write_text(json.dumps(stages, indent=2))
    print("\nFull system (detection, then type), 7 true classes:")
    for name, r in stages.items():
        print(f"{name:18} accuracy {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}")

    sp = speed()
    sp.to_csv(OUT_DIR / "speed.csv", index=False)
    print("\nPrediction speed (one CPU thread):")
    print(sp.to_string(index=False))

    fig_macro_f1(table)
    fig_per_class(classes)
    fig_roc()
    best = table[table.task == "binary"].sort_values("test_macro_f1").model.iloc[-1]
    fig_two_stage(stages, best)
    print(f"\nsaved tables in {OUT_DIR}")


if __name__ == "__main__":
    main()
