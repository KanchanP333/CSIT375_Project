"""
Random Forest figures for the report, drawn from the saved result files.

Run from the project root (after src/random_forest/analysis.py):
    python src/random_forest/figures.py

Writes five PNG files to figures/random_forest/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from plot_style import (BLUE, BLUE_RAMP, GROUP_COLORS, INK, INK_MUTED,  # noqa: E402
                        INK_SECONDARY, SURFACE, apply_style, plt, value_grid)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "random_forest"
FIG = ROOT / "figures" / "random_forest"
TITLES = {"binary": "Benign vs malware", "family": "Malware type"}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", FIG / name)


def confusion_matrices(test):
    """Counts in each cell; colour shows the share of that true class (row)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.9]})
    for ax, task in zip(axes, ("binary", "family")):
        m = np.array(test[task]["confusion_matrix"])
        share = m / m.sum(axis=1, keepdims=True)
        labels = test[task]["labels"]
        ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if share[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=30 if task == "family" else 0,
                      ha="right" if task == "family" else "center")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{TITLES[task]} (test set, {m.sum():,} files)")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # white gaps between cells
        ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle("Random Forest confusion matrices: number of files, shaded by share of the true class",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rf1_confusion_matrices.png")


def per_class_f1(test):
    r = test["family"]
    names = r["labels"]
    f1 = [r["per_class"][n]["f1-score"] for n in names]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh([names[i] for i in order], [f1[i] for i in order], height=0.5, color=BLUE)
    for y, i in enumerate(order):
        ax.text(f1[i] + 0.012, y, f"{f1[i]:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("F1 on the test set")
    ax.set_title(f"Random Forest F1 per malware type (macro-F1 {r['macro_f1']:.3f})")
    value_grid(ax, "x")
    save(fig, "rf2_f1_per_malware_type.png")


def tuning_curve(tuned):
    curve = pd.read_csv(RES / "max_features_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, task in zip(axes, ("binary", "family")):
        c = curve[curve.task == task]
        base = tuned[task]["baseline_cv_macro_f1"]
        ax.fill_between(c.max_features, c.cv_macro_f1 - c.cv_macro_f1_std,
                        c.cv_macro_f1 + c.cv_macro_f1_std, color=BLUE, alpha=0.10, linewidth=0)
        ax.plot(c.max_features, c.cv_macro_f1, color=BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.axhline(base, color=INK_MUTED, linewidth=1)
        ax.text(0.5, base, "baseline (default settings)", ha="right", va="bottom",
                fontsize=9, color=INK_SECONDARY)
        last = c.iloc[-1]
        ax.annotate(f"{last.cv_macro_f1:.4f}", (last.max_features, last.cv_macro_f1),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK)
        ax.set_xticks(list(c.max_features), [f"{int(v * 100)}%" for v in c.max_features])
        ax.set_xlabel("Share of features each split may choose from (max_features)")
        ax.set_ylabel("CV macro-F1")
        ax.set_title(TITLES[task])
        pad = 0.004
        ax.set_ylim(min(base, (c.cv_macro_f1 - c.cv_macro_f1_std).min()) - pad,
                    (c.cv_macro_f1 + c.cv_macro_f1_std).max() + pad)
        value_grid(ax, "y")
    fig.suptitle("Random Forest tuning: cross-validated macro-F1 (shaded band = spread across the 5 folds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rf3_tuning_max_features.png")


def feature_importance():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, task in zip(axes, ("binary", "family")):
        full = pd.read_csv(RES / f"feature_importance_{task}.csv")
        n_features = len(full)
        t = full.head(15).iloc[::-1]
        ax.barh(t.feature, t.mdi, height=0.55, color=[GROUP_COLORS[g] for g in t.group])
        ax.set_xlabel("Importance (MDI, share of all splitting work)")
        ax.set_title(f"{TITLES[task]}: 15 most important of {n_features} features")
        ax.tick_params(axis="y", labelsize=8)
        value_grid(ax, "x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    fig.legend(handles, list(GROUP_COLORS), title="Feature group", loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Random Forest feature importance", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rf4_feature_importance.png")


def ablation():
    table = pd.read_csv(RES / "ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    for ax, task in zip(axes, ("binary", "family")):
        t = table[table.task == task].iloc[::-1]
        labels = [f"{v} ({n})" for v, n in zip(t.variant, t.n_features)]
        ax.errorbar(t.cv_macro_f1, labels, xerr=t.cv_macro_f1_std, fmt="o", color=BLUE,
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    ecolor=BLUE, elinewidth=2, capsize=0)
        for x, e, y in zip(t.cv_macro_f1, t.cv_macro_f1_std, labels):
            ax.annotate(f"{x:.3f}", (x + e, y), textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=9, color=INK)
        span = t.cv_macro_f1.max() - t.cv_macro_f1.min()
        ax.set_xlim(t.cv_macro_f1.min() - 0.08 * span - 0.005, t.cv_macro_f1.max() + 0.14 * span + 0.005)
        ax.set_xlabel("CV macro-F1 (line = spread across the 5 folds)")
        ax.set_title(TITLES[task])
        value_grid(ax, "x")
    fig.suptitle("Random Forest ablation: score after removing features (number of features kept in brackets)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rf5_ablation.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    apply_style()
    test = json.loads((RES / "test_results.json").read_text())
    tuned = json.loads((RES / "tuned_params.json").read_text())
    confusion_matrices(test)
    per_class_f1(test)
    tuning_curve(tuned)
    feature_importance()
    ablation()


if __name__ == "__main__":
    main()
