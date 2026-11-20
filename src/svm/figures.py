"""
SVM figures for the report, drawn from the saved result files.

Run from the project root (after src/svm/analysis.py):
    python src/svm/figures.py

Writes five PNG files to figures/svm/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from plot_style import (BLUE, BLUE_RAMP, GROUP_COLORS, INK, SURFACE,  # noqa: E402
                        apply_style, plt, value_grid)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "svm"
FIG = ROOT / "figures" / "svm"
TITLES = {"binary": "Benign vs malware", "family": "Malware type"}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", FIG / name)


def cell_gaps(ax, n_cols, n_rows):
    """Surface-coloured gaps between heatmap cells, no ticks or frame."""
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)


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
        cell_gaps(ax, len(labels), len(labels))
    fig.suptitle("SVM confusion matrices: number of files, shaded by share of the true class",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "svm1_confusion_matrices.png")


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
    ax.set_title(f"SVM F1 per malware type (macro-F1 {r['macro_f1']:.3f})")
    value_grid(ax, "x")
    save(fig, "svm2_f1_per_malware_type.png")


def tuning_grid():
    """CV macro-F1 for every C and gamma in the grid search. The colour scale
    is set per panel, from that task's lowest to highest score."""
    grid = pd.read_csv(RES / "large_c_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for ax, task in zip(axes, ("binary", "family")):
        t = grid[grid.task == task].pivot(index="gamma", columns="C", values="cv_macro_f1")
        t = t.sort_index(ascending=False)
        v = t.values
        ax.imshow(v, cmap=BLUE_RAMP, vmin=v.min(), vmax=v.max(), aspect="auto")
        best = np.unravel_index(v.argmax(), v.shape)
        mid = (v.min() + v.max()) / 2
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ax.text(j, i, f"{v[i, j]:.4f}", ha="center", va="center", fontsize=9,
                        fontweight="bold" if (i, j) == best else "normal",
                        color="#ffffff" if v[i, j] > mid else INK)
        ax.set_xticks(range(v.shape[1]), [f"{int(c):,}" for c in t.columns])
        ax.set_yticks(range(v.shape[0]), [f"{g:g}" for g in t.index])
        ax.set_xlabel("C (regularisation parameter)")
        ax.set_ylabel("gamma (RBF kernel coefficient)")
        ax.set_title(f"{TITLES[task]}: best {v.max():.4f} at C={int(t.columns[best[1]]):,}, "
                     f"gamma={t.index[best[0]]:g}")
        cell_gaps(ax, v.shape[1], v.shape[0])
    fig.suptitle("SVM tuning: cross-validated macro-F1 for each C and gamma in the grid search "
                 "(darker = higher within each panel)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "svm3_tuning_c_gamma.png")


def feature_importance():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, task in zip(axes, ("binary", "family")):
        full = pd.read_csv(RES / f"feature_importance_{task}.csv")
        t = full.head(15).iloc[::-1]
        ax.barh(t.feature, t.permutation, height=0.55, color=[GROUP_COLORS[g] for g in t.group])
        ax.errorbar(t.permutation, t.feature, xerr=t.permutation_std, fmt="none",
                    ecolor=INK, elinewidth=1, capsize=0)
        ax.set_xlabel("Permutation importance (fall in test macro-F1 when the feature is shuffled)")
        ax.set_title(f"{TITLES[task]}: 15 most important of {len(full)} features")
        ax.tick_params(axis="y", labelsize=8)
        value_grid(ax, "x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    fig.legend(handles, list(GROUP_COLORS), title="Feature group", loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("SVM feature importance (thin line = standard deviation over 5 shuffles)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "svm4_feature_importance.png")


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
        ax.set_xlabel("CV macro-F1 (line = standard deviation across the 5 folds)")
        ax.set_title(TITLES[task])
        value_grid(ax, "x")
    fig.suptitle("SVM ablation: score after removing features (number of features kept in brackets)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "svm5_ablation.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    apply_style()
    test = json.loads((RES / "test_results.json").read_text())
    confusion_matrices(test)
    per_class_f1(test)
    tuning_grid()
    feature_importance()
    ablation()


if __name__ == "__main__":
    main()
