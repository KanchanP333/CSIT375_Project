"""
Part 1 - Data: exploratory data analysis plots for the report.
Run after build_dataset.py:  python src/eda.py
Figures are written to reports/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "windows-malwares"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["Benign", "RedLineStealer", "Downloader", "RAT", "BankingTrojan",
         "SnakeKeyLogger", "Spyware"]
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = dict(zip(ORDER, sns.color_palette("tab10", len(ORDER))))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("saved", name)


df = pd.concat([pd.read_csv(PROC / "train.csv"), pd.read_csv(PROC / "test.csv")])
scores = pd.read_csv(PROC / "feature_scores.csv")
selected = (PROC / "selected_features.txt").read_text().splitlines()
log = json.loads((PROC / "cleaning_log.json").read_text())

# 1. class distribution: raw vs cleaned
names = dict(enumerate(ORDER))
raw_counts = pd.read_csv(RAW / "PE_Header.csv", usecols=["Type"])["Type"].map(names).value_counts()
clean_counts = df["type_name"].value_counts()
comp = pd.DataFrame({"Raw": raw_counts, "Cleaned": clean_counts}).reindex(ORDER)
fig, ax = plt.subplots(figsize=(9, 4.5))
comp.plot.bar(ax=ax, color=["#bbbbbb", "#1f77b4"])
ax.set_title("Samples per class: raw vs after cleaning")
ax.set_ylabel("Samples"); ax.set_xlabel("")
ax.tick_params(axis="x", rotation=30)
for c in ax.containers:
    ax.bar_label(c, fontsize=8)
save(fig, "01_class_distribution.png")

# 2. binary balance
fig, ax = plt.subplots(figsize=(4.5, 4))
b = df["is_malware"].map({0: "Benign", 1: "Malware"}).value_counts()
ax.bar(b.index, b.values, color=["#2ca02c", "#d62728"])
ax.bar_label(ax.containers[0])
ax.set_title("Benign vs malware (cleaned)")
ax.set_ylabel("Samples")
save(fig, "02_binary_distribution.png")

# 3. cleaning funnel
steps = [("Raw (PE header file)", log["hdr_rows_raw"]),
         ("In all 4 feature sets", log["rows_after_merge"]),
         ("Duplicates removed", log["rows_after_merge"] - log["duplicate_samples_removed"]),
         ("Conflicting labels removed", log["final_rows"])]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh([s[0] for s in steps][::-1], [s[1] for s in steps][::-1], color="#1f77b4")
ax.bar_label(ax.containers[0], padding=3)
ax.set_title("Cleaning steps (samples remaining)")
ax.set_xlim(0, log["hdr_rows_raw"] * 1.15)
save(fig, "03_cleaning_funnel.png")

# 4. top features by mutual information
top = scores.head(25).iloc[::-1]
colors = top["group"].map({"api": "#1f77b4", "dll": "#ff7f0e", "hdr": "#2ca02c", "sec": "#9467bd"})
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(top["feature"], top["mutual_info"], color=colors)
ax.set_title("Top 25 features by mutual information with malware type")
ax.set_xlabel("Mutual information")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]]
ax.legend(handles, ["API call", "DLL", "PE header", "PE section"], loc="lower right")
save(fig, "04_top_features_mi.png")

# 5. selected features per group
fig, ax = plt.subplots(figsize=(6, 4))
g = scores.loc[scores.selected, "group"].map(
    {"api": "API calls", "dll": "DLLs", "hdr": "PE header", "sec": "PE section"}).value_counts()
ax.bar(g.index, g.values, color="#1f77b4"); ax.bar_label(ax.containers[0])
ax.set_title(f"{len(selected)} selected features by source")
save(fig, "05_selected_by_group.png")

# 6. API/DLL usage per class (heatmap of % samples using each)
bin_top = [f for f in scores["feature"] if f.startswith(("api_", "dll_"))][:30]
usage = df.groupby("type_name")[bin_top].mean().reindex(ORDER) * 100
fig, ax = plt.subplots(figsize=(12, 5.5))
sns.heatmap(usage, cmap="viridis", ax=ax, cbar_kws={"label": "% of samples importing"},
            xticklabels=[f[4:] for f in bin_top])
ax.set_title("Most informative API calls / DLLs: share of samples in each class that import them")
ax.set_ylabel(""); ax.set_xlabel("")
ax.tick_params(axis="x", rotation=75, labelsize=8)
save(fig, "06_api_usage_heatmap.png")

# 7. numeric PE features per class (log scale boxplots)
num_top = [f for f in scores["feature"] if f.startswith(("hdr_", "sec_"))][:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, f in zip(axes.flat, num_top):
    sns.boxplot(data=df, x="type_name", y=np.log1p(df[f].clip(lower=0)), order=ORDER,
                hue="type_name", palette=PALETTE, legend=False, ax=ax, fliersize=1)
    ax.set_title(f); ax.set_xlabel(""); ax.set_ylabel("log(1 + value)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
fig.suptitle("Top PE header / section features by class")
save(fig, "07_numeric_boxplots.png")

# 8. correlation of selected numeric features
num_sel = [f for f in selected if f.startswith(("hdr_", "sec_"))][:25]
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df[num_sel].corr(), cmap="coolwarm", center=0, ax=ax,
            xticklabels=True, yticklabels=True)
ax.tick_params(labelsize=7)
ax.set_title("Correlation between selected PE header / section features")
save(fig, "08_numeric_correlation.png")

# 9. PCA projection of the selected features
X = StandardScaler().fit_transform(np.log1p(df[selected].clip(lower=0)))
pca = PCA(n_components=2, random_state=42).fit(X)
P = pca.transform(X)
fig, ax = plt.subplots(figsize=(9, 7))
for name in ORDER:
    m = (df["type_name"] == name).values
    ax.scatter(P[m, 0], P[m, 1], s=4, alpha=0.5, label=name, color=PALETTE[name])
ev = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
ax.set_title("PCA of the 300 selected features")
ax.legend(markerscale=4)
save(fig, "09_pca.png")
