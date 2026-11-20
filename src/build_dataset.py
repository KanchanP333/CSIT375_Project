"""
Part 1 - Data: build the clean, merged malware dataset.

Source: Kaggle "Windows Malwares" (joebeachcapital/windows-malwares, CC BY 4.0),
originally https://figshare.com/articles/dataset/Windows_Malware_Detection_Dataset/21608262

Four feature sets, joined on SHA256:
  PE_Header.csv      52 PE header fields             -> prefix hdr_
  PE_Section.csv     9 fields x 10 PE sections       -> prefix sec_
  DLLs_Imported.csv  0/1 flag per imported DLL       -> prefix dll_
  API_Functions.csv  0/1 flag per imported API call  -> prefix api_  (~21.9k columns)

Labels:
  type       0=Benign 1=RedLineStealer 2=Downloader 3=RAT 4=BankingTrojan 5=SnakeKeyLogger 6=Spyware
  is_malware 1 if type != 0 else 0

Outputs (data/processed/):
  full_clean.parquet      every cleaned feature + labels (before feature selection)
  train.csv / test.csv    stratified 80/20 split, selected features only + labels
  selected_features.txt   names of the selected features (in column order)
  feature_scores.csv      mutual-information score of every candidate feature (train only)
  cleaning_log.json       counts for each cleaning step (for the Dataset section of the report)

Run from the project root:  python src/build_dataset.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "windows-malwares"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.20
MIN_BINARY_FREQ = 0.01   # drop DLL/API flags present in < 1% of samples
CORR_THRESHOLD = 0.95    # drop one of each pair of numeric features with |r| > 0.95
N_SELECT = 300           # total features kept after mutual-information ranking

TYPE_NAMES = {0: "Benign", 1: "RedLineStealer", 2: "Downloader", 3: "RAT",
              4: "BankingTrojan", 5: "SnakeKeyLogger", 6: "Spyware"}

log = {}


def load_small(name, prefix):
    df = pd.read_csv(RAW / name)
    before = len(df)
    # same SHA256 listed twice with the same label -> keep one copy
    df = df.drop_duplicates("SHA256")
    log[f"{prefix}rows_raw"] = before
    log[f"{prefix}duplicate_hashes_removed"] = before - len(df)
    feats = [c for c in df.columns if c not in ("SHA256", "Type")]
    df = df.rename(columns={c: prefix + c for c in feats})
    return df


def load_binary_large(name, prefix):
    """Read a wide 0/1 file in chunks as uint8 and drop rare columns."""
    t = time.time()
    header = pd.read_csv(RAW / name, nrows=0).columns
    dtypes = {c: np.uint8 for c in header if c not in ("SHA256",)}
    chunks = pd.read_csv(RAW / name, dtype=dtypes, chunksize=3000)
    df = pd.concat(chunks, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates("SHA256")
    log[f"{prefix}rows_raw"] = before
    log[f"{prefix}duplicate_hashes_removed"] = before - len(df)
    feats = [c for c in df.columns if c not in ("SHA256", "Type")]
    freq = df[feats].mean()
    keep = freq[freq >= MIN_BINARY_FREQ].index.tolist()
    log[f"{prefix}columns_raw"] = len(feats)
    log[f"{prefix}columns_kept_min_freq"] = len(keep)
    df = df[["SHA256", "Type"] + keep].rename(columns={c: prefix + c for c in keep})
    print(f"  {name}: {df.shape} in {time.time() - t:.0f}s")
    return df


def main():
    print("Loading raw files...")
    hdr = load_small("PE_Header.csv", "hdr_")
    sec = load_small("PE_Section.csv", "sec_")
    dll = load_small("DLLs_Imported.csv", "dll_")
    # small DLL file: apply the same rare-column rule
    dll_feats = [c for c in dll.columns if c.startswith("dll_")]
    dll_keep = [c for c in dll_feats if dll[c].mean() >= MIN_BINARY_FREQ]
    log["dll_columns_raw"], log["dll_columns_kept_min_freq"] = len(dll_feats), len(dll_keep)
    dll = dll[["SHA256", "Type"] + dll_keep]
    api = load_binary_large("API_Functions.csv", "api_")

    print("Merging on SHA256 (inner join: samples present in all four sets)...")
    df = hdr
    for other in (sec, dll, api):
        df = df.merge(other, on="SHA256", suffixes=("", "_dup"))
        assert (df["Type"] == df["Type_dup"]).all(), "label mismatch between files"
        df = df.drop(columns="Type_dup")
    log["rows_after_merge"] = len(df)

    df = df.rename(columns={"Type": "type"}).copy()  # .copy() de-fragments the wide frame
    df["is_malware"] = (df["type"] != 0).astype(np.uint8)
    df["type_name"] = df["type"].map(TYPE_NAMES)

    feats = [c for c in df.columns if c[:4] in ("hdr_", "sec_", "dll_", "api_")]

    # missing values
    log["missing_values"] = int(df[feats].isna().sum().sum())
    df[feats] = df[feats].fillna(0)

    # constant columns
    nunique = df[feats].nunique()
    const = nunique[nunique <= 1].index.tolist()
    log["constant_columns_removed"] = len(const)
    df = df.drop(columns=const)
    feats = [f for f in feats if f not in const]

    # one hash per row of features (fast way to find identical feature vectors)
    row_hash = pd.util.hash_pandas_object(df[feats], index=False)

    # exact duplicate samples (same features + same label) -> would leak across train/test
    before = len(df)
    keep_mask = ~pd.DataFrame({"h": row_hash, "t": df["type"]}).duplicated()
    df, row_hash = df[keep_mask], row_hash[keep_mask]
    log["duplicate_samples_removed"] = before - len(df)

    # identical features but different labels -> ambiguous, remove
    n_labels = df["type"].groupby(row_hash).transform("nunique")
    log["conflicting_label_samples_removed"] = int((n_labels > 1).sum())
    df = df[n_labels <= 1].reset_index(drop=True)

    log["final_rows"] = len(df)
    log["final_cleaned_features"] = len(feats)
    log["class_counts"] = df["type_name"].value_counts().to_dict()
    log["binary_counts"] = {"benign": int((df.is_malware == 0).sum()),
                            "malware": int((df.is_malware == 1).sum())}

    df.to_parquet(OUT / "full_clean.parquet", index=False)
    print(f"  cleaned: {df.shape}")

    # ---------- split BEFORE feature selection (no leakage into test) ----------
    train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE,
                                           stratify=df["type"], random_state=SEED)
    train, test = df.loc[train_idx], df.loc[test_idx]

    # ---------- feature selection on the training set only ----------
    print("Feature selection...")
    numeric = [f for f in feats if f.startswith(("hdr_", "sec_"))]
    binary = [f for f in feats if f.startswith(("dll_", "api_"))]

    # 1) correlation filter (numeric + binary separately to keep it cheap)
    def corr_filter(cols):
        c = train[cols].astype(np.float32).corr().abs()
        upper = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1))
        return [col for col in upper.columns if (upper[col] > CORR_THRESHOLD).any()]

    drop_num = corr_filter(numeric)
    drop_bin = corr_filter(binary)
    log["correlated_numeric_removed"] = len(drop_num)
    log["correlated_binary_removed"] = len(drop_bin)
    numeric = [f for f in numeric if f not in drop_num]
    binary = [f for f in binary if f not in drop_bin]

    # 2) mutual information with the malware type (covers the binary task too, since
    #    benign is one of the classes)
    y = train["type"].values
    mi_num = mutual_info_classif(train[numeric].values, y, discrete_features=False,
                                 random_state=SEED)
    mi_bin = mutual_info_classif(train[binary].values, y, discrete_features=True,
                                 random_state=SEED)
    scores = pd.DataFrame({"feature": numeric + binary,
                           "group": [f[:3] for f in numeric + binary],
                           "mutual_info": np.r_[mi_num, mi_bin]})
    scores = scores.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    scores["selected"] = scores.index < N_SELECT
    scores.to_csv(OUT / "feature_scores.csv", index=False)

    selected = scores.loc[scores.selected, "feature"].tolist()
    log["selected_features"] = len(selected)
    log["selected_by_group"] = scores.loc[scores.selected, "group"].value_counts().to_dict()
    (OUT / "selected_features.txt").write_text("\n".join(selected))

    cols = ["SHA256"] + selected + ["is_malware", "type", "type_name"]
    train[cols].to_csv(OUT / "train.csv", index=False)
    test[cols].to_csv(OUT / "test.csv", index=False)
    log["train_rows"], log["test_rows"] = len(train), len(test)

    (OUT / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=int))
    print(json.dumps(log, indent=2, default=int))


if __name__ == "__main__":
    main()
