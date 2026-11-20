"""
Shared loader for the cleaned dataset (built by src/build_dataset.py).

Usage (from the model scripts):
    from data import load_data
    X_train, X_test, y_train, y_test = load_data(task="binary")   # benign vs malware
    X_train, X_test, y_train, y_test = load_data(task="type")     # 7 classes incl. benign
    X_train, X_test, y_train, y_test = load_data(task="family")   # 6 malware types only

Features are NOT scaled; scale inside your pipeline if your model needs it (e.g. SVM),
so the scaler is fitted on training folds only.
"""
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
# works both inside the project (src/ -> data/processed) and from the handover zip
# (data.py sitting next to train.csv / test.csv)
PROCESSED = _HERE if (_HERE / "train.csv").exists() else _HERE.parent / "data" / "processed"
LABEL_COLS = ["SHA256", "is_malware", "type", "type_name"]
# Columns that stay in the CSV files but are never given to a model.
# hdr_TimeDateStamp is the compile date. It says when a file was built, not what
# it does: in this dataset most malware was compiled in 2020-2022 and the benign
# files earlier, so a model learns the date instead of the file. Results with it
# included are kept in results/with_timestamp/ for comparison.
EXCLUDED_FEATURES = ["hdr_TimeDateStamp"]

TYPE_NAMES = {0: "Benign", 1: "RedLineStealer", 2: "Downloader", 3: "RAT",
              4: "BankingTrojan", 5: "SnakeKeyLogger", 6: "Spyware"}


def load_data(task="binary"):
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")

    if task == "family":  # malware samples only, predict which type
        train, test = train[train.is_malware == 1], test[test.is_malware == 1]

    target = {"binary": "is_malware", "type": "type", "family": "type"}[task]
    feats = [c for c in train.columns if c not in LABEL_COLS + EXCLUDED_FEATURES]
    return train[feats], test[feats], train[target], test[target]


def feature_names():
    names = (PROCESSED / "selected_features.txt").read_text().splitlines()
    return [c for c in names if c not in EXCLUDED_FEATURES]
