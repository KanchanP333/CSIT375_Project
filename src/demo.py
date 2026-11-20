"""
Demonstration of the malware detection and classification system.

The system works in two stages, as the project brief asks:
    1. the detection model decides whether a file is benign or malware;
    2. if it is malware, the type model names which of the 6 malware types it is.

The input is the static features of a Windows PE file (header fields, section
fields, imported DLLs and API functions), one row per file. No file is opened
or run.

Run from the project root, after each model's final.py:
    python src/demo.py                         a fixed sample of test files: one benign,
                                               one of each malware type, and two errors
    python src/demo.py --random 10             10 random test files
    python src/demo.py --sha256 3f2a9c         the test file whose SHA256 starts with 3f2a9c
    python src/demo.py --csv my_files.csv      new files: a CSV with the feature columns
    python src/demo.py --model random_forest   use another model (default gradient_boosting)
    python src/demo.py --compare               show all three models' verdicts for each file

Test files come from the held-out test set, which no model saw during training
or tuning; their true labels are shown next to the prediction.
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from data import PROCESSED, TYPE_NAMES, feature_names  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"gradient_boosting": "Gradient Boosting", "random_forest": "Random Forest", "svm": "SVM"}
MACHINE = {332: "32-bit (x86)", 34404: "64-bit (x64)"}


class System:
    """The two models of one algorithm, used one after the other."""

    def __init__(self, key):
        self.name = MODELS[key]
        self.detector = joblib.load(ROOT / "models" / key / "binary.joblib")
        self.typer = joblib.load(ROOT / "models" / key / "family.joblib")
        for model in (self.detector, self.typer):
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1   # a few files: starting parallel workers costs more than it saves

    def classify(self, X):
        """Return one result per row: verdict, malware type and their confidence."""
        start = time.perf_counter()
        is_malware = self.detector.predict(X) == 1
        if hasattr(self.detector, "predict_proba"):
            p_malware = self.detector.predict_proba(X)[:, 1]
            stage1 = [f"{p if m else 1 - p:.1%}" for p, m in zip(p_malware, is_malware)]
        else:   # the SVM gives a distance from the boundary, not a probability
            score = self.detector.decision_function(X)
            stage1 = [f"score {s:+.2f}" for s in score]

        types = np.full(len(X), "", dtype=object)
        stage2 = np.full(len(X), "", dtype=object)
        if is_malware.any():
            Xm = X[is_malware]
            pred = self.typer.predict(Xm)
            types[is_malware] = [TYPE_NAMES[v] for v in pred]
            if hasattr(self.typer, "predict_proba"):
                stage2[is_malware] = [f"{p:.1%}" for p in self.typer.predict_proba(Xm).max(axis=1)]
            else:
                stage2[is_malware] = "n/a"
        ms = 1000 * (time.perf_counter() - start) / len(X)
        final = [t if m else "Benign" for t, m in zip(types, is_malware)]
        return final, is_malware, stage1, types, stage2, ms


def describe(row):
    """A few readable facts about the file, taken from its features."""
    api = sum(int(v) for k, v in row.items() if k.startswith("api_"))
    dlls = [k[4:] for k, v in row.items() if k.startswith("dll_") and v == 1]
    facts = [MACHINE.get(int(row["hdr_Machine"]), f"machine {int(row['hdr_Machine'])}"),
             f"{int(row['hdr_NumberOfSections'])} sections",
             f"image size {int(row['hdr_SizeOfImage']):,} bytes",
             f"{api} of the 222 tracked API functions imported"]
    if "mscoree.dll" in dlls:
        facts.append(".NET program (imports mscoree.dll)")
    return "; ".join(facts)


def default_sample(test):
    """One correctly classified benign file, one of each malware type, and two
    files the default model gets wrong, so the demo shows its limits too."""
    preds = pd.read_csv(ROOT / "results" / "gradient_boosting" / "test_predictions_binary.csv")
    fam = pd.read_csv(ROOT / "results" / "gradient_boosting" / "test_predictions_family.csv")
    rng = np.random.RandomState(7)
    picks = [preds[(preds.true == "Benign") & preds.correct].SHA256.sample(1, random_state=rng).item()]
    for name in list(TYPE_NAMES.values())[1:]:
        ok = fam[(fam.true == name) & fam.correct]
        picks.append(ok.SHA256.sample(1, random_state=rng).item())
    picks.append(preds[(preds.true == "Benign") & ~preds.correct].SHA256.sample(1, random_state=rng).item())
    picks.append(fam[(fam.true == "SnakeKeyLogger") & (fam.predicted == "Spyware")]
                 .SHA256.sample(1, random_state=rng).item())
    position = {sha: i for i, sha in enumerate(test.SHA256)}
    return test.iloc[[position[sha] for sha in picks]]


def load_input(args):
    feats = feature_names()
    if args.csv:
        rows = pd.read_csv(args.csv)
        missing = [c for c in feats if c not in rows.columns]
        if missing:
            sys.exit(f"{args.csv} is missing {len(missing)} feature columns, e.g. {missing[:5]}")
        if "SHA256" not in rows.columns:
            rows = rows.copy()
            rows.insert(0, "SHA256", [f"row {i + 1}" for i in range(len(rows))])
        return rows, feats
    test = pd.read_csv(PROCESSED / "test.csv")
    if args.sha256:
        rows = test[test.SHA256.str.startswith(args.sha256.lower())]
        if rows.empty:
            sys.exit(f"no test file has a SHA256 starting with {args.sha256}")
    elif args.random:
        rows = test.sample(args.random, random_state=args.seed)
    else:
        rows = default_sample(test)
    return rows.reset_index(drop=True), feats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=list(MODELS), default="gradient_boosting")
    parser.add_argument("--compare", action="store_true", help="show all three models")
    parser.add_argument("--random", type=int, help="number of random test files")
    parser.add_argument("--seed", type=int, default=None, help="seed for --random")
    parser.add_argument("--sha256", help="start of a test file's SHA256")
    parser.add_argument("--csv", help="CSV of new files with the feature columns")
    args = parser.parse_args()

    rows, feats = load_input(args)
    X = rows[feats]
    keys = list(MODELS) if args.compare else [args.model]
    systems = [System(k) for k in keys]
    results = {s.name: s.classify(X) for s in systems}
    truth = rows["type_name"].tolist() if "type_name" in rows.columns else None

    print(f"\nMalware detection and classification: {len(rows)} file(s)")
    print("Stage 1 decides benign or malware; stage 2 names the malware type.\n")
    correct = {s.name: 0 for s in systems}
    for i, row in rows.iterrows():
        sha = str(row["SHA256"])
        print(f"[{i + 1}] " + (f"SHA256 {sha[:24]}..." if len(sha) == 64 else sha))
        print(f"    File:       {describe(row[feats])}")
        for s in systems:
            final, mal, c1, types, c2, _ = results[s.name]
            verdict = (f"MALWARE ({c1[i]})  ->  type {types[i]} ({c2[i]})" if mal[i]
                       else f"BENIGN ({c1[i]})")
            mark = ""
            if truth:
                ok = final[i] == truth[i]
                correct[s.name] += ok
                mark = "   correct" if ok else "   WRONG"
            label = f"{s.name}:" if args.compare else "Prediction:"
            print(f"    {label:20}{verdict}{mark}")
        if truth:
            print(f"    {'True label:':20}{truth[i]}")
        print()

    for s in systems:
        ms = results[s.name][5]
        line = f"{s.name}: {ms:.1f} ms per file"
        if truth:
            line = f"{s.name}: {correct[s.name]} of {len(rows)} correct, " + line[len(s.name) + 2:]
        print(line)
    if args.csv is None and not (args.random or args.sha256):
        print("(The sample deliberately includes two files the Gradient Boosting system gets wrong.)")


if __name__ == "__main__":
    main()
