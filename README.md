# AI-Powered Malware Detection and Classification

CSIT375 AI for Cybersecurity, Project 1.

The system classifies Windows executable (PE) files in two stages, using only
static features extracted from each file (header fields, section fields,
imported DLLs and imported API functions):

1. **Detection:** is the file benign or malware?
2. **Classification:** if it is malware, which of six types is it?
   (RedLineStealer, Downloader, RAT, BankingTrojan, SnakeKeyLogger, Spyware)

Three supervised models are built and compared for both stages: Random
Forest, Support Vector Machine (SVM) and Gradient Boosting.

## Results in brief (held-out test set, macro-F1)

| Model | Detection (4,693 files) | Malware type (4,318 files) | Full two-stage system (7 classes) |
|---|---|---|---|
| Gradient Boosting | **0.9390** | **0.8381** | **0.8390** |
| Random Forest | 0.9273 | 0.8325 | 0.8307 |
| SVM | 0.8950 | 0.7785 | 0.7713 |

Full tables: `results/comparison/metrics_table.md`.

## Dataset

Windows Malware Detection Dataset (Yousuf, 2023, CC BY 4.0), obtained through
its Kaggle mirror "Windows Malwares". After cleaning: 23,461 files, split
80/20 into 18,768 training and 4,693 test files, with 300 selected features.
The compile date (`hdr_TimeDateStamp`) is excluded from every model, so the
models use 299 features. `DATA.md` describes the cleaning and feature
selection.

## Setup

Python 3.12 was used. From this folder:

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`scikit-learn` is pinned to 1.7.0 because the saved models in `models/` were
created with it.

## Running the demonstration

The trained models are in `models/`, so the demo runs straight away:

```
python src/demo.py                          # a fixed sample: one benign file, one of each malware type, two errors
python src/demo.py --compare                # the same files, with all three models side by side
python src/demo.py --random 10              # 10 random test files
python src/demo.py --sha256 7ab4a278        # one test file, by the start of its SHA256
python src/demo.py --csv new_files.csv      # new files: a CSV with the 299 feature columns
python src/demo.py --model random_forest    # another model (default: gradient_boosting)
```

Each file's verdict, malware type, confidence and true label are printed.

## Reproducing the results

All commands are run from this folder. Every script uses seed 42.

**Quick check (about 2 minutes):** retrain the final models with the tuned
hyperparameters, score them on the test set, and rebuild the comparison:

```
python src/random_forest/final.py
python src/svm/final.py
python src/gradient_boosting/final.py
python src/compare.py
```

**Full pipeline (several hours on a 24-core machine):** each model has six
steps, run in order: baseline, tuning, tuning follow-up, final test,
analysis, figures.

| Model | Scripts, in order | Approximate time |
|---|---|---|
| Random Forest | `baseline.py`, `tune.py`, `max_features.py`, `final.py`, `analysis.py`, `figures.py` in `src/random_forest/` | 15 to 20 min |
| SVM | `baseline.py`, `tune.py`, `tune_large_c.py`, `final.py`, `analysis.py`, `figures.py` in `src/svm/` | about 2.5 h |
| Gradient Boosting | `baseline.py`, `tune.py`, `tune_follow_up.py`, `final.py`, `analysis.py`, `figures.py` in `src/gradient_boosting/` | about 1.5 h |

Then `python src/compare.py` for the comparison, and optionally
`python src/feature_audit.py` for the header-field audit described in
`RANDOM_FOREST.md`. The heavy scripts choose how many models to train at once
from the machine's free memory.

## Folder layout

```
README.md                 this file
DATA.md                   the dataset: source, cleaning, split, feature selection
RANDOM_FOREST.md          Random Forest: method, every result, file reference
SVM.md                    SVM: the same
GRADIENT_BOOSTING.md      Gradient Boosting: the same
requirements.txt          Python packages
data/processed/           train.csv, test.csv, selected_features.txt
src/
  data.py                 loads the data; excludes hdr_TimeDateStamp
  evaluation.py           shared cross-validation and scoring
  plot_style.py           shared figure styling
  feature_audit.py        header-field audit
  compare.py              comparison of the three models
  demo.py                 the two-stage demonstration
  random_forest/, svm/, gradient_boosting/   the six scripts of each model
results/<model>/          every number each script produces (CSV and JSON)
results/comparison/       comparison tables, significance tests, demo output
results/with_timestamp/   earlier results that included the compile date
models/<model>/           the trained models (binary.joblib, family.joblib)
figures/<model>/          figures for the report
figures/comparison/       comparison figures
report/                   the Word report
```

## Method in brief

- **Evaluation:** the test set (20%) was set aside before modelling and used
  only for each model's final evaluation. All tuning used 5-fold stratified
  cross-validation on the training set.
- **Main metric:** macro-F1, the unweighted mean of the per-class F1 scores.
  Accuracy is misleading here because 92% of files are malware.
- **Class imbalance:** stratified folds, and `class_weight` was offered as a
  hyperparameter during tuning.
- **SVM preprocessing:** log transform of header and section fields, then
  standardisation, inside a scikit-learn pipeline so that it is fitted on
  training folds only.
