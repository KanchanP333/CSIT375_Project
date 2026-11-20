# Dataset (Part 1 – Data)

**Source:** Kaggle "Windows Malwares" – `joebeachcapital/windows-malwares` (CC BY 4.0),
originally the *Windows Malware Detection Dataset* on figshare (doi article 21608262).

It has four static feature sets per Windows PE file, linked by SHA256:

| File | Features | Prefix |
|---|---|---|
| PE_Header.csv | 52 PE header fields | `hdr_` |
| PE_Section.csv | 9 fields × 10 sections (.text, .data, .rsrc …) | `sec_` |
| DLLs_Imported.csv | 0/1 per imported DLL (629) | `dll_` |
| API_Functions.csv | 0/1 per imported API call (21,918) | `api_` |

**Labels:** `type` 0=Benign, 1=RedLineStealer, 2=Downloader, 3=RAT, 4=BankingTrojan,
5=SnakeKeyLogger, 6=Spyware. `is_malware` = 1 if type ≠ 0.

## Rebuild

```
pip install pandas numpy scikit-learn pyarrow matplotlib seaborn kaggle
kaggle datasets download -d joebeachcapital/windows-malwares -p data/raw/windows-malwares --unzip
python src/build_dataset.py   # ~10 min, mostly reading the 1.3 GB API file
python src/eda.py             # figures -> reports/figures/
```

## Cleaning steps (numbers from `processed/cleaning_log.json`)

1. Removed duplicate SHA256 rows (3 in DLLs, 5 in API calls, all with the same label).
2. Dropped DLL/API flags used by < 1% of samples (DLLs 629 → 26, APIs 21,918 → 1,179).
3. Inner join on SHA256 across all four files → 29,489 samples. Labels agree across files.
4. No missing values. Dropped 10 constant columns.
5. Removed 5,981 exact duplicate samples (same features and label). Without this step, copies would land in both train and test and inflate scores. Downloader lost the most (4,864 → 1,360) and RAT lost 4,973 → 2,998.
6. Removed 47 samples whose feature vectors were identical but had different labels.
7. **Final: 23,461 samples, 1,337 features.**

## Split and feature selection

- Stratified 80/20 train/test split on `type`, seed 42 (18,768 train / 4,693 test).
- Feature selection was fitted **on the training set only**:
  - correlation filter |r| > 0.95 (removed 25 numeric and 168 binary features)
  - top **300** by mutual information with `type`: 222 API, 38 section, 31 header, 9 DLL.

## Files in `processed/`

| File | Contents |
|---|---|
| `train.csv`, `test.csv` | SHA256 + 300 selected features + `is_malware`, `type`, `type_name` |
| `full_clean.parquet` | all 1,337 cleaned features, before selection |
| `selected_features.txt` | selected feature names |
| `feature_scores.csv` | MI score for every candidate feature |
| `cleaning_log.json` | counts for the report |

## Using it in model scripts

```python
import sys; sys.path.append("src")
from data import load_data
X_tr, X_te, y_tr, y_te = load_data("binary")  # benign vs malware
X_tr, X_te, y_tr, y_te = load_data("type")    # 7 classes
X_tr, X_te, y_tr, y_te = load_data("family")  # 6 malware types only
```

**Notes for the model teammates (Persons 2–5):**
- Features are **not scaled**. For SVM, put `StandardScaler` (ideally after `np.log1p` on the
  `hdr_`/`sec_` columns, which are heavily skewed) inside a `Pipeline` so it is fitted per CV fold.
- Classes are **imbalanced**: benign is only 8% of samples. Use stratified k-fold,
  `class_weight="balanced"`, and report macro F1 alongside accuracy.
- Use `test.csv` only for the final evaluation. Tune with cross-validation on `train.csv`.
