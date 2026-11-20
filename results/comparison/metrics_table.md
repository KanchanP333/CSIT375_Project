### Benign vs malware

| Model | CV macro-F1 | Test accuracy | Test macro-precision | Test macro-recall | Test macro-F1 | ROC-AUC | Benign recall | Benign wrongly flagged (of 375) | Malware missed (of 4,318) |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.9243 (+/- 0.0034) | 0.9785 | 0.9247 | 0.9299 | **0.9273** | 0.9930 | 0.872 | 48 | 53 |
| SVM | 0.8964 (+/- 0.0076) | 0.9732 | 0.9683 | 0.8442 | **0.8950** | 0.9731 | 0.691 | 116 | 10 |
| Gradient Boosting | 0.9348 (+/- 0.0029) | 0.9817 | 0.9299 | 0.9486 | **0.9390** | 0.9917 | 0.909 | 34 | 52 |

### Malware type

| Model | CV macro-F1 | Test accuracy | Test macro-precision | Test macro-recall | Test macro-F1 |
|---|---|---|---|---|---|
| Random Forest | 0.8289 (+/- 0.0066) | 0.8270 | 0.8338 | 0.8326 | **0.8325** |
| SVM | 0.7779 (+/- 0.0036) | 0.7668 | 0.7925 | 0.7755 | **0.7785** |
| Gradient Boosting | 0.8356 (+/- 0.0048) | 0.8319 | 0.8431 | 0.8363 | **0.8381** |
