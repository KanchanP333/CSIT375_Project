"""
The SVM pipeline, shared by the SVM scripts.

An SVM compares files by distance, so features on huge scales would drown out
the rest. The pipeline therefore prepares the data before the SVM sees it:

    1. log(1 + x) on the header and section columns (hdr_, sec_), which range
       from 0 to billions. The DLL and API columns are already 0/1 and are
       left alone at this step.
    2. StandardScaler on every column (mean 0, spread 1).
    3. The SVM itself (RBF kernel).

Because these steps are inside one Pipeline, cross-validation fits the scaler
on each fold's training part only, so nothing leaks from the quiz part.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

SEED = 42


def make_svm(columns, **svc_params):
    skewed = [c for c in columns if c.startswith(("hdr_", "sec_"))]
    log_step = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), skewed)],
        remainder="passthrough", verbose_feature_names_out=False)
    params = {"kernel": "rbf", "class_weight": "balanced", "random_state": SEED}
    params.update(svc_params)
    return Pipeline([("log", log_step), ("scale", StandardScaler()), ("svm", SVC(**params))])
