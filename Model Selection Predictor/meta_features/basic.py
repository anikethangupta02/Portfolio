import numpy as np

def compute_basic_meta_features(X, y):
    n_samples, n_features = X.shape
    num_cols = X.select_dtypes(include=["int64", "float64"]).shape[1]
    cat_cols = n_features - num_cols

    missing_ratio = X.isnull().mean().mean()
    imbalance = y.value_counts(normalize=True).max()

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "pct_numeric": num_cols / n_features,
        "pct_categorical": cat_cols / n_features,
        "missing_ratio": missing_ratio,
        "class_imbalance": imbalance
    }