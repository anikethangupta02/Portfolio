import numpy as np
from scipy.stats import skew, kurtosis

def compute_statistical_meta_features(X):
    X_num = X.select_dtypes(include=["int64", "float64"])

    # If no numeric features, return safe defaults
    if X_num.shape[1] == 0:
        return {
            "mean_skewness": 0.0,
            "mean_kurtosis": 0.0,
            "mean_variance": 0.0,
            "max_corr": 0.0
        }

    X_num = X_num.fillna(0)

    return {
        "mean_skewness": np.nanmean(skew(X_num, axis=0)),
        "mean_kurtosis": np.nanmean(kurtosis(X_num, axis=0)),
        "mean_variance": X_num.var().mean(),
        "max_corr": X_num.corr().abs().max().max()
    }