import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy

def compute_information_meta_features(X, y):
    X_num = X.select_dtypes(include=["int64", "float64"])

    # Target entropy is always computable
    target_entropy = entropy(y.value_counts(normalize=True))

    # If no numeric features, MI is zero
    if X_num.shape[1] == 0:
        return {
            "target_entropy": target_entropy,
            "mean_mutual_info": 0.0
        }

    X_num = X_num.fillna(0)

    mi = mutual_info_classif(
        X_num,
        y,
        discrete_features=False
    )

    return {
        "target_entropy": target_entropy,
        "mean_mutual_info": np.mean(mi)
    }