from utils.file_utils import ensure_dirs

ensure_dirs()

import pandas as pd
from utils.openml_loader import load_openml_dataset, DATASET_IDS
from utils.preprocessing import build_preprocessor
from meta_features.basic import compute_basic_meta_features
from meta_features.statistical import compute_statistical_meta_features
from meta_features.information import compute_information_meta_features
from meta_features.landmarking import compute_landmark_features

meta_rows = []

for did in DATASET_IDS:
    X, y = load_openml_dataset(did)
    preprocessor = build_preprocessor(X)

    features = {}
    features.update(compute_basic_meta_features(X, y))
    features.update(compute_statistical_meta_features(X))
    features.update(compute_information_meta_features(X, y))
    features.update(compute_landmark_features(X, y, preprocessor))

    features["dataset_id"] = did
    meta_rows.append(features)

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv("data/processed/meta_dataset.csv", index=False)