import openml
import time
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASET_IDS = [
    31, 37, 44, 50, 1461, 1480, 1590, 4134, 41945
]

CACHE_DIR = "data/raw"

def load_openml_dataset(dataset_id, retries=3):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f"{CACHE_DIR}/dataset_{dataset_id}.csv"

    # Load from cache
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        X = df.drop(columns=["__target__"])
        y = df["__target__"]
    else:
        for attempt in range(1, retries + 1):
            try:
                print(f"Downloading OpenML dataset {dataset_id} (attempt {attempt})")
                dataset = openml.datasets.get_dataset(dataset_id)
                X, y, _, _ = dataset.get_data(
                    target=dataset.default_target_attribute,
                    dataset_format="dataframe"
                )

                df = X.copy()
                df["__target__"] = y
                df.to_csv(cache_path, index=False)
                break
            except Exception as e:
                print(f"Download failed for dataset {dataset_id}: {e}")
                time.sleep(5)
        else:
            raise RuntimeError(f"Failed to download dataset {dataset_id}")

    
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y