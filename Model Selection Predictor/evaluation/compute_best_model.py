import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from utils.openml_loader import load_openml_dataset, DATASET_IDS
from utils.preprocessing import build_preprocessor
from models.candidate_models import get_candidate_models

rows = []

for did in DATASET_IDS:
    print(f"Evaluating candidate models for dataset {did}")
    X, y = load_openml_dataset(did)
    preprocessor = build_preprocessor(X)

    model_scores = {}

    for name, model in get_candidate_models().items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        score = cross_val_score(
            pipeline,
            X,
            y,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1
        ).mean()

        model_scores[name] = score

    best_model = max(model_scores, key=model_scores.get)

    rows.append({
        "dataset_id": did,
        "best_model": best_model
    })

best_df = pd.DataFrame(rows)
best_df.to_csv("data/processed/best_models.csv", index=False)