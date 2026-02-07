import pandas as pd

meta = pd.read_csv("data/processed/meta_dataset.csv")
labels = pd.read_csv("data/processed/best_models.csv")

final = meta.merge(labels, on="dataset_id")
final.to_csv("data/processed/meta_dataset_labeled.csv", index=False)

print("Meta-dataset with labels saved.")