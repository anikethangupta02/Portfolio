from sklearn.model_selection import train_test_split
from models.meta_model import build_meta_model
import pandas as pd

df = pd.read_csv("data/processed/meta_dataset.csv")
X = df.drop(columns=["best_model"])
y = df["best_model"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = build_meta_model()
model.fit(X_train, y_train)

print("Meta-model accuracy:", model.score(X_test, y_test))