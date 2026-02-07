import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from models.meta_model import build_meta_model

# Load labeled meta-dataset
df = pd.read_csv("data/processed/meta_dataset_labeled.csv")

X = df.drop(columns=["dataset_id", "best_model"])
y = df["best_model"]

# Encode model names (rf, xgb, svm...)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train / test split (dataset-level)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=42
)

# Train meta-model
meta_model = build_meta_model()
meta_model.fit(X_train, y_train)

# Evaluate Top-1 accuracy
y_pred = meta_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)



# Predict probabilities
probs = meta_model.predict_proba(X_test)
classes = meta_model.classes_

# Get top-2 predictions
top2 = np.argsort(probs, axis=1)[:, -2:]
top2_labels = [[classes[j] for j in row] for row in top2]

# Compute Top-2 accuracy
top2_acc = np.mean([
    y_test[i] in top2_labels[i]
    for i in range(len(y_test))
])


print(f"Top-1 Meta-Model Accuracy: {acc:.2f}")
print(f"Top-2 Meta-Model Accuracy: {top2_acc:.2f}")
