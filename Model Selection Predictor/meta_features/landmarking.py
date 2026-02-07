import time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

def compute_landmark_features(X, y, preprocessor):
    features = {}

    models = {
        "lr": LogisticRegression(max_iter=1000),
        "nb": GaussianNB(),
        "dt": DecisionTreeClassifier(max_depth=3),
        "knn": KNeighborsClassifier(n_neighbors=5)
    }

    for name, model in models.items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        start = time.time()
        score = cross_val_score(
            pipeline, X, y, cv=3, scoring="roc_auc"
        ).mean()
        features[f"{name}_auc"] = score
        if name == "lr":
            features["lr_train_time"] = time.time() - start

    return features