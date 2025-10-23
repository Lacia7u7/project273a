import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from train.metrics import compute_metrics

def train_and_eval_baselines(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train multiple baseline ML models and evaluate on val and test sets."""
    results = {}
    # Define models with some default hyperparameters
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            # Some models (e.g. SVM without probability) we set probability=True for consistency
            val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
            val_metrics = compute_metrics(y_val, val_prob)
        else:
            val_metrics = {}
        # Evaluate on test set
        test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        test_metrics = compute_metrics(y_test, test_prob)
        results[name] = {"val_metrics": val_metrics, "test_metrics": test_metrics}
    return results
