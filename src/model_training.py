import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score


# ***********************************************************
# Step 5: Model Training
# ***********************************************************

# Evaluation helper used across all models
def evaluate_one_model_across_datasets(model, train_sets, valid_sets, y_train, y_valid):
    results = []
    for dataset_name, X_train_current in train_sets.items():
        X_valid_current = valid_sets[dataset_name]
        model.fit(X_train_current, y_train)
        prob = model.predict_proba(X_valid_current)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results.append({
            "dataset": dataset_name,
            "roc_auc": roc_auc_score(y_valid, prob),
            "f1": f1_score(y_valid, pred),
            "accuracy": accuracy_score(y_valid, pred),
            "precision": precision_score(y_valid, pred),
            "recall": recall_score(y_valid, pred)
        })
    return pd.DataFrame(results).sort_values(["roc_auc", "f1"], ascending=False)


# 5.1 Logistic Regression (Baseline Model)
def train_logistic_regression(train_sets, valid_sets, y_train, y_valid):
    logistic_model = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=2000)
    results = evaluate_one_model_across_datasets(logistic_model, train_sets, valid_sets, y_train, y_valid)
    print("\nLogistic Regression Results")
    print(results)
    print("\n[train_logistic_regression] Logistic Regression training completed.")
    return logistic_model, results

