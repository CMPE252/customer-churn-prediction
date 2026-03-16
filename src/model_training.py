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


# 5.2 Naive Bayes
def train_naive_bayes(train_sets, valid_sets, y_train, y_valid):
    nb_model = GaussianNB()
    results = evaluate_one_model_across_datasets(nb_model, train_sets, valid_sets, y_train, y_valid)
    print("\nNaive Bayes Results")
    print(results)
    print("\n[train_naive_bayes] Naive Bayes training completed.")
    return nb_model, results


# 5.3 K nearest neighbors (kNN)
def train_knn(train_sets, valid_sets, y_train, y_valid):
    # n_neighbors = 5 -> common starting point
    # weights = "distance" -> closer neighbors influence more
    knn_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    results = evaluate_one_model_across_datasets(knn_model, train_sets, valid_sets, y_train, y_valid)
    print("\nkNN Results")
    print(results)
    print("\n[train_knn] kNN training completed.")
    return knn_model, results


# 5.4 Random Forest
def train_random_forest(train_sets, valid_sets, y_train, y_valid):
    rf_model = RandomForestClassifier(
        n_estimators=300,       # more trees, more stable predictions
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,     # prevents overfitting
        random_state=42,        # reproducibility
        n_jobs=-1               # use all CPU cores
    )
    results = evaluate_one_model_across_datasets(rf_model, train_sets, valid_sets, y_train, y_valid)
    print("\nRandom Forest Results")
    print(results)
    print("\n[train_random_forest] Random Forest training completed.")
    return rf_model, results


# 5.5 AdaBoost
def train_adaboost(train_sets, valid_sets, y_train, y_valid):
    adaboost_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )
    results = evaluate_one_model_across_datasets(adaboost_model, train_sets, valid_sets, y_train, y_valid)
    print("\nAdaBoost Results")
    print(results)
    print("\n[train_adaboost] AdaBoost training completed.")
    return adaboost_model, results


# 5.6 XGBoost
def train_xgboost(train_sets, valid_sets, y_train, y_valid):
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    results = evaluate_one_model_across_datasets(xgb_model, train_sets, valid_sets, y_train, y_valid)
    print("\nXGBoost Results")
    print(results)
    print("\n[train_xgboost] XGBoost training completed.")
    return xgb_model, results


# 5.7 Neural Networks
def train_neural_network(train_sets, valid_sets, y_train, y_valid):
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),   # 2 hidden layers
        activation="relu",             # ReLU activation
        solver="adam",                 # Adam optimizer
        alpha=0.0001,                  # L2 regularization
        learning_rate_init=0.001,
        max_iter=500,                  # gives the model enough time to converge
        random_state=42
    )
    results = evaluate_one_model_across_datasets(nn_model, train_sets, valid_sets, y_train, y_valid)
    print("\nNeural Network Results")
    print(results)
    print("\n[train_neural_network] Neural Network training completed.")
    return nn_model, results


# ***********************************************************
# Pipeline orchestrator
# ***********************************************************

def run_all_model_training(train_sets, valid_sets, y_train, y_valid):
    print("\n" + "*" * 60)
    print("STEP 5: MODEL TRAINING")
    print("*" * 60)

    logistic_model, lr_results = train_logistic_regression(train_sets, valid_sets, y_train, y_valid)
    nb_model, nb_results = train_naive_bayes(train_sets, valid_sets, y_train, y_valid)
    knn_model, knn_results = train_knn(train_sets, valid_sets, y_train, y_valid)
    rf_model, rf_results = train_random_forest(train_sets, valid_sets, y_train, y_valid)
    adaboost_model, ada_results = train_adaboost(train_sets, valid_sets, y_train, y_valid)
    xgb_model, xgb_results = train_xgboost(train_sets, valid_sets, y_train, y_valid)
    nn_model, nn_results = train_neural_network(train_sets, valid_sets, y_train, y_valid)

    models = {
        "Logistic Regression": logistic_model,
        "Naive Bayes": nb_model,
        "KNN": knn_model,
        "Random Forest": rf_model,
        "AdaBoost": adaboost_model,
        "XGBoost": xgb_model,
        "Neural Network": nn_model
    }

    all_results = {
        "Logistic Regression": lr_results,
        "Naive Bayes": nb_results,
        "KNN": knn_results,
        "Random Forest": rf_results,
        "AdaBoost": ada_results,
        "XGBoost": xgb_results,
        "Neural Network": nn_results
    }

    print("\n[run_all_model_training] Step 5 completed. All models trained.")
    return models, all_results
