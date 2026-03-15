import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# ***********************************************************
# Step 7: Hyperparameter Tuning
# ***********************************************************

# 7.1 Random Forest tuning (Random Search CV)
def tune_random_forest(X_train_corr, y_train):
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    rf_param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    rf_random = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_param_dist,
        n_iter=5,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    rf_random.fit(X_train_corr, y_train)

    print("Best RF Params:", rf_random.best_params_)
    print("Best RF CV ROC-AUC:", rf_random.best_score_)

    print("\n[tune_random_forest] Random Forest tuning completed.")
    return rf_random


# 7.2 XGBoost tuning (Random Search CV)
def tune_xgboost(X_train_corr, y_train):
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss"
    )

    xgb_param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.3],
        "min_child_weight": [1, 3, 5]
    }

    xgb_random = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=xgb_param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    xgb_random.fit(X_train_corr, y_train)

    print("Best XGB Params:", xgb_random.best_params_)
    print("Best XGB CV ROC-AUC:", xgb_random.best_score_)

    print("\n[tune_xgboost] XGBoost tuning completed.")
    return xgb_random


# 7.3 Logistic Regression tuning (Grid Search CV)
def tune_logistic_regression(X_train_corr, y_train):
    logistic_model = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=2000)

    lr_param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"],
        "penalty": ["l2"]
    }

    lr_grid = GridSearchCV(
        estimator=logistic_model,
        param_grid=lr_param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    lr_grid.fit(X_train_corr, y_train)

    print("Best LR Params:", lr_grid.best_params_)
    print("Best LR CV ROC-AUC:", lr_grid.best_score_)

    print("\n[tune_logistic_regression] Logistic Regression tuning completed.")
    return lr_grid


# ***********************************************************
# Pipeline orchestrator
# ***********************************************************

def run_hyperparameter_tuning(train_sets, y_train):
    print("\n" + "*" * 60)
    print("STEP 7: HYPERPARAMETER TUNING")
    print("*" * 60)

    X_train_corr = train_sets["final_corr"]

    rf_random = tune_random_forest(X_train_corr, y_train)
    xgb_random = tune_xgboost(X_train_corr, y_train)
    lr_grid = tune_logistic_regression(X_train_corr, y_train)

    best_models = {
        "Random Forest": rf_random,
        "XGBoost": xgb_random,
        "Logistic Regression": lr_grid
    }

    print("\n[run_hyperparameter_tuning] Step 7 completed.")
    return best_models
