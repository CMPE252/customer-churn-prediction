import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
)
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ***********************************************************
# Step 4: Model Development
# ***********************************************************

# 4.1 Feature Selection - checking constant features
def check_constant_features(X_train):
    # Find columns with only one unique value
    cols_have_one_value = X_train.columns[X_train.nunique() == 1].tolist()
    print("\nColumns with constant value:")
    print(cols_have_one_value)
    print("\n[check_constant_features] Constant feature check done.")
    return cols_have_one_value


# 4.1 Feature Selection - checking low-variance binary features
def check_low_variance(X_train):
    binary_mask = X_train.apply(lambda col: set(col.dropna().unique()).issubset({0, 1}))
    binary_cols_found = X_train.columns[binary_mask].tolist()
    mean_of_binary_cols = X_train[binary_cols_found].mean()

    threshold = 0.005
    low_variance_binary_cols = mean_of_binary_cols[mean_of_binary_cols < threshold].index.tolist()

    print("\nThe low-variance binary columns are:", low_variance_binary_cols)
    print("\n[check_low_variance] Low-variance check done.")
    return low_variance_binary_cols


# 4.1 Feature Selection - finding highly correlated features
def find_high_corr_features(df, threshold=0.9):
    """
    Identify highly correlated features in a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        Feature dataset
    threshold : float
        Correlation threshold (default = 0.9)

    Returns
    -------
    high_corr_cols : list
        Columns recommended for removal
    high_corr_pairs : DataFrame
        Highly correlated feature pairs
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_cols = [col for col in upper.columns if any(upper[col] > threshold)]

    high_corr_pairs = upper.stack().reset_index()
    high_corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
    high_corr_pairs = high_corr_pairs[high_corr_pairs["correlation"] > threshold]

    print("Highly correlated pairs:")
    print(high_corr_pairs)
    print("\nHighly correlated columns:")
    print(high_corr_cols)

    return high_corr_cols


# 4.1 Feature Selection - L1 logistic regression
def l1_feature_selection(X_train, y_train, X_train_before_fe, X_train_fe):
    """
    Perform feature selection using L1-regularized Logistic Regression.
    """
    def l1_selected_columns(X, y, C_values):
        """
        L1 regularization (Lasso) shrinks some model coefficients to exactly zero.
        Features with non-zero coefficients are considered informative.

        Parameters
        ----------
        X : pandas DataFrame
        y : pandas Series
        C_values : list of float

        Returns
        -------
        selected_columns : dict
        """
        selected_columns = {}
        for C in C_values:
            model = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000)
            model.fit(X, y)
            coef = model.coef_.ravel()
            selected_columns[C] = X.columns[coef != 0].tolist()
        return selected_columns

    C_values = [0.1, 0.3, 0.5, 1.0]

    train_datasets = {
        "final": X_train,
        "before_fe": X_train_before_fe,
        "fe": X_train_fe
    }

    results = {}
    for name, X in train_datasets.items():
        results[name] = l1_selected_columns(X, y_train, C_values)

    # Summary table
    summary = []
    for dataset_name, dataset_results in results.items():
        for c, cols in dataset_results.items():
            summary.append({"dataset": dataset_name, "C": c, "n_features": len(cols)})
    summary_df = pd.DataFrame(summary)
    print("\nL1 Feature Selection Summary:")
    print(summary_df.sort_values(["dataset", "C"]))

    # Display selected features
    for dataset_name, dataset_results in results.items():
        print(f"\n===== DATASET: {dataset_name} =====")
        for c, cols in dataset_results.items():
            print(f"\nC = {c} | {len(cols)} features selected")
            for col in cols:
                print(f"  - {col}")

    print("\n[l1_feature_selection] L1 feature selection completed.")
    return results


# 4.1 Feature Selection - random forest importance
def random_forest_importance(X_train, y_train):
    random_forest = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    random_forest.fit(X_train, y_train)

    importance = pd.DataFrame({
        "importance": random_forest.feature_importances_,
        "the most important features": X_train.columns
    })

    important_graph = importance.head(20).sort_values("importance", ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='importance', y='the most important features', data=important_graph)
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_rf.png"), bbox_inches='tight')
    plt.close()

    print("\n[random_forest_importance] Feature importance chart saved.")
    return importance


# 4.2 Checking schema alignment across train, validation and test
def run_schema_control(X_train, X_valid, X_test):
    print("\ntrain dataset:", X_train.shape, "validation dataset:", X_valid.shape, "test dataset:", X_test.shape)

    print("The order of the columns of X_train and X_valid are same:", X_train.columns.equals(X_valid.columns))
    print("The order of the columns of X_train and X_test are same:", X_train.columns.equals(X_test.columns))

    print("\n[run_schema_control] Schema control completed.")


# 4.3 Metric functions
def m_accuracy(y_actual, y_prediction):
    acc = accuracy_score(y_actual, y_prediction)
    return float(acc)


def m_precision(y_actual, y_prediction, pos_label=1, zero_division=0):
    prec = precision_score(y_actual, y_prediction, pos_label=pos_label, zero_division=zero_division)
    return float(prec)


def m_recall(y_actual, y_prediction, pos_label=1, zero_division=0):
    rec = recall_score(y_actual, y_prediction, pos_label=pos_label, zero_division=zero_division)
    return float(rec)


def m_f1(y_actual, y_prediction, pos_label=1, zero_division=0):
    f1 = f1_score(y_actual, y_prediction, pos_label=pos_label, zero_division=zero_division)
    return float(f1)


def m_roc_auc(y_actual, y_score):
    y_score = np.asarray(y_score)
    rocauc = roc_auc_score(y_actual, y_score)
    return float(rocauc)


def confusion_matrix_plot(model, X, y, filename="confusion_matrix.png"):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    conf_matrix = pd.DataFrame(cm, columns=["Predicted:0", "Predicted:1"],
                               index=["Actual:0", "Actual:1"])
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()


def roc_plot(model, X, y, filename="roc_curve.png"):
    y_pred_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    auc_score = roc_auc_score(y, y_pred_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("ROC Curve", fontsize=14)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()


# ***********************************************************
# Preparing dataset variants
# ***********************************************************

def drop_columns_from_datasets(columns, *datasets):
    return [df.drop(columns=columns, errors="ignore") for df in datasets]


def prepare_dataset_variants(X_train, X_valid, X_test,
                              X_train_before_fe, X_valid_before_fe, X_test_before_fe,
                              X_train_fe, X_valid_fe, X_test_fe):
    # Find high-correlation columns for each variant
    print("\nChecking high correlations for baseline dataset:")
    high_corr_cols_before_fe = find_high_corr_features(X_train_before_fe, threshold=0.9)

    print("\nChecking high correlations for feature-engineered dataset:")
    high_corr_cols_fe_raw = find_high_corr_features(X_train_fe, threshold=0.9)
    # Keep days_of_tenure and user_engagement because they are more informative
    high_corr_cols_fe = ['past_complaint', 'joining_year', 'login_freq_ratio']

    print("\nChecking high correlations for final (scaled) dataset:")
    high_corr_cols_final_raw = find_high_corr_features(X_train, threshold=0.9)
    high_corr_cols_final = ['past_complaint', 'joining_year', 'login_freq_ratio']

    # Remove correlated columns
    X_train_before_fe_corr, X_valid_before_fe_corr, X_test_before_fe_corr = drop_columns_from_datasets(
        high_corr_cols_before_fe, X_train_before_fe, X_valid_before_fe, X_test_before_fe)

    X_train_fe_corr, X_valid_fe_corr, X_test_fe_corr = drop_columns_from_datasets(
        high_corr_cols_fe, X_train_fe, X_valid_fe, X_test_fe)

    X_train_corr, X_valid_corr, X_test_corr = drop_columns_from_datasets(
        high_corr_cols_final, X_train, X_valid, X_test)

    # Training datasets
    train_sets = {
        "baseline": X_train_before_fe,
        "feature_engineered": X_train_fe,
        "final": X_train,
        "baseline_corr": X_train_before_fe_corr,
        "feature_engineered_corr": X_train_fe_corr,
        "final_corr": X_train_corr
    }

    # Validation datasets
    valid_sets = {
        "baseline": X_valid_before_fe,
        "feature_engineered": X_valid_fe,
        "final": X_valid,
        "baseline_corr": X_valid_before_fe_corr,
        "feature_engineered_corr": X_valid_fe_corr,
        "final_corr": X_valid_corr
    }

    # Test datasets
    test_sets = {
        "baseline": X_test_before_fe,
        "feature_engineered": X_test_fe,
        "final": X_test,
        "baseline_corr": X_test_before_fe_corr,
        "feature_engineered_corr": X_test_fe_corr,
        "final_corr": X_test_corr
    }

    print("\n[prepare_dataset_variants] Dataset variants prepared.")
    return train_sets, valid_sets, test_sets


# ***********************************************************
# Pipeline orchestrator
# ***********************************************************

def run_model_development(X_train, X_valid, X_test, y_train,
                           X_train_before_fe, X_valid_before_fe, X_test_before_fe,
                           X_train_fe, X_valid_fe, X_test_fe):
    print("\n" + "*" * 60)
    print("STEP 4: MODEL DEVELOPMENT")
    print("*" * 60)

    check_constant_features(X_train)
    check_low_variance(X_train)

    l1_feature_selection(X_train, y_train, X_train_before_fe, X_train_fe)
    random_forest_importance(X_train, y_train)

    run_schema_control(X_train, X_valid, X_test)

    train_sets, valid_sets, test_sets = prepare_dataset_variants(
        X_train, X_valid, X_test,
        X_train_before_fe, X_valid_before_fe, X_test_before_fe,
        X_train_fe, X_valid_fe, X_test_fe
    )

    print("\n[run_model_development] Step 4 completed.")
    return train_sets, valid_sets, test_sets
