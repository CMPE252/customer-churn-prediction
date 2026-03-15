import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ***********************************************************
# Step 8: Threshold Optimization
# ***********************************************************

# 8 Finding the best classification threshold
def optimize_threshold(best_model, X_valid_corr, y_valid):
    # Select the best model
    y_prob_valid = best_model.predict_proba(X_valid_corr)[:, 1]

    # Select range of threshold
    thresholds = np.arange(0.1, 0.6, 0.05)

    # Test all the threshold
    results = []
    for t in thresholds:
        y_pred = (y_prob_valid >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": precision_score(y_valid, y_pred),
            "recall": recall_score(y_valid, y_pred),
            "f1": f1_score(y_valid, y_pred)
        })

    threshold_df = pd.DataFrame(results)
    print("\nThreshold optimization results:")
    print(threshold_df)

    # Select the best threshold
    best_threshold = 0.35

    # Adjust prediction
    y_pred_adjusted = (y_prob_valid >= best_threshold).astype(int)

    print(f"\n[optimize_threshold] Best threshold selected: {best_threshold}")
    return best_threshold


# ***********************************************************
# Step 9: Final Test Evaluation
# ***********************************************************

# 9.1 Refitting on train + validation
def refit_on_full_train(X_train_corr, X_valid_corr, y_train, y_valid):
    X_train_full = pd.concat([X_train_corr, X_valid_corr], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)
    print(f"\n[refit_on_full_train] Full training set shape: {X_train_full.shape}")
    return X_train_full, y_train_full


# 9.2 Evaluating on the test set
def final_test_evaluation(best_model, X_train_full, y_train_full, X_test_corr, y_test, threshold=0.5):
    # Train on train + validation
    best_model.fit(X_train_full, y_train_full)

    # Predict probabilities
    y_prob_test = best_model.predict_proba(X_test_corr)[:, 1]

    # Apply custom threshold
    y_pred_test = (y_prob_test >= threshold).astype(int)

    print(f"\nFinal Test Results (threshold = {threshold})")
    print("Accuracy :", accuracy_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test))
    print("Recall   :", recall_score(y_test, y_pred_test))
    print("F1 Score :", f1_score(y_test, y_pred_test))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob_test))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    print("\n[final_test_evaluation] Final evaluation completed.")
    return y_pred_test, y_prob_test


# ***********************************************************
# Step 10: Model Interpretation
# ***********************************************************

# 10.1 Confusion matrix visualization
def plot_confusion_matrix(y_test, y_pred_test):
    cm = confusion_matrix(y_test, y_pred_test)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    labels = np.array([
        [f"{cm[i,j]}\n({cm_percent[i,j]*100:.2f}%)" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
                xticklabels=["Pred No Churn", "Pred Churn"],
                yticklabels=["Actual No Churn", "Actual Churn"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "final_confusion_matrix.png"), bbox_inches='tight')
    plt.close()

    print("\n[plot_confusion_matrix] Confusion matrix chart saved.")


# 10.2 ROC curve
def plot_roc_curve(y_test, y_prob_test):
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_roc_curve.png"), bbox_inches='tight')
    plt.close()

    print("\n[plot_roc_curve] ROC curve saved.")


# 10.3 Precision-Recall curve
def plot_pr_curve(y_test, y_prob_test):
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob_test)

    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(OUTPUT_DIR, "final_pr_curve.png"), bbox_inches='tight')
    plt.close()

    print("\n[plot_pr_curve] Precision-Recall curve saved.")


# 10.4 Feature importance from the best model
def plot_feature_importance(best_model, X_train_corr):
    feature_importance = pd.DataFrame({
        "Feature": X_train_corr.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nTop 15 Feature Importances:")
    print(feature_importance.head(15))

    top_features = feature_importance.head(15)
    plt.figure(figsize=(8, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.savefig(os.path.join(OUTPUT_DIR, "final_feature_importance.png"), bbox_inches='tight')
    plt.close()

    print("\n[plot_feature_importance] Feature importance chart saved.")


# ***********************************************************
# Pipeline orchestrator
# ***********************************************************

def run_evaluation(best_tuned_models, train_sets, valid_sets, test_sets,
                   y_train, y_valid, y_test):
    print("\n" + "*" * 60)
    print("STEPS 8-10: THRESHOLD OPTIMIZATION, FINAL EVALUATION, INTERPRETATION")
    print("*" * 60)

    X_train_corr = train_sets["final_corr"]
    X_valid_corr = valid_sets["final_corr"]
    X_test_corr = test_sets["final_corr"]

    # Use the best tuned RF model
    best_model = best_tuned_models["Random Forest"].best_estimator_

    # Step 8: Threshold Optimization
    best_threshold = optimize_threshold(best_model, X_valid_corr, y_valid)

    # Step 9: Final Test Evaluation
    X_train_full, y_train_full = refit_on_full_train(X_train_corr, X_valid_corr, y_train, y_valid)
    y_pred_test, y_prob_test = final_test_evaluation(
        best_model, X_train_full, y_train_full, X_test_corr, y_test, best_threshold
    )

    # Step 10: Model Interpretation
    plot_confusion_matrix(y_test, y_pred_test)
    plot_roc_curve(y_test, y_prob_test)
    plot_pr_curve(y_test, y_prob_test)
    plot_feature_importance(best_model, X_train_corr)

    print("\n[run_evaluation] Steps 8-10 completed.")
    return y_pred_test, y_prob_test
