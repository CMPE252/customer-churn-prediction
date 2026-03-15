import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


# ***********************************************************
# Step 6: Model Comparison
# ***********************************************************

# 6 Comparing all models on the same dataset
def validate_models(models, X_train, y_train, X_valid, y_valid):
    results = []
    for name, model in models.items():
        # Fit on training set only
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob_valid = model.predict_proba(X_valid)[:, 1]
            roc_auc = roc_auc_score(y_valid, y_prob_valid)
        else:
            roc_auc = None

        results.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, y_pred_train),
            "Validation Accuracy": accuracy_score(y_valid, y_pred_valid),
            "Precision": precision_score(y_valid, y_pred_valid),
            "Recall": recall_score(y_valid, y_pred_valid),
            "F1": f1_score(y_valid, y_pred_valid),
            "ROC-AUC": roc_auc
        })

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    return results_df


# ***********************************************************
# Pipeline orchestrator
# ***********************************************************

def run_model_comparison(models, train_sets, valid_sets, y_train, y_valid):
    print("\n" + "*" * 60)
    print("STEP 6: MODEL COMPARISON")
    print("*" * 60)

    # Use the corr dataset variant for comparison (best from RF)
    X_train_corr = train_sets["final_corr"]
    X_valid_corr = valid_sets["final_corr"]

    val_results = validate_models(
        models=models,
        X_train=X_train_corr,
        y_train=y_train,
        X_valid=X_valid_corr,
        y_valid=y_valid
    )

    print("\nValidation Results:")
    print(val_results)

    print("\n[run_model_comparison] Step 6 completed.")
    return val_results
