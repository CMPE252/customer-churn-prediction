import os
import sys
import warnings
warnings.filterwarnings("ignore")

from src.data_preprocessing import run_data_cleaning, run_feature_engineering
from src.encoding_eda import run_encoding, run_basic_eda
from src.eda import run_churn_comparison, run_correlation_analysis
from src.model_development import run_model_development
from src.model_training import run_all_model_training
from src.model_comparison import run_model_comparison
from src.hyperparameter import run_hyperparameter_tuning
from src.model_evaluation import run_evaluation


class TeeStream:
    """Write to both console and a log file simultaneously."""
    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)

    def flush(self):
        self.console.flush()
        self.log_file.flush()


def main():
    # Make sure outputs folder exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Set up dual logging: console + outputs/output_logs.txt
    log_path = os.path.join(output_dir, "output_logs.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)

    print("*" * 60)
    print("   CMPE 252: Customer Churn Prediction Pipeline")
    print("*" * 60)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "churn.csv")

    # ******************************************************************
    # Step 1: Data Cleaning & Pre-processing (1.1 - 1.9)
    # ******************************************************************
    (X_train, X_valid, X_test, y_train, y_valid, y_test,
     X_train_eda_raw, X_train_eda_clean, X_train_eda_outliers_removed,
     X_train_eda_imputed, y_train_eda, num_cols, outlier_bounds) = run_data_cleaning(csv_path)

    # ******************************************************************
    # Step 1.10: Categorical Encoding + Step 2.1-2.3: Basic EDA
    # ******************************************************************
    (X_train, X_valid, X_test,
     binary_cols, nominal_cols) = run_encoding(X_train, X_valid, X_test)

    run_basic_eda(X_train_eda_raw, X_train_eda_outliers_removed, y_train_eda, X_train)

    # ******************************************************************
    # Step 2.4 - 2.5: Advanced EDA
    # ******************************************************************
    run_churn_comparison(
        X_train_eda_clean, X_train_eda_outliers_removed,
        X_train_eda_imputed, y_train_eda, num_cols, binary_cols, nominal_cols
    )

    run_correlation_analysis(
        X_train_eda_clean, X_train_eda_outliers_removed,
        X_train_eda_imputed, y_train_eda, num_cols, binary_cols, nominal_cols
    )

    # ******************************************************************
    # Step 3: Feature Engineering
    # ******************************************************************
    (X_train, X_valid, X_test,
     X_train_before_fe, X_valid_before_fe, X_test_before_fe,
     X_train_fe, X_valid_fe, X_test_fe) = run_feature_engineering(X_train, X_valid, X_test)

    # ******************************************************************
    # Step 4: Model Development (Feature Selection + Schema Control)
    # ******************************************************************
    train_sets, valid_sets, test_sets = run_model_development(
        X_train, X_valid, X_test, y_train,
        X_train_before_fe, X_valid_before_fe, X_test_before_fe,
        X_train_fe, X_valid_fe, X_test_fe
    )

    # ******************************************************************
    # Step 5: Model Training
    # ******************************************************************
    models, all_results = run_all_model_training(train_sets, valid_sets, y_train, y_valid)

    # ******************************************************************
    # Step 6: Model Comparison
    # ******************************************************************
    val_results = run_model_comparison(models, train_sets, valid_sets, y_train, y_valid)

    # ******************************************************************
    # Step 7: Hyperparameter Tuning
    # ******************************************************************
    best_tuned_models = run_hyperparameter_tuning(train_sets, y_train)

    # ******************************************************************
    # Steps 8-10: Threshold Optimization, Final Evaluation, Interpretation
    # ******************************************************************
    y_pred_test, y_prob_test = run_evaluation(
        best_tuned_models, train_sets, valid_sets, test_sets,
        y_train, y_valid, y_test
    )

    print("\n" + "*" * 60)
    print("   Pipeline completed. All outputs saved to: outputs/")
    print("*" * 60)

    # Close log file and restore stdout
    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
