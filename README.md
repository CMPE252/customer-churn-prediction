# CMPE 252: Customer Churn Prediction

An end-to-end machine learning pipeline for predicting customer churn risk. The project covers data cleaning, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and final evaluation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

## Usage

Run the full pipeline from the project root:

```bash
python main.py
```

All charts are saved to `outputs/` and the full console log is written to `outputs/output_logs.txt`.
The Google Colab notebook is located in notebook folder to review the building process.

## Project Structure

```
customer-churn-prediction
├── main.py                  # Pipeline entry point
├── requirements.txt         # pip dependencies
├── .gitignore
├── data/
│   └── churn.csv            # Customer churn dataset
├── notebook/
│   └── CMPE_252_Customer_Churn_Prediction.ipynb
├── outputs/                 # Generated charts and logs (git-ignored)
│   ├── output_logs.txt
│   └── *.png
└── src/
    ├── data_preprocessing.py   # Steps 1.1–1.9: Cleaning & Step 3: Feature Engineering
    ├── encoding_eda.py         # Step 1.10: Encoding & Steps 2.1–2.3: Basic EDA
    ├── eda.py                  # Steps 2.4–2.5: Advanced EDA & Correlation
    ├── model_development.py    # Step 4: Feature Selection & Schema Control
    ├── model_training.py       # Step 5: Training 7 classifiers
    ├── model_comparison.py     # Step 6: Model Comparison
    ├── hyperparameter.py       # Step 7: Hyperparameter Tuning
    └── model_evaluation.py     # Steps 8–10: Threshold, Evaluation, Interpretation
```

## Pipeline Steps

| Step    | Description                                                                                    |
| ------- | ---------------------------------------------------------------------------------------------- |
| 1.1–1.9 | Data loading, duplicate check, train/valid/test split, missing value handling, outlier capping |
| 1.10    | Datetime feature extraction, label encoding, one-hot encoding                                  |
| 2.1–2.3 | Target distribution, descriptive statistics, numerical & categorical distributions             |
| 2.4–2.5 | Churn vs non-churn comparison, correlation analysis                                            |
| 3       | Tenure, recency, frequency, RFM, offer-discount features, scaling                              |
| 4       | Constant/low-variance removal, L1 selection, RF importance, dataset variants                   |
| 5       | Logistic Regression, Naive Bayes, KNN, Random Forest, AdaBoost, XGBoost, MLP                   |
| 6       | Side-by-side model comparison on validation set                                                |
| 7       | RandomizedSearchCV (RF, XGB) and GridSearchCV (LR)                                             |
| 8–10    | Threshold optimization, refit on train+valid, test evaluation, interpretation plots            |

## Models

- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Random Forest
- AdaBoost
- XGBoost
- Multi-Layer Perceptron (Neural Network)

## Dataset

`data/churn.csv` — 36,992 rows × 24 columns containing customer demographics, behavior metrics, and a binary target column `churn_risk_score` (1 = high-risk, 0 = low-risk).
