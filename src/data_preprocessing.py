import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Utility function used across steps
def check_missing_values(df):
    """
    Check the percentage of missing values in a DataFrame.
    """
    missing_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Count": df.isnull().sum(),
        "Missing Percentage (%)": df.isnull().mean() * 100
    })
    return missing_df


# ***********************************************************
# Step 1: Data Cleaning & Pre-processing
# ***********************************************************

# 1.1 Importing the data
def load_data(path):
    # Import the dataset
    df = pd.read_csv(path, encoding='latin1')
    print(df.head())
    print("\n[load_data] Dataset loaded successfully.")
    return df


# 1.2 Checking data dimensions and types
def check_data_info(df):
    # Check data size
    print("\nData shape:", df.shape)
    # Check columns and data types
    print(df.info())
    print("\n[check_data_info] Data dimensions and types reviewed.")
    return df.shape


# 1.3 Looking at the target variable and class imbalance
def analyze_target(df, target='churn_risk_score'):
    print("\nTarget value counts:")
    print(df[target].value_counts())

    imbalance = df[target].value_counts(normalize=True)
    print(imbalance * 100)

    churn_label = ['High-risk churn customers', 'Low-risk churn customers']
    plt.figure()
    plt.pie(imbalance, labels=churn_label, startangle=90, autopct='%1.1f%%')
    plt.title('Class Imbalance of Target Variable')
    plt.savefig(os.path.join(OUTPUT_DIR, "class_imbalance_target.png"), bbox_inches='tight')
    plt.close()

    print("[analyze_target] Class imbalance chart saved.")
    return imbalance


# 1.4 Checking for duplicates
def check_duplicates(df):
    dup_count = df.duplicated().sum()
    print(f"\n[check_duplicates] Duplicate rows found: {dup_count}")
    return dup_count


# 1.5 Splitting into train, validation and test
def split_data(df, target='churn_risk_score'):
    X_features = df.drop(columns=[target])
    y_churn = df[target]

    # First step: The data is splitted as train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features, y_churn,
        test_size=0.30,
        stratify=y_churn,
        random_state=42
    )

    # Second step: The temp portions are splitted as validation 15% and test 15%
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    churn_label = ['High-risk churn customers', 'Low-risk churn customers']
    imbalance_train = y_train.value_counts(normalize=True)
    imbalance_valid = y_valid.value_counts(normalize=True)
    imbalance_test = y_test.value_counts(normalize=True)
    print(imbalance_train * 100)
    print(imbalance_valid * 100)
    print(imbalance_test * 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    imbalance_train.plot(kind='pie', autopct='%1.1f%%', ax=axes[0],
                         title='Training Set Class Imbalance', ylabel='', labels=None, startangle=90)
    imbalance_valid.plot(kind='pie', autopct='%1.1f%%', ax=axes[1],
                         title='Validation Set Class Imbalance', ylabel='', labels=None, startangle=90)
    imbalance_test.plot(kind='pie', autopct='%1.1f%%', ax=axes[2],
                        title='Test Set Class Imbalance', ylabel='', labels=None, startangle=90)
    fig.legend(churn_label, loc='center right', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "class_imbalance_splits.png"), bbox_inches='tight')
    plt.close()

    # Test before copy
    assert all(X_train.index == y_train.index)

    # Copy data
    X_train_eda_raw = X_train.copy()
    y_train_eda = y_train.copy()

    print("\n[split_data] Train={}, Valid={}, Test={}".format(
        X_train.shape, X_valid.shape, X_test.shape))
    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_eda_raw, y_train_eda


# 1.6 Dropping identification features
def create_referral_features(X_train, X_valid, X_test):
    # Check values of referral_id before dropping
    print("\nFirst letter of referral_id:")
    print(X_train['referral_id'].str[0].unique())
    print("\nreferral_id starting with x:")
    print(X_train['referral_id'][X_train['referral_id'].str[0] == 'x'].unique())

    # Check unique values of joined_through_referral
    print(X_train['joined_through_referral'].unique())

    # Binary flag: whether a user has a valid referral ID
    for df in [X_train, X_valid, X_test]:
        df["has_referral_id"] = (
            (df["referral_id"] != "xxxxxxxx") &
            (df["referral_id"] != "No referral")
        ).astype(int)

    # Valid referral IDs from training data only
    valid_referrals = X_train.loc[
        ~X_train["referral_id"].isin(["No referral", "xxxxxxxx"]),
        "referral_id"
    ]

    # Frequency encoding: how often each referral_id appears
    referral_id_count = valid_referrals.value_counts()

    for df in [X_train, X_valid, X_test]:
        df["referral_id_frequency"] = (
            df["referral_id"]
            .map(referral_id_count)
            .fillna(0)
            .astype(float)
        )

    print("\n[create_referral_features] Referral features created.")
    return X_train, X_valid, X_test


def drop_id_columns(X_train, X_valid, X_test, X_train_eda_raw):
    # Column names to dropping
    columns_to_drop = ["Unnamed: 0", "security_no", "referral_id"]

    def drop_columns(datasets, cols):
        """
        Drop specified columns from multiple datasets.

        Parameters
        ----------
        datasets : list of DataFrames
            List of datasets (e.g., train, validation, test)
        columns_to_drop : list
            Columns to remove
        """
        return [df.drop(columns=cols) for df in datasets]

    X_train, X_valid, X_test, X_train_eda_raw = drop_columns(
        [X_train, X_valid, X_test, X_train_eda_raw], columns_to_drop
    )

    print("\n[drop_id_columns] Dropped: {}".format(columns_to_drop))
    return X_train, X_valid, X_test, X_train_eda_raw


# 1.7 Handling missing values
def handle_invalid_values(X_train, X_valid, X_test, X_train_eda_raw):
    # Checks every categorical columns value counts
    for col in X_train.select_dtypes(include="object").columns:
        print(X_train[col].value_counts())
        print("_" * 40)

    # Before replacing the Error values with NaN, flagged the invalid data in a new column
    X_train["avg_frequency_login_days_was_invalid"] = (X_train["avg_frequency_login_days"] == "Error").astype(int)
    X_valid["avg_frequency_login_days_was_invalid"] = (X_valid["avg_frequency_login_days"] == "Error").astype(int)
    X_test["avg_frequency_login_days_was_invalid"] = (X_test["avg_frequency_login_days"] == "Error").astype(int)

    # Function to convert to a numeric column
    def convert_to_numeric(datasets, column):
        """
        Convert a column to numeric for multiple datasets.
        Non-standard values (e.g., 'Error') will be coerced to NaN.
        """
        for name, df in datasets.items():
            df[column] = pd.to_numeric(df[column], errors="coerce")
            print(f"{name} {column} dtype:", df[column].dtype)

    datasets = {
        "Train": X_train, "Validation": X_valid, "Test": X_test,
        "Train EDA Raw": X_train_eda_raw
    }
    convert_to_numeric(datasets, "avg_frequency_login_days")

    # Create a separate column for each variable to record whether the values were invalid
    nonstandard_values = ['Unknown', '?']
    nonstandard_cols = ['gender', 'joined_through_referral', 'medium_of_operation']

    for col in nonstandard_cols:
        col_new = col + "_was_invalid"
        X_train[col_new] = X_train[col].isin(nonstandard_values).astype(int)
        X_valid[col_new] = X_valid[col].isin(nonstandard_values).astype(int)
        X_test[col_new] = X_test[col].isin(nonstandard_values).astype(int)

    print('\nCounts of invalid data')
    for col in nonstandard_cols:
        print(X_train[col + "_was_invalid"].value_counts())

    # Replace the invalid values with NaN
    for name, df in datasets.items():
        df[nonstandard_cols] = df[nonstandard_cols].replace(nonstandard_values, np.nan)

    # Print value counts after NaN replacement
    for name, df in datasets.items():
        print(f"\n{name} dataset value counts:")
        for column in nonstandard_cols:
            print(df[column].value_counts())
        print("*" * 40)

    print("\n[handle_invalid_values] Invalid values flagged and replaced with NaN.")
    return X_train, X_valid, X_test, X_train_eda_raw


# 1.7 continued - handling negative values in numerical features
def handle_negative_values(X_train, X_valid, X_test, X_train_eda_clean):
    # Checks every numerical columns min 5 values
    for col in X_train.select_dtypes(include=np.number).columns:
        print(X_train[col].sort_values().head(5))
        print("_" * 40)

    # List all columns that have negative values
    negative_cols = ['days_since_last_login', 'avg_time_spent',
                     'avg_frequency_login_days', 'points_in_wallet']

    # Create new columns with neg value flag
    cols_flag = []
    for i, col in enumerate(negative_cols):
        cols_flag.append(col + "_was_negative")

    for col_new, col in zip(cols_flag, negative_cols):
        X_train[col_new] = (X_train[col] < 0).astype(int)
        X_valid[col_new] = (X_valid[col] < 0).astype(int)
        X_test[col_new] = (X_test[col] < 0).astype(int)

    # Validate flag columns
    for col, flag_col in zip(negative_cols, cols_flag):
        print(f"\nColumn: {col}")
        train_neg = (X_train[col] < 0).sum()
        train_flag = X_train[flag_col].sum()
        print(f"Train  - negatives: {train_neg}, flags: {train_flag}")
        print("*" * 40)

    datasets = {
        "Train": X_train, "Validation": X_valid, "Test": X_test,
        "Train EDA Clean": X_train_eda_clean
    }

    # Replace negative values with NaN
    for name, df in datasets.items():
        print(f"\n{name} dataset")
        for column in negative_cols:
            df.loc[df[column] < 0, column] = np.nan
            print(f"\n{column} smallest values:")
            print(df[column].sort_values().head(5))
        print("*" * 40)

    print("\n[handle_negative_values] Negative values flagged and replaced with NaN.")
    return X_train, X_valid, X_test, X_train_eda_clean


# 1.7 continued - imputing missing values
def impute_missing_values(X_train, X_valid, X_test, X_train_eda_imputed):
    datasets = {
        "Train": X_train, "Validation": X_valid, "Test": X_test,
        "Train EDA Imputed": X_train_eda_imputed
    }

    # List numerical columns
    numerical_cols = ['points_in_wallet', 'avg_frequency_login_days',
                      'avg_time_spent', 'days_since_last_login']

    # For numerical columns, median should be used
    median = SimpleImputer(strategy="median")
    # Fit the imputer on training data only to avoid data leakage
    median.fit(X_train[numerical_cols])
    # Apply the learned median values to all datasets
    for df in datasets.values():
        df[numerical_cols] = median.transform(df[numerical_cols])

    # Categorical columns
    categorical_cols = [
        'gender', 'region_category', 'joined_through_referral',
        'preferred_offer_types', 'medium_of_operation'
    ]

    # Mode imputer
    mode = SimpleImputer(strategy="most_frequent")
    mode.fit(X_train[categorical_cols])
    for df in datasets.values():
        df[categorical_cols] = mode.transform(df[categorical_cols])

    # Check missing values for all datasets
    for name, df in datasets.items():
        print(f"\n{name} missing values:")
        print(df.isnull().sum())
        print("*" * 40)

    print("\n[impute_missing_values] Missing values imputed (median for numeric, mode for categorical).")
    return X_train, X_valid, X_test, X_train_eda_imputed


# 1.8 Reviewing data patterns
def analyze_patterns(X_train):
    # Look at statistical analysis using describe method
    print("\nNumerical statistics:")
    print(X_train.describe().T)

    # Filter categorical data and use describe method
    print("\nCategorical statistics:")
    print(X_train.describe(include='object').T)

    # Check the correlation between the original numeric data except flag columns
    correlation = [
        "age", "days_since_last_login", "avg_time_spent",
        "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet"
    ]

    plt.figure(figsize=(10, 8))
    sns.heatmap(X_train[correlation].corr(), annot=True, square=True, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Churn Data")
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_numeric.png"), bbox_inches='tight')
    plt.close()

    # Check the skewness of original numeric columns except flag columns
    skewness = X_train[correlation].skew().sort_values(ascending=False)
    print("\nSkewness:")
    print(skewness)

    print("\n[analyze_patterns] Data patterns analyzed, correlation heatmap saved.")


# 1.9 Detecting and handling outliers
def handle_outliers(X_train, X_valid, X_test, X_train_eda_imputed, X_train_eda_outliers_removed):
    # Detect outliers in numerical features using the IQR method
    num_cols = [
        "age", "days_since_last_login", "avg_time_spent",
        "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet"
    ]

    outlier_bounds = {}

    for column in num_cols:
        q1 = X_train[column].quantile(0.25)
        q3 = X_train[column].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outlier_bounds[column] = (low, high)
        outlier_count = ((X_train[column] < low) | (X_train[column] > high)).sum()
        print(f"{column}: low: {low:.2f}, high: {high:.2f}, outliers in train: {outlier_count}")

    # Visualize the distribution of numerical features before capping
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes_flat = axes.flatten()
    for i, column in enumerate(num_cols):
        axes_flat[i].boxplot(X_train[column].dropna())
        axes_flat[i].set_title("Boxplot of " + column)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplots_before_capping.png"), bbox_inches='tight')
    plt.close()

    cap_cols = ['avg_time_spent', 'avg_transaction_value',
                'avg_frequency_login_days', 'points_in_wallet']

    datasets = {
        "Train": X_train, "Validation": X_valid, "Test": X_test,
        "Train EDA Imputed": X_train_eda_imputed,
        "Train EDA Outliers Removed": X_train_eda_outliers_removed
    }

    # Apply IQR-based capping (winsorization) to selected numerical features
    for df in datasets.values():
        for column in cap_cols:
            low, high = outlier_bounds[column]
            df[column] = df[column].clip(lower=low, upper=high)

    # Testing that no observations exceed the learned IQR bounds
    for column in cap_cols:
        low, high = outlier_bounds[column]
        print(f"\nColumn: {column}")
        for name, df in datasets.items():
            count = ((df[column] < low) | (df[column] > high)).sum()
            print(f"{name}: {count} values outside bounds")
        print("*" * 40)

    # Visualize the distribution after capping
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes_flat = axes.flatten()
    for i, column in enumerate(num_cols):
        axes_flat[i].boxplot(X_train[column].dropna())
        axes_flat[i].set_title("Boxplot of " + column)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplots_after_capping.png"), bbox_inches='tight')
    plt.close()

    print("\nDescriptive stats after capping:")
    print(X_train.describe().T)

    print("\n[handle_outliers] Outliers capped. Boxplots saved.")
    return X_train, X_valid, X_test, X_train_eda_imputed, X_train_eda_outliers_removed, outlier_bounds


# ***********************************************************
# Step 3: Feature Engineering
# ***********************************************************

# Converting columns to proper numeric types after encoding
def convert_to_numeric_types(X_train, X_valid, X_test):
    cols_to_numeric = X_train.columns.difference(['joining_date'])
    X_train[cols_to_numeric] = X_train[cols_to_numeric].apply(pd.to_numeric, errors="coerce")
    X_valid[cols_to_numeric] = X_valid[cols_to_numeric].apply(pd.to_numeric, errors="coerce")
    X_test[cols_to_numeric] = X_test[cols_to_numeric].apply(pd.to_numeric, errors="coerce")
    print("\n[convert_to_numeric_types] All non-date columns converted to numeric.")
    print(X_train.info())
    return X_train, X_valid, X_test


# 3.1 Calculating tenure days
def calculate_tenure(X_train, X_valid, X_test):
    # The last joining date in training subset is used as reference day
    last_joining_date = pd.to_datetime(X_train["joining_date"], errors="coerce").max()

    joining_date_train = pd.to_datetime(X_train["joining_date"], errors="coerce")
    joining_date_valid = pd.to_datetime(X_valid["joining_date"], errors="coerce")
    joining_date_test = pd.to_datetime(X_test["joining_date"], errors="coerce")

    X_train["days_of_tenure"] = (last_joining_date - joining_date_train).dt.days
    X_train["days_of_tenure"] = X_train["days_of_tenure"].clip(lower=0)
    X_valid["days_of_tenure"] = (last_joining_date - joining_date_valid).dt.days
    X_valid["days_of_tenure"] = X_valid["days_of_tenure"].clip(lower=0)
    X_test["days_of_tenure"] = (last_joining_date - joining_date_test).dt.days
    X_test["days_of_tenure"] = X_test["days_of_tenure"].clip(lower=0)

    print(X_train['days_of_tenure'].head(5))
    print("\n[calculate_tenure] Tenure days calculated.")
    return X_train, X_valid, X_test


# 3.2 Calculating recency ratio
def calculate_recency(X_train, X_valid, X_test):
    X_train["recency_ratio"] = X_train["days_since_last_login"] / (X_train["days_of_tenure"] + 1)
    X_valid["recency_ratio"] = X_valid["days_since_last_login"] / (X_valid["days_of_tenure"] + 1)
    X_test["recency_ratio"] = X_test["days_since_last_login"] / (X_test["days_of_tenure"] + 1)
    print(X_train['recency_ratio'].head(5))
    print("\n[calculate_recency] Recency ratio calculated.")
    return X_train, X_valid, X_test


# 3.3 Flagging new users
def flag_new_users(X_train, X_valid, X_test):
    X_train["new_user"] = (X_train["days_of_tenure"] < 30).astype(int)
    X_valid["new_user"] = (X_valid["days_of_tenure"] < 30).astype(int)
    X_test["new_user"] = (X_test["days_of_tenure"] < 30).astype(int)
    print(X_train[["new_user", "days_of_tenure"]].head())
    print("\n[flag_new_users] New user flag created.")
    return X_train, X_valid, X_test


# 3.4 Calculating login frequency ratio
def calculate_frequency(X_train, X_valid, X_test):
    X_train["login_freq_ratio"] = 1 / (X_train["avg_frequency_login_days"] + 1)
    X_valid["login_freq_ratio"] = 1 / (X_valid["avg_frequency_login_days"] + 1)
    X_test["login_freq_ratio"] = 1 / (X_test["avg_frequency_login_days"] + 1)
    print(X_train[["login_freq_ratio", "avg_frequency_login_days"]].head())

    # User engagement
    X_train["user_engagement"] = np.log1p(X_train["avg_time_spent"]) * X_train["login_freq_ratio"]
    X_valid["user_engagement"] = np.log1p(X_valid["avg_time_spent"]) * X_valid["login_freq_ratio"]
    X_test["user_engagement"] = np.log1p(X_test["avg_time_spent"]) * X_test["login_freq_ratio"]
    print(X_train[["user_engagement", "avg_time_spent", "login_freq_ratio"]].head())

    print("\n[calculate_frequency] Login frequency ratio and user engagement calculated.")
    return X_train, X_valid, X_test


# 3.5 Calculating RFM score
def calculate_rfm(X_train, X_valid, X_test):
    X_train["rfm"] = np.log1p(X_train["avg_transaction_value"]) * X_train["login_freq_ratio"] / (X_train["days_since_last_login"] + 1)
    X_valid["rfm"] = np.log1p(X_valid["avg_transaction_value"]) * X_valid["login_freq_ratio"] / (X_valid["days_since_last_login"] + 1)
    X_test["rfm"] = np.log1p(X_test["avg_transaction_value"]) * X_test["login_freq_ratio"] / (X_test["days_since_last_login"] + 1)
    print(X_train[["rfm", "avg_transaction_value", "login_freq_ratio", "days_since_last_login"]].head())
    print("\n[calculate_rfm] RFM score calculated.")
    return X_train, X_valid, X_test


# 3.6 Calculating offer and discount preference score
def calculate_offer_discount(X_train, X_valid, X_test):
    X_train["offer_discount_score"] = X_train["used_special_discount"].astype(int) + X_train["offer_application_preference"].astype(int)
    X_valid["offer_discount_score"] = X_valid["used_special_discount"].astype(int) + X_valid["offer_application_preference"].astype(int)
    X_test["offer_discount_score"] = X_test["used_special_discount"].astype(int) + X_test["offer_application_preference"].astype(int)
    print(X_train[["offer_discount_score", "used_special_discount", "offer_application_preference"]].head())
    print("\n[calculate_offer_discount] Offer discount score calculated.")
    return X_train, X_valid, X_test


# 3.7 Scaling features
def scale_features(X_train, X_valid, X_test, X_train_before_fe, X_valid_before_fe, X_test_before_fe):
    # Identify binary columns (values only 0 or 1)
    is_binary = X_train.apply(lambda col: set(col.dropna().unique()).issubset({0, 1}))
    scale_cols = X_train.columns[~is_binary].tolist()
    print("\nScaled Features:")
    print(scale_cols)

    # Copy dataset before scaling (for feature engineered unscaled variant)
    X_train_fe = X_train.copy()
    X_valid_fe = X_valid.copy()
    X_test_fe = X_test.copy()

    # Final model scaling
    scaler_final = StandardScaler()
    scaler_final.fit(X_train[scale_cols])
    for df in [X_train, X_valid, X_test]:
        df[scale_cols] = scaler_final.transform(df[scale_cols])

    # Baseline model scaling
    exc = ['days_of_tenure', 'recency_ratio', 'login_freq_ratio',
           'user_engagement', 'rfm', 'offer_discount_score']
    scale_cols_before_fe = [col for col in scale_cols if col not in exc]

    scaler_baseline = StandardScaler()
    scaler_baseline.fit(X_train_before_fe[scale_cols_before_fe])
    for df in [X_train_before_fe, X_valid_before_fe, X_test_before_fe]:
        df[scale_cols_before_fe] = scaler_baseline.transform(df[scale_cols_before_fe])

    print("\nDescriptive stats after scaling (min/max):")
    print(X_train.describe().loc[["min", "max"]].T)

    print("\n[scale_features] Feature scaling completed.")
    return X_train, X_valid, X_test, X_train_before_fe, X_valid_before_fe, X_test_before_fe, X_train_fe, X_valid_fe, X_test_fe


# ***********************************************************
# Pipeline orchestrators
# ***********************************************************

def run_data_cleaning(csv_path):
    print("\n" + "*" * 60)
    print("STEP 1: DATA CLEANING & PRE-PROCESSING (1.1 - 1.9)")
    print("*" * 60)

    df = load_data(csv_path)
    check_data_info(df)
    analyze_target(df)
    check_duplicates(df)

    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_eda_raw, y_train_eda = split_data(df)
    X_train, X_valid, X_test = create_referral_features(X_train, X_valid, X_test)
    X_train, X_valid, X_test, X_train_eda_raw = drop_id_columns(X_train, X_valid, X_test, X_train_eda_raw)
    X_train, X_valid, X_test, X_train_eda_raw = handle_invalid_values(X_train, X_valid, X_test, X_train_eda_raw)

    # Create eda_clean copy before negative handling
    X_train_eda_clean = X_train_eda_raw.copy()

    X_train, X_valid, X_test, X_train_eda_clean = handle_negative_values(X_train, X_valid, X_test, X_train_eda_clean)

    # Create eda_imputed copy before imputation
    X_train_eda_imputed = X_train_eda_clean.copy()

    X_train, X_valid, X_test, X_train_eda_imputed = impute_missing_values(X_train, X_valid, X_test, X_train_eda_imputed)

    analyze_patterns(X_train)

    # Create eda_outliers_removed copy from eda_clean before capping
    X_train_eda_outliers_removed = X_train_eda_clean.copy()

    X_train, X_valid, X_test, X_train_eda_imputed, X_train_eda_outliers_removed, outlier_bounds = handle_outliers(
        X_train, X_valid, X_test, X_train_eda_imputed, X_train_eda_outliers_removed
    )

    num_cols = ["age", "days_since_last_login", "avg_time_spent",
                "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet"]

    print("\n[run_data_cleaning] Step 1 (1.1-1.9) completed.")
    return (X_train, X_valid, X_test, y_train, y_valid, y_test,
            X_train_eda_raw, X_train_eda_clean, X_train_eda_outliers_removed,
            X_train_eda_imputed, y_train_eda, num_cols, outlier_bounds)


def run_feature_engineering(X_train, X_valid, X_test):
    print("\n" + "*" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("*" * 60)

    X_train, X_valid, X_test = convert_to_numeric_types(X_train, X_valid, X_test)

    # Copy data for baseline model before feature engineering
    X_train_before_fe = X_train.copy()
    X_valid_before_fe = X_valid.copy()
    X_test_before_fe = X_test.copy()

    X_train, X_valid, X_test = calculate_tenure(X_train, X_valid, X_test)

    # Drop joining_date from all subsets
    X_train.drop(columns=['joining_date'], inplace=True)
    X_valid.drop(columns=['joining_date'], inplace=True)
    X_test.drop(columns=['joining_date'], inplace=True)
    X_train_before_fe.drop(columns=['joining_date'], inplace=True)
    X_valid_before_fe.drop(columns=['joining_date'], inplace=True)
    X_test_before_fe.drop(columns=['joining_date'], inplace=True)

    X_train, X_valid, X_test = calculate_recency(X_train, X_valid, X_test)
    X_train, X_valid, X_test = flag_new_users(X_train, X_valid, X_test)
    X_train, X_valid, X_test = calculate_frequency(X_train, X_valid, X_test)
    X_train, X_valid, X_test = calculate_rfm(X_train, X_valid, X_test)
    X_train, X_valid, X_test = calculate_offer_discount(X_train, X_valid, X_test)

    (X_train, X_valid, X_test,
     X_train_before_fe, X_valid_before_fe, X_test_before_fe,
     X_train_fe, X_valid_fe, X_test_fe) = scale_features(
        X_train, X_valid, X_test, X_train_before_fe, X_valid_before_fe, X_test_before_fe
    )

    print("\n[run_feature_engineering] Step 3 completed.")
    return (X_train, X_valid, X_test,
            X_train_before_fe, X_valid_before_fe, X_test_before_fe,
            X_train_fe, X_valid_fe, X_test_fe)
