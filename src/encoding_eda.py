import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ***********************************************************
# Step 1.10: Categorical Encoding
# ***********************************************************

# 1.10 Encoding date and time features
def extract_datetime_features(X_train, X_valid, X_test):
    # Feature Engineering for 'joining_date' and 'last_visit_time'
    # Convert to datetime and extract numerical features
    print("\nPerforming feature engineering for 'joining_date' and 'last_visit_time'...")
    for df_subset in [X_train, X_valid, X_test]:
        if 'joining_date' in df_subset.columns and df_subset['joining_date'].dtype == 'object':
            df_subset['joining_date'] = pd.to_datetime(df_subset['joining_date'])
            df_subset['joining_year'] = df_subset['joining_date'].dt.year
            df_subset['joining_month'] = df_subset['joining_date'].dt.month
            df_subset['joining_day'] = df_subset['joining_date'].dt.day
            df_subset['joining_dayofweek'] = df_subset['joining_date'].dt.dayofweek
            print("Engineered 'joining_date' features.")

        if 'last_visit_time' in df_subset.columns and df_subset['last_visit_time'].dtype == 'object':
            df_subset['last_visit_time_dt'] = pd.to_datetime('2000-01-01 ' + df_subset['last_visit_time'])
            df_subset['last_visit_hour'] = df_subset['last_visit_time_dt'].dt.hour
            df_subset['last_visit_minute'] = df_subset['last_visit_time_dt'].dt.minute
            df_subset.drop(columns=['last_visit_time', 'last_visit_time_dt'], inplace=True)
            print("Engineered 'last_visit_time' features.")

    print("\n[extract_datetime_features] Datetime features extracted.")
    return X_train, X_valid, X_test


# 1.10 Label encoding binary categorical features
def label_encode_binary(X_train, X_valid, X_test):
    # 2. Apply Label Encoding to binary categorical features
    binary_cols = ['gender', 'joined_through_referral', 'used_special_discount',
                   'offer_application_preference', 'past_complaint']

    label_encoder = LabelEncoder()

    print(f"\nApplying Label Encoding to binary columns: {binary_cols}")
    for col in binary_cols:
        if col in X_train.columns and X_train[col].dtype == 'object':
            X_train[col] = label_encoder.fit_transform(X_train[col])
            X_valid[col] = label_encoder.transform(X_valid[col])
            X_test[col] = label_encoder.transform(X_test[col])
            print(f"Label encoded column: {col}")
        else:
            print(f"Skipping Label Encoding for '{col}' (not found or not object dtype).")

    print("\n[label_encode_binary] Binary label encoding completed.")
    return X_train, X_valid, X_test, binary_cols


# 1.10 One-hot encoding nominal categorical features
def one_hot_encode_nominal(X_train, X_valid, X_test):
    # 3. Apply One-Hot Encoding to nominal categorical features
    nominal_cols = ['region_category', 'membership_category', 'preferred_offer_types',
                    'medium_of_operation', 'internet_option', 'complaint_status', 'feedback']

    ohe_cols_present = [col for col in nominal_cols if col in X_train.columns and X_train[col].dtype == 'object']

    print(f"\nApplying One-Hot Encoding to nominal columns: {ohe_cols_present}")

    if ohe_cols_present:
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), ohe_cols_present)
            ],
            remainder='passthrough'
        )

        X_train_encoded = preprocessor.fit_transform(X_train)
        X_valid_encoded = preprocessor.transform(X_valid)
        X_test_encoded = preprocessor.transform(X_test)

        ohe_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(ohe_cols_present)
        remaining_cols = [col for col in X_train.columns if col not in ohe_cols_present]

        X_train = pd.DataFrame(X_train_encoded, columns=list(ohe_feature_names) + remaining_cols, index=X_train.index)
        X_valid = pd.DataFrame(X_valid_encoded, columns=list(ohe_feature_names) + remaining_cols, index=X_valid.index)
        X_test = pd.DataFrame(X_test_encoded, columns=list(ohe_feature_names) + remaining_cols, index=X_test.index)
        print("One-Hot Encoding applied successfully.")
    else:
        print("No nominal object columns found for One-Hot Encoding, skipping.")

    print("\n[one_hot_encode_nominal] Nominal one-hot encoding completed.")
    return X_train, X_valid, X_test, nominal_cols


# ***********************************************************
# Step 2: EDA (2.1 - 2.3)
# ***********************************************************

# 2.1 Target churn distribution
def plot_target_distribution(y_train_eda):
    churn_label = ["Churn", "Not Churn"]
    imbalance_train = y_train_eda.value_counts(normalize=True)
    plt.figure(figsize=(8, 6))
    plt.pie(imbalance_train, labels=churn_label, startangle=90, autopct='%1.1f%%')
    plt.title('Target Churn Distribution')
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_target_churn_distribution.png"), bbox_inches='tight')
    plt.close()
    print("\n[plot_target_distribution] Target distribution chart saved.")


# 2.2 Descriptive statistics
def show_descriptive_stats(X_train_eda_raw, X_train_eda_outliers_removed):
    # Before removing outliers
    print("\nDescriptive Statistics - Numerical (Before removing outliers):")
    print(X_train_eda_raw.describe().transpose())

    # After removing outliers and negative values
    print("\nDescriptive Statistics - Numerical (After removing outliers):")
    print(X_train_eda_outliers_removed.describe().transpose())

    # Categorical Variables
    print("\nDescriptive Statistics - Categorical:")
    print(X_train_eda_raw.describe(include='object').transpose())

    print("\n[show_descriptive_stats] Descriptive statistics printed.")


# 2.3 Data distribution - numerical variables
def plot_numerical_distributions(X_train_eda_raw, X_train_eda_outliers_removed):
    # Function to plot density distribution curve for numerical variables
    def plot_kde_num(df, title_suffix, filename):
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        for variable, subplot in zip(num_cols, ax.flatten()):
            sns.kdeplot(x=df[variable], ax=subplot)
            subplot.set_xlabel(variable, fontsize=10)

        if len(num_cols) < 6:
            fig.delaxes(ax[1][2])

        plt.suptitle(f"Numerical Distributions {title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
        plt.close()

    # Before removing outliers
    plot_kde_num(X_train_eda_raw, "(Before Outlier Removal)", "eda_kde_before_outliers.png")

    # After removing outliers and replacing negative values to NaN
    plot_kde_num(X_train_eda_outliers_removed, "(After Outlier Removal)", "eda_kde_after_outliers.png")

    print("\n[plot_numerical_distributions] KDE distribution plots saved.")


# 2.3 Data distribution - categorical variables
def plot_categorical_distributions(X_train_eda_raw, X_train):
    # Customer Demographics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(x='gender', data=X_train_eda_raw, ax=axes[0])
    axes[0].set_title("Gender Distribution")
    sns.countplot(x='region_category', data=X_train_eda_raw, ax=axes[1])
    axes[1].set_title("Region Category Distribution")
    sns.countplot(y='membership_category', data=X_train_eda_raw, ax=axes[2])
    axes[2].set_title("Membership Category Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_demographics.png"), bbox_inches='tight')
    plt.close()

    # Customer Behavior
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.countplot(x='preferred_offer_types', data=X_train_eda_raw, ax=axes[0])
    axes[0].set_title("Preferred Offer Types")
    axes[0].tick_params(axis='x', rotation=45)
    sns.countplot(x='medium_of_operation', data=X_train_eda_raw, ax=axes[1])
    axes[1].set_title("Medium of Operation")
    sns.countplot(x='internet_option', data=X_train_eda_raw, ax=axes[2])
    axes[2].set_title("Internet Option")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_customer_behavior.png"), bbox_inches='tight')
    plt.close()

    # Promotions & Marketing Engagement
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.countplot(x='joined_through_referral', data=X_train_eda_raw, ax=axes[0])
    axes[0].set_title("Joined Through Referral")
    sns.countplot(x='used_special_discount', data=X_train_eda_raw, ax=axes[1])
    axes[1].set_title("Used Special Discount")
    sns.countplot(x='offer_application_preference', data=X_train_eda_raw, ax=axes[2])
    axes[2].set_title("Offer Application Preference")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_promotions.png"), bbox_inches='tight')
    plt.close()

    # Complaints & Feedback
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.countplot(x='past_complaint', data=X_train_eda_raw, ax=axes[0])
    axes[0].set_title("Past Complaint")
    sns.countplot(y='complaint_status', data=X_train_eda_raw, ax=axes[1])
    axes[1].set_title("Complaint Status")
    sns.countplot(y='feedback', data=X_train_eda_raw, ax=axes[2])
    axes[2].set_title("Feedback")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_complaints_feedback.png"), bbox_inches='tight')
    plt.close()

    # Year Customer Joined
    if 'joining_year' in X_train.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(X_train['joining_year'], bins=5, edgecolor='k')
        plt.xlabel('Joining Year')
        plt.ylabel('Frequency')
        plt.title('Distribution of Joining Year')
        min_year = int(X_train['joining_year'].min())
        max_year = int(X_train['joining_year'].max())
        plt.xticks(range(min_year, max_year + 1))
        plt.savefig(os.path.join(OUTPUT_DIR, "eda_joining_year.png"), bbox_inches='tight')
        plt.close()

    # Joining Month
    if 'joining_month' in X_train.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(X_train['joining_month'], bins=23, edgecolor='k')
        plt.xlabel('Joining Month')
        plt.ylabel('Frequency')
        plt.title('Distribution of Joining Month')
        plt.savefig(os.path.join(OUTPUT_DIR, "eda_joining_month.png"), bbox_inches='tight')
        plt.close()

    print("\n[plot_categorical_distributions] All categorical distribution charts saved.")


# ***********************************************************
# Pipeline orchestrators
# ***********************************************************

def run_encoding(X_train, X_valid, X_test):
    print("\n" + "*" * 60)
    print("STEP 1.10: CATEGORICAL ENCODING")
    print("*" * 60)

    X_train, X_valid, X_test = extract_datetime_features(X_train, X_valid, X_test)
    X_train, X_valid, X_test, binary_cols = label_encode_binary(X_train, X_valid, X_test)
    X_train, X_valid, X_test, nominal_cols = one_hot_encode_nominal(X_train, X_valid, X_test)

    print("\nEncoded X_train preview:")
    print(X_train.head().T)

    print("\n[run_encoding] Encoding step completed.")
    return X_train, X_valid, X_test, binary_cols, nominal_cols


def run_basic_eda(X_train_eda_raw, X_train_eda_outliers_removed, y_train_eda, X_train):
    print("\n" + "*" * 60)
    print("STEP 2: EDA (2.1 - 2.3)")
    print("*" * 60)

    plot_target_distribution(y_train_eda)
    show_descriptive_stats(X_train_eda_raw, X_train_eda_outliers_removed)
    plot_numerical_distributions(X_train_eda_raw, X_train_eda_outliers_removed)
    plot_categorical_distributions(X_train_eda_raw, X_train)

    print("\n[run_basic_eda] Basic EDA (2.1-2.3) completed.")
