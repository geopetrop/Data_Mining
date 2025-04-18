import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.stats import skew

#Final update to task 1B. I believe random imputation makes conceptual sense over simplying replacing missing categorical values with "unknown"
#since we have multiple columns that are highly dispersed in their answers. This allows us to better preserve their distributions. For numerical attributes I chose
#the mean_median approach, since there is very little correlation in our numerical columns Knn is overkill. Since a few of our numerical attributes displayed high skeweness
#I added a choice determined by threshold to use either mean or median in how the missing data is replaced. You can view some plots
#of the cleaned data if you uncomment the results section (line 270-292)

#watch out to edit your file paths (line 19, 152, 293). 

###The time columns are still uncleaned, too messy!

df = pd.read_csv("/Users/georgepetropoulos/Desktop/DMT/ODI-2025.csv", sep=';')

numeric_cols = [
    'How many students do you estimate there are in the room?',
    'What is your stress level (0-100)?',
    'How many hours per week do you do sports (in whole hours)? ',
    'Give a random number'
]

# Task 1B
number_words = {
    'zero': 0.0,
    'one': 1.0,
    'two': 2.0,
    'three': 3.0,
    'four': 4.0,
    'five': 5.0,
    'six': 6.0,
    'seven': 7.0,
    'eight': 8.0,
    'nine': 9.0,
    'ten': 10.0,
}

def parse_numeric_value(val):
    """
    Attempts to parse a messy string into a float.
    Returns None if parsing fails.
    """
    if not isinstance(val, str):
        return val

    val = val.strip().lower()
    if not val:
        return None

    if val in number_words:
        return number_words[val]

    # Check for range like "180-200"
    range_match = re.match(r'^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$', val)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0

    # Check for arithmetic expression like "20*5*5"
    expr_match = re.match(r'^[0-9\.\+\-\*/\s]+$', val)
    if expr_match:
        try:
            return float(eval(val))
        except:
            pass

    # Replace commas if it's likely a decimal comma
    val_decimal = val.replace(',', '.')

    # Extract numeric substring from text
    num_search = re.search(r'[+-]?\d+(\.\d+)?([eE][+-]?\d+)?', val_decimal)
    if num_search:
        num_str = num_search.group(0)
        try:
            return float(num_str)
        except:
            return None

    return None

def clean_iqr(series, parse_fn, iqr_factor=1.5):
    """
    Computes Q1, Q3, and IQR (Q3 - Q1). Flags outliers as < (Q1 - iqr_factor*IQR)
    or > (Q3 + iqr_factor*IQR).
    """
    parsed = series.apply(parse_fn).astype(float)
    q1 = parsed.quantile(0.25)
    q3 = parsed.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or pd.isnull(iqr):
        return parsed

    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    parsed[(parsed < lower_bound) | (parsed > upper_bound)] = np.nan
    return parsed

def clean_stress_level(series):
    """
    Parse into numeric and clip to [0, 100].
    """
    parsed = series.apply(parse_numeric_value).astype(float)
    return parsed.clip(lower=0, upper=100)

def clean_hours_sports(series):
    """
    Parse into numeric and clip to [0, 60].
    """
    parsed = series.apply(parse_numeric_value).astype(float)
    return parsed.clip(lower=0, upper=60)

def clean_students_in_room(series):
    """
    Parse into numeric; treat negative as NaN.
    """
    parsed = series.apply(parse_numeric_value).astype(float)
    parsed.loc[parsed < 0] = np.nan
    return parsed

# Outlier / Data Cleaning
df.iloc[:, 9] = clean_iqr(df.iloc[:, 9], parse_numeric_value, iqr_factor=1.5)
df.iloc[:, 9] = clean_students_in_room(df.iloc[:, 9])
df.iloc[:, 10] = clean_stress_level(df.iloc[:, 10])
df.iloc[:, 11] = clean_iqr(df.iloc[:, 11], parse_numeric_value, iqr_factor=3)
df.iloc[:, 11] = clean_hours_sports(df.iloc[:, 11])
df.iloc[:, 12] = clean_iqr(df.iloc[:, 12], parse_numeric_value, iqr_factor=1.5)

def is_missing(value):
    """
    Decides if the value should be considered missing based on placeholders, no letters, etc.
    """
    val = str(value).strip()
    placeholders = ["-", "n/a", "none", "()", "nan"]
    if val == "" or val.lower() in placeholders:
        return True
    if not re.search(r"[a-zA-Z]", val):
        return True
    invalid_chars = re.findall(r"[^a-zA-Z!\?\s]", val)
    return (len(invalid_chars) > 3)

df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: np.nan if is_missing(x) else x)
df.iloc[:, 14] = df.iloc[:, 14].apply(lambda x: np.nan if is_missing(x) else x)
df.iloc[:, 15] = df.iloc[:, 15].apply(lambda x: np.nan if is_missing(x) else x)

output_path = "/Users/georgepetropoulos/Desktop/initial_cleaned_ODI-2025.csv"
df.to_csv(output_path, index=False)

#Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

def knn_numerical(dataframe, numeric_cols, n_neighbors=5):
    """
    Scales numeric columns with StandardScaler, applies KNN imputation, and inverse-scales.
    """
    df_numeric = dataframe[numeric_cols]

    # Temporary fill for scaling
    df_temp = df_numeric.fillna(df_numeric.mean())
    scaler = StandardScaler()
    df_numeric_scaled = scaler.fit_transform(df_temp)

    # KNN Imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_scaled = imputer.fit_transform(df_numeric_scaled)

    # Inverse transform
    imputed = scaler.inverse_transform(imputed_scaled)

    # Overwrite the original numeric columns
    dataframe.loc[:, numeric_cols] = imputed

    return dataframe

def random_categorical(df, categorical_cols, random_state=42):
    """
    Impute missing categorical values by sampling from the existing distribution.
    """
    np.random.seed(random_state)
    for col in categorical_cols:
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue

        known_values = df.loc[~missing_mask, col]
        if known_values.empty:
            continue

        value_counts = known_values.value_counts(dropna=False)
        categories = value_counts.index
        freqs = value_counts.values.astype(float)
        probs = freqs / freqs.sum()

        n_missing = missing_mask.sum()
        random_fills = np.random.choice(categories, size=n_missing, p=probs)
        df.loc[missing_mask, col] = random_fills

    return df

def clean_missing_values(df, numeric_cols, categorical_cols, numeric_method='mean_median', cat_method='unknown',
    n_neighbors=5, random_state=42, skew_threshold=1.0):
    """
    Handle missing values in numeric and categorical columns in place, returning
    the updated DataFrame reference.

    numeric_method:
        - 'mean_median': decides mean vs median based on skew
        - 'knn': uses KNN-based imputation

    cat_method:
        - 'unknown': fills missing categoricals with 'unknown'
        - 'random': samples from existing distribution in that column

    skew_threshold:
        - If abs(skew) > skew_threshold, use median; else use mean.
    """

    imputation_log = {}

    # 1. Handle Numeric Columns
    if numeric_method == 'mean_median':
        for col in numeric_cols:
            col_skew = df[col].skew(skipna=True)
            if abs(col_skew) > skew_threshold:
                # Use median if data is highly skewed
                impute_value = df[col].median(skipna=True)
                imputation_log[col] = "median"
            else:
                # Use mean otherwise
                impute_value = df[col].mean(skipna=True)
                imputation_log[col] = "mean"

            df[col].fillna(round(impute_value, 2), inplace=True)

    elif numeric_method == 'knn':
        # If using KNN, it applies to all numeric columns
        df = knn_numerical(df, numeric_cols, n_neighbors=n_neighbors)
        for col in numeric_cols:
            imputation_log[col] = "knn"

    else:
        raise ValueError("numeric_method must be 'mean_median' or 'knn'")

    # 2. Handle Categorical Columns
    if cat_method == 'unknown':
        for col in categorical_cols:
            df[col].fillna('unknown', inplace=True)

    elif cat_method == 'random':
        df = random_categorical(df, categorical_cols, random_state=random_state)

    else:
        raise ValueError("cat_method must be 'unknown' or 'random'")

    return df, imputation_log

df_cleaned, log = clean_missing_values(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols, numeric_method='mean_median', cat_method='random',
    skew_threshold=1.0, n_neighbors=5, random_state=42)

#tracking median/mean results
#print(log)

#Printing results
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#
# for col in numeric_cols:
#     plt.figure()
#     df[col].plot(kind='box')
#     plt.title(f'Boxplot of {col}')
#     plt.ylabel('Value')
#     plt.show()
#
#     plt.figure()
#     df[col].hist(bins=20)
#     plt.title(f'Histogram of {col}')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.show()
#
# corr_matrix = df[numeric_cols].corr(method='pearson')
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Matrix')
# plt.show()

output_path = "/Users/georgepetropoulos/Desktop/final_cleaned_ODI-2025.csv"
df_cleaned.to_csv(output_path, index=False)
