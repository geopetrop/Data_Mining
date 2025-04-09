import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

#Make sure you update file paths to your local situation (Line 13, Line 175)
#We need to find a way to clean the really messy birthday and time to bed columns
#K-nearest neighbor approximation or any other method you like is an open task for method 2 of 1B, if someone wants to take a look at that
#All my reporitng of figures and results for the report is still at the preliminary level

#Task 1A
df = pd.read_csv("/Users/georgepetropoulos/Desktop/DMT/ODI-2025.csv", sep = ';')
numeric_cols = [
    'How many students do you estimate there are in the room?',
    'What is your stress level (0-100)?',
    'How many hours per week do you do sports (in whole hours)? ',
    'Give a random number'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in numeric_cols:
    plt.figure()
    df[col].plot(kind='box')
    plt.title(f'Boxplot of {col}')
    plt.ylabel('Value')
    plt.show()
    plt.figure()
    df[col].hist(bins=20)
    plt.title(f'Histogram of {col}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

corr_matrix = df[numeric_cols].corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


#Task 1B
NUMBER_WORDS = {
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

    if val in NUMBER_WORDS:
        return NUMBER_WORDS[val]

    #Check for range like "180-200"
    range_match = re.match(r'^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$', val)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0

    #Check for arithmetic expression like "20*5*5"
    expr_match = re.match(r'^[0-9\.\+\-\*/\s]+$', val)
    if expr_match:
        try:
            return float(eval(val))
        except:
            pass

    #Replace commas if it's likely a decimal comma
    val_decimal = val.replace(',', '.')

    #Extract numeric substring from text
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
    Computes Q1, Q3 and IQR (Q3 - Q1) and flag outliers as anything < (Q1 - iqr_factor * IQR) or > (Q3 + iqr_factor * IQR).
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
    Parse into numeric and check if x < 0 => 0, if x > 100 => 100.
    """
    parsed = series.apply(parse_numeric_value).astype(float)
    parsed = parsed.clip(lower=0, upper=100)
    return parsed

def clean_hours_sports(series):
    """
    Parse into numeric and clamp to [0, 60].
    """
    parsed = series.apply(parse_numeric_value).astype(float)
    parsed = parsed.clip(lower=0, upper=60)
    return parsed

def clean_students_in_room(series):
    """
    Parse into numeric and treat negative entries as NaN.
    """
    parsed = series.apply(parse_numeric_value).astype(float)
    parsed.loc[parsed < 0] = np.nan
    return parsed

df.iloc[:, 9] = clean_iqr(df.iloc[:, 9], parse_numeric_value, iqr_factor=1.5)
df.iloc[:, 9] = clean_students_in_room(df.iloc[:, 9])
df.iloc[:, 10] = clean_stress_level(df.iloc[:, 10])
df.iloc[:, 11] = clean_iqr(df.iloc[:, 11], parse_numeric_value, iqr_factor=3)
df.iloc[:, 11] = clean_hours_sports(df.iloc[:, 11])
df.iloc[:, 12] = clean_iqr(df.iloc[:, 12], parse_numeric_value, iqr_factor=1.5)

def is_missing(value):
    """
    Decides if the value should be considered missing:
      1. Empty or known placeholders.
      2. No letters at all.
      3. More than 3 'invalid' characters (anything not a-z, A-Z, '!', or '?' -- ignoring whitespace).
    """
    val = str(value).strip()

    placeholders = ["-", "n/a", "none", "()", "nan"]
    if val == "" or val.lower() in placeholders:
        return True

    if not re.search(r"[a-zA-Z]", val):
        return True

    invalid_chars = re.findall(r"[^a-zA-Z!\?\s]", val)
    if len(invalid_chars) > 3:
        return True

    return False

df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: np.nan if is_missing(x) else x)
df.iloc[:, 14] = df.iloc[:, 14].apply(lambda x: np.nan if is_missing(x) else x)
df.iloc[:, 15] = df.iloc[:, 15].apply(lambda x: np.nan if is_missing(x) else x)

output_path = "/Users/georgepetropoulos/Desktop/cleaned_ODI-2025.csv"
df.to_csv(output_path, index=False)

for col in numeric_cols:
    plt.figure()
    df[col].plot(kind='box')
    plt.title(f'Boxplot of {col}')
    plt.ylabel('Value')
    plt.show()
    plt.figure()
    df[col].hist(bins=20)
    plt.title(f'Histogram of {col}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

corr_matrix = df[numeric_cols].corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

#Method 1: Unknown Categorical, Mean Numerical Missing Values
for col in numeric_cols:
    mean_value = df[col].mean(skipna=True)
    df[col].fillna(mean_value, inplace=True)

for col in categorical_columns:
    df[col].fillna('unknown', inplace=True)

#Method 2: K-Neirghest Neighbor Approximation
