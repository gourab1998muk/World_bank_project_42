import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import os
print("update with get data path : done")
def get_data_path(filename):
    # Get absolute path to the `data` directory relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    return os.path.join(base_dir, filename)



# Step 1: Load the dataset
def load_data(file_path, file_type='csv', sheet_name=None):
    """
    Loads dataset from a given file path.
    
    Parameters:
        file_path (str): Path to the dataset file.
        file_type (str): 'csv' or 'excel' (default is 'csv').
        sheet_name (str): If loading an Excel file, specify the sheet name.

    Returns:
        DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    
    file_path = get_data_path(filename=file_path)
    if file_type == 'csv':
        df = pd.read_csv(file_path,encoding='utf-8', on_bad_lines='skip')
        return df
    elif file_type == 'excel':
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")

# Step 2: Explore the data
def explore_data(df):
    """
    Provides a basic summary of the dataset.

    Parameters:
        df (DataFrame): The dataset to explore.
    
    Returns:
        None
    """
    print("Dataset Info:\n", df.info())
    print("\nFirst 5 Rows:\n", df.head())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Rows: ", df.duplicated().sum())
    print("\nSummary Statistics:\n", df.describe(include='all'))

# Step 3: Handle missing values
def handle_missing_values(df, strategy='mean', categorical_fill='Unknown'):
    """
    Handles missing values in the dataset.

    Parameters:
        df (DataFrame): The dataset.
        strategy (str): Strategy for numerical columns ('mean', 'median', 'mode').
        categorical_fill (str): Fill value for categorical variables.

    Returns:
        DataFrame: Processed dataset.
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # Categorical columns
            df[col].fillna(categorical_fill, inplace=True)
        else:  # Numerical columns
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

# Step 4: Encode categorical variables
def encode_categorical(df):
    """
    Encodes categorical variables using Label Encoding.

    Parameters:
        df (DataFrame): The dataset.

    Returns:
        DataFrame: Processed dataset with encoded categorical features.
    """
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# Step 5: Normalize or Standardize numerical data
def scale_data(df, method='minmax'):
    """
    Scales numerical data using MinMaxScaler or StandardScaler.

    Parameters:
        df (DataFrame): The dataset.
        method (str): 'minmax' for MinMaxScaler or 'standard' for StandardScaler.

    Returns:
        DataFrame: Scaled dataset.
    """
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler

# Step 6: Remove duplicates
def remove_duplicates(df):
    """
    Removes duplicate rows from the dataset.

    Parameters:
        df (DataFrame): The dataset.

    Returns:
        DataFrame: Deduplicated dataset.
    """
    return df.drop_duplicates()

# Step 7: Detect and handle outliers using IQR method
def handle_outliers(df, method='iqr', threshold=1.5):
    """
    Detects and handles outliers in numerical data using the IQR method.

    Parameters:
        df (DataFrame): The dataset.
        method (str): Outlier detection method ('iqr' or 'zscore').
        threshold (float): Threshold for defining outliers (default 1.5 for IQR).

    Returns:
        DataFrame: Processed dataset without outliers.
    """
    numerical_cols = df.select_dtypes(include=['number']).columns
    if method == 'iqr':
        Q1 = df[numerical_cols].quantile(0.25)
        Q3 = df[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[numerical_cols] < (Q1 - threshold * IQR)) | (df[numerical_cols] > (Q3 + threshold * IQR))).any(axis=1)]
    
    return df

# Step 8: Save the processed dataset
def save_data(df, file_name):
    """
    Saves the processed dataset to a CSV file.

    Parameters:
        df (DataFrame): The dataset.
        file_name (str): Name of the output CSV file.

    Returns:
        None
    """
    df.to_csv(file_name, index=False)
    print(f"Processed dataset saved as {file_name}")

# Sample Pipeline Execution
if __name__ == "__main__":
    # Load the dataset
    file_path = "dataset.csv"  # Replace with actual path
    df = load_data(file_path)

    # Data exploration
    explore_data(df)

    # Handle missing values
    df = handle_missing_values(df, strategy='mean')

    # Encode categorical variables
    df, encoders = encode_categorical(df)

    # Normalize numerical data
    df, scaler = scale_data(df, method='minmax')

    # Remove duplicates
    df = remove_duplicates(df)

    # Handle outliers
    df = handle_outliers(df, method='iqr')

    # Save processed data
    save_data(df, "processed_dataset.csv")
