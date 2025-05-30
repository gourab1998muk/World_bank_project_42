# data_sanity_check.py
import pandas as pd
import re
print("updated ")
def clean_string(s):
    """Clean a string by removing extra spaces, stripping, and converting to lowercase."""
    if pd.isnull(s):  # Handle NaN values
        return ""
    return re.sub(r'\s+', ' ', s.strip()).lower()

def clean_columns(df, column_name, new_column_name):
    """Apply cleaning to a specified column and store in a new column."""
    df[new_column_name] = df[column_name].apply(clean_string)
    return df

def check_indicator_match(df1, col1, df2, col2):
    """Check if all values in df1[col1] match those in df2[col2] after cleaning."""
    wide_format_values = set(df1[col1])
    indicator_values = set(df2[col2])
    all_match = indicator_values.issubset(wide_format_values)
    
    if all_match:
        return True, None
    else:
        unmatched_values = indicator_values - wide_format_values
        return False, unmatched_values

def set_categorical_order(df, column, categories, sort_columns=None):
    """Set a categorical order for a column and optionally sort the DataFrame."""
    df[column] = pd.Categorical(df[column], categories=categories, ordered=True)
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df

def compare_columns(df, col1, col2):
    """Compare two columns element-wise and return unmatched rows."""
    are_equal = (df[col1] == df[col2]).all()
    if are_equal:
        return True, None
    else:
        unmatched = df[df[col1] != df[col2]]
        return False, unmatched

def combine_datasets(df1, df2, suffix1="_1", suffix2="_2"):
    """Combine two DataFrames side by side and append suffixes to column names."""
    combined_df = pd.concat([df1, df2], axis=1)
    new_col_names = []
    for i, col in enumerate(combined_df.columns):
        if i < len(df1.columns):
            new_col_names.append(f"{col}{suffix1}")
        else:
            new_col_names.append(f"{col}{suffix2}")
    combined_df.columns = new_col_names
    return combined_df

def clean_column_names(df):
    """Remove numerical suffixes from column names (e.g., '_1', '_2')."""
    new_col_names = []
    for col in df.columns:
        if '_' in col and col.rsplit('_', 1)[1].isdigit():
            new_col_names.append(col.rsplit('_', 1)[0])
        else:
            new_col_names.append(col)
    df.columns = new_col_names
    return df

def pivot_to_wide(df, index_cols, pivot_col, value_col):
    """Pivot a DataFrame from long to wide format."""
    return df.pivot(index=index_cols, columns=pivot_col, values=value_col).reset_index()

