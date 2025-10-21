"""
Feature Encoding Script
========================
Performs intelligent encoding on categorical features from the standardized dataset.

Strategy:
- One-hot encoding for low-cardinality features (<10 unique values)
- Label encoding for high-cardinality features (≥10 unique values)
- Special handling for datetime and boolean columns

Note: By just doing one-hot encoding created ~4000 columns, so I used a hybrid approach
      to maintain dimensionality while preserving feature information.
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


def load_data(input_path):
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


def analyze_categorical_columns(df, special_cols, cardinality_threshold=10):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print("=" * 70)
    print("CATEGORICAL COLUMNS ANALYSIS")
    print("=" * 70)
    print(f"Found {len(categorical_cols)} categorical columns:\n")
    
    low_cardinality = []
    high_cardinality = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        
        if col in special_cols:
            print(f"  {col:30s} {unique_count:5d} unique  [SPECIAL]")
        elif unique_count < cardinality_threshold:
            low_cardinality.append(col)
            print(f"  {col:30s} {unique_count:5d} unique  [ONE-HOT]")
        else:
            high_cardinality.append(col)
            print(f"  {col:30s} {unique_count:5d} unique  [LABEL]")
    
    print()
    return low_cardinality, high_cardinality


def handle_datetime_column(df, col_name='day'):
    if col_name not in df.columns:
        return df
    
    print(f"Converting '{col_name}' to datetime features...")
    df[col_name] = pd.to_datetime(df[col_name])
    df['day_of_week'] = df[col_name].dt.dayofweek
    df['day_of_year'] = df[col_name].dt.dayofyear
    df['month'] = df[col_name].dt.month
    df['year'] = df[col_name].dt.year
    df = df.drop(col_name, axis=1)
    print("Created: day_of_week, day_of_year, month, year\n")
    return df

#Convert boolean column to binary (0/1)
def handle_binary_column(df, col_name='train_mask'):
    if col_name not in df.columns:
        return df
    
    print(f"Converting '{col_name}' to binary...")
    null_count = df[col_name].isnull().sum()
    
    if null_count > 0:
        print(f"  ⚠ Found {null_count} NaN values, filling with False")
        df[col_name] = df[col_name].fillna(False).infer_objects(copy=False)
    
    df[col_name] = df[col_name].astype(int)
    print("Converted to 0/1\n")
    return df

# Lable encoding for high-cardinality columns 
def label_encode_columns(df, columns):
    if not columns:
        print("No high-cardinality columns to encode\n")
        return df
    
    print("=" * 70)
    print("LABEL ENCODING")
    print("=" * 70)
    
    for col in columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        df = df.drop(col, axis=1)
        print(f" {col:30s} → {col}_encoded")
    
    print(f"\nLabel encoded {len(columns)} columns\n")
    return df


def one_hot_encode_columns(df, columns):
    if not columns:
        print("No low-cardinality columns to encode\n")
        return df
    
    print("=" * 70)
    print("ONE-HOT ENCODING")
    print("=" * 70)
    print(f"Encoding {len(columns)} columns...")
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    print(f"  Complete\n")
    return df

def validate_encoding(df):
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    remaining = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if remaining:
        print(f"Warning: {len(remaining)} categorical columns remain:")
        for col in remaining:
            print(f"  - {col}")
    else:
        print("All categorical columns encoded successfully")
    
    print()


def save_dataset(df, output_path):
    print("=" * 70)
    print("SAVING DATASET")
    print("=" * 70)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}\n")


def print_summary(original_cols, final_cols, special_count, label_count, onehot_count):
    print("=" * 70)
    print("ENCODING SUMMARY")
    print("=" * 70)
    print(f"Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Original cols:  {original_cols}")
    print(f"Final cols:     {final_cols}")
    print(f"Expansion:      {final_cols - original_cols:+d} columns")
    print(f"\nBreakdown:")
    print(f"  • Special handling: {special_count} columns")
    print(f"  • Label encoded:    {label_count} columns")
    print(f"  • One-hot encoded:  {onehot_count} columns")
    print("\n✓ Feature encoding completed successfully!")


def main():
    # Setup paths
    script_dir = os.path.dirname(__file__)
    input_path = os.path.join(script_dir, "../../data/ml_ready/merged_weather_outages_2019_2024_standardized.csv")
    input_path = os.path.normpath(input_path)
    
    output_dir = os.path.join(script_dir, "../../data/ml_ready")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'merged_weather_outages_2019_2024_encoded.csv')
    
    # Load data
    df = load_data(input_path)
    original_cols = df.shape[1]
    
    # Analyze columns
    special_cols = ['day', 'train_mask']
    low_card_cols, high_card_cols = analyze_categorical_columns(df, special_cols)
    
    # Handle special columns
    print("=" * 70)
    print("SPECIAL COLUMN HANDLING")
    print("=" * 70)
    df = handle_datetime_column(df, 'day')
    df = handle_binary_column(df, 'train_mask')
    
    # Apply encoding
    df = label_encode_columns(df, high_card_cols)
    df = one_hot_encode_columns(df, low_card_cols)
    
    # Validate and save
    validate_encoding(df)
    save_dataset(df, output_path)
    
    # Print summary
    print_summary(original_cols, df.shape[1], len(special_cols), len(high_card_cols), len(low_card_cols))

if __name__ == "__main__":
    main()