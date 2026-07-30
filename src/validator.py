import pandas as pd


def validate_dataset(df: pd.DataFrame):
    print("\n===== DATASET VALIDATION =====")

    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nData Types:")
    print(df.dtypes)