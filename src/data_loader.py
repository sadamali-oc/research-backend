import pandas as pd


def load_employee_dataset(file_path):

    # Read dataset
    df = pd.read_csv(
        file_path,
        header=1
    )


    print("\n===== ORIGINAL DATASET =====")
    print(df.head())


    print("\n===== COLUMN NAMES =====")
    print(df.columns.tolist())


    # Remove completely empty rows
    df = df.dropna(how="all")


    # Reset index
    df = df.reset_index(drop=True)


    # Convert numeric columns safely
    for column in df.columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Only replace column if conversion produced valid numeric values
        if converted.notna().sum() > 0:
            df[column] = converted



    # Fill missing numeric values
    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns


    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )


    # Fill missing categorical values
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns


    df[categorical_columns] = df[categorical_columns].fillna(
        "Unknown"
    )


    print("\n===== DATASET AFTER CLEANING =====")

    print(
        "Rows:",
        df.shape[0]
    )

    print(
        "Columns:",
        df.shape[1]
    )


    print("\nMissing Values After Cleaning:")

    print(
        df.isnull().sum().sum()
    )


    return df