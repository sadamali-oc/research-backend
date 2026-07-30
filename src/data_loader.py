import pandas as pd



def load_employee_dataset(file_path):


    # ==============================
    # READ DATASET
    # ==============================

    df = pd.read_csv(
        file_path,
        header=1
    )



    print("\n===== ORIGINAL DATASET =====")

    print(
        df.head()
    )



    print("\n===== COLUMN NAMES =====")

    print(
        df.columns.tolist()
    )



    # ==============================
    # REMOVE EMPTY ROWS
    # ==============================

    df = df.dropna(
        how="all"
    )


    df = df.reset_index(
        drop=True
    )



    # ==============================
    # REMOVE DUPLICATE HEADER ROWS
    # ==============================

    if "Gender" in df.columns:

        df = df[
            df["Gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            != "gender"
        ]



    if "Ethnicity" in df.columns:

        df = df[
            df["Ethnicity"]
            .astype(str)
            .str.strip()
            .str.lower()
            != "ethnicity"
        ]



    # ==============================
    # REMOVE INVALID DEMOGRAPHIC ROWS
    # ==============================

    if "Gender" in df.columns:

        df = df[
            df["Gender"].notna()
        ]

        df = df[
            df["Gender"]
            .astype(str)
            .str.strip()
            != ""
        ]



    if "Ethnicity" in df.columns:

        df = df[
            df["Ethnicity"].notna()
        ]

        df = df[
            df["Ethnicity"]
            .astype(str)
            .str.strip()
            != ""
        ]



    # ==============================
    # REMOVE UNKNOWN DEMOGRAPHIC VALUES
    # ==============================

    if "Gender" in df.columns:

        df = df[
            ~df["Gender"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "unknown",
                    "nan",
                    "none"
                ]
            )
        ]



    if "Ethnicity" in df.columns:

        df = df[
            ~df["Ethnicity"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "unknown",
                    "nan",
                    "none"
                ]
            )
        ]



    df = df.reset_index(
        drop=True
    )



    # ==============================
    # CONVERT NUMERIC COLUMNS
    # ==============================

    for column in df.columns:


        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )


        if converted.notna().sum() > 0:

            df[column] = converted



    # ==============================
    # HANDLE MISSING VALUES
    # ==============================


    numeric_columns = df.select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    ).columns



    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(
            df[numeric_columns]
            .median()
        )
    )



    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "string"
        ]
    ).columns



    df[categorical_columns] = (
        df[categorical_columns]
        .fillna(
            "Unknown"
        )
    )



    # ==============================
    # DISPLAY CLEANING RESULTS
    # ==============================

    print(
        "\n===== DATASET AFTER CLEANING ====="
    )


    print(
        "Rows:",
        df.shape[0]
    )


    print(
        "Columns:",
        df.shape[1]
    )



    print(
        "\nMissing Values:"
    )


    print(
        df.isnull().sum().sum()
    )



    if "Gender" in df.columns:

        print(
            "\nGender Distribution:"
        )


        print(
            df["Gender"]
            .value_counts()
        )



    if "Ethnicity" in df.columns:

        print(
            "\nEthnicity Distribution:"
        )


        print(
            df["Ethnicity"]
            .value_counts()
        )



    return df