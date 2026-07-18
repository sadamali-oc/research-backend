import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    # Remove completely empty rows
    df = df.dropna(how="all")


    # Convert Date of Birth into Age
    df["Date of Birth"] = pd.to_datetime(
        df["Date of Birth"],
        errors="coerce"
    )

    current_year = datetime.now().year

    df["Age"] = current_year - df["Date of Birth"].dt.year

    # Remove original DOB column
    df.drop(
        columns=["Date of Birth"],
        inplace=True
    )


    # Remove Employee ID
    df.drop(
        columns=["Employee ID"],
        inplace=True
    )


    # Remove metric name columns
    metric_name_columns = [
        "Metric 1 Name",
        "Metric 2 Name",
        "Metric 3 Name",
        "Metric 4 Name",
        "Metric 5 Name",
        "Metric 6 Name",
        "Metric 7 Name"
    ]

    df.drop(
        columns=metric_name_columns,
        inplace=True
    )


    # Encode categorical columns
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns


    encoder = LabelEncoder()

    for col in categorical_columns:
        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )


    return df