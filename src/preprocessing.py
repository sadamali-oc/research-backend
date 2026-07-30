import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    # Remove completely empty rows
    df = df.dropna(how="all")


    # Convert Date of Birth into Age

    if "Date of Birth" in df.columns:

        df["Date of Birth"] = pd.to_datetime(
            df["Date of Birth"],
            errors="coerce"
        )

        current_year = datetime.now().year

        df["Age"] = (
            current_year -
            df["Date of Birth"].dt.year
        )


        df.drop(
            columns=["Date of Birth"],
            inplace=True
        )


    # Remove Employee ID

    if "Employee ID" in df.columns:

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


    existing_columns = [
        col for col in metric_name_columns
        if col in df.columns
    ]


    df.drop(
        columns=existing_columns,
        inplace=True
    )



    # Remove accidental header rows

    if "Gender" in df.columns:

        df = df[
            df["Gender"].astype(str).str.lower()
            != "gender"
        ]


    if "Ethnicity" in df.columns:

        df = df[
            df["Ethnicity"].astype(str).str.lower()
            != "ethnicity"
        ]



    # Encode categorical columns except fairness attributes

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns


    fairness_columns = [
        "Gender",
        "Ethnicity"
    ]


    encoding_columns = [
        col for col in categorical_columns
        if col not in fairness_columns
    ]



    for col in encoding_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )



    return df