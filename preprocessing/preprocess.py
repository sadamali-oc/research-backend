# ======================================================
# EMPLOYEE FUTURE PERFORMANCE PREPROCESSING
# Random Forest Score + Band Prediction
# ======================================================

import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import LabelEncoder


# ======================================================
# CONFIGURATION
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR /
    "data" /
    "dataset.xlsx"
)


OUTPUT_FILE = BASE_DIR / "data" / "performance_future_features.xlsx"

RESULT_FOLDER = BASE_DIR / "results"

RESULT_FOLDER.mkdir(exist_ok=True)


ENCODER_FILE = RESULT_FOLDER / "future_encoders.pkl"

FEATURE_FILE = RESULT_FOLDER / "future_feature_names.pkl"



# ======================================================
# LOAD DATA
# ======================================================

def load_dataset():

    print("\nLoading dataset...")

    df = pd.read_excel(INPUT_FILE)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    print(
        "Dataset Shape:",
        df.shape
    )

    return df



# ======================================================
# CREATE FUTURE TARGET
# ======================================================

def create_future_target(df):

    print("\nCreating future target...")


    quarter_mapping = {

        "Q1":1,
        "Q2":2,
        "Q3":3,
        "Q4":4

    }


    df["Quarter_Number"] = (
        df["Period Quarter"]
        .map(quarter_mapping)
    )


    df = df.sort_values(

        [
            "Employee ID",
            "Period Year",
            "Quarter_Number"
        ]

    )


    grouped = df.groupby(
        "Employee ID"
    )


    # Next quarter performance score

    df["Future_Performance_Score"] = (
        grouped[
            "Performance Score (Current Period)"
        ]
        .shift(-1)
    )


    # Next quarter performance band

    df["Future_Performance_Band"] = (
        grouped[
            "Performance Band (Current Period)"
        ]
        .shift(-1)
    )


    # Ensure actual next quarter exists

    df["Next_Year"] = (
        grouped["Period Year"]
        .shift(-1)
    )


    df["Next_Quarter"] = (
        grouped["Quarter_Number"]
        .shift(-1)
    )


    quarter_difference = (

        (df["Next_Year"] - df["Period Year"]) * 4

        +

        (df["Next_Quarter"] - df["Quarter_Number"])

    )


    df = df[
        quarter_difference == 1
    ]


    df.dropna(

        subset=[
            "Future_Performance_Score"
        ],

        inplace=True

    )


    return df



# ======================================================
# HISTORICAL FEATURES
# ======================================================

def create_history_features(df):

    print("\nCreating history features...")


    grouped = df.groupby(
        "Employee ID"
    )


    df["Current_Performance_Score"] = (
        df[
            "Performance Score (Current Period)"
        ]
    )


    df["Previous_Performance_Score"] = (

        grouped[
            "Performance Score (Current Period)"
        ]
        .shift(1)

    )


    df["Historical_Average_Score"] = (

        grouped[
            "Performance Score (Current Period)"
        ]
        .transform(

            lambda x:

            x.shift(1)
            .expanding()
            .mean()

        )

    )


    df["Performance_Trend"] = (

        df["Current_Performance_Score"]

        -

        df["Previous_Performance_Score"]

    )


    df.dropna(

        subset=[
            "Previous_Performance_Score"
        ],

        inplace=True

    )


    return df



# ======================================================
# KPI EXTRACTION
# ======================================================

def extract_metric(row, metric):

    for i in range(1,8):

        if str(
            row[f"Metric {i} Name"]
        ).strip() == metric:

            return row[
                f"Metric {i} Value"
            ]

    return np.nan



def extract_kpis(df):

    print("\nExtracting KPIs...")


    metrics = [

        "tasks_assigned",
        "tasks_completed",
        "tasks_on_time",
        "bug_count",
        "bugs_fixed_count",
        "code_quality (1-10)",
        "rework_count",
        "total_test_cases_executed",
        "total_pass_test_cases",
        "total_deployments",
        "successful_deployments",
        "automated_pipeline_stages",
        "total_pipeline_stages",
        "system_downtime (hrs)",
        "sprint_velocity"

    ]


    for metric in metrics:


        df[metric] = df.apply(

            lambda row:

            extract_metric(
                row,
                metric
            ),

            axis=1

        )


        # Missing tracking flag

        df[f"{metric}_tracked"] = (

            df[metric]
            .notna()
            .astype(int)

        )


    return df



# ======================================================
# KPI FEATURE CREATION
# ======================================================

def percentage(a,b):

    return np.where(

        b.isna() | (b <= 0),

        np.nan,

        (a/b)*100

    )



def create_kpi_features(df):

    print("\nCreating KPI features...")


    df["Task_Completion_Rate"] = percentage(

        df.tasks_completed,

        df.tasks_assigned

    )


    df["On_Time_Delivery_Rate"] = percentage(

        df.tasks_on_time,

        df.tasks_completed

    )


    df["Bug_Resolution_Rate"] = percentage(

        df.bugs_fixed_count,

        df.bug_count

    )


    df["Code_Quality_Score"] = (

        df["code_quality (1-10)"]

        *10

    )


    df["Rework_Score"] = (

        100 /

        (1 + df.rework_count.fillna(0))

    )


    df["Test_Pass_Rate"] = percentage(

        df.total_pass_test_cases,

        df.total_test_cases_executed

    )


    df["Deployment_Success_Rate"] = percentage(

        df.successful_deployments,

        df.total_deployments

    )


    df["System_Reliability"] = (

        100 -

        df["system_downtime (hrs)"]*2

    )


    # Keep raw value to avoid leakage

    df["Sprint_Velocity_Raw"] = (

        df.sprint_velocity

    )


    return df



# ======================================================
# FINAL DATASET
# ======================================================

def prepare_final_dataset(df):

    print("\nPreparing final dataset...")


    features = [

        "Job Role",
        "Years of Experience",
        "Department",

        "Duration (Weeks)",
        "Relative Effort (Story Pts)",
        "Team Size",

        "Project Complexity",

        "Rework Count",
        "No-Pay Leave",
        "Blockers",


        "Task_Completion_Rate",
        "On_Time_Delivery_Rate",
        "Bug_Resolution_Rate",

        "Code_Quality_Score",
        "Rework_Score",

        "Test_Pass_Rate",

        "Deployment_Success_Rate",

        "System_Reliability",

        "Sprint_Velocity_Raw",


        "Current_Performance_Score",
        "Previous_Performance_Score",
        "Historical_Average_Score",
        "Performance_Trend"

    ]


    features += [

        c for c in df.columns

        if c.endswith("_tracked")

    ]



    final = df[

        [

            "Employee ID",
            "Period Year",
            "Period Quarter"

        ]

        +

        features

        +

        [

            "Future_Performance_Score",
            "Future_Performance_Band"

        ]

    ].copy()



    encoders = {}


    for col in [

        "Job Role",
        "Department"

    ]:


        encoder = LabelEncoder()


        final[col] = encoder.fit_transform(

            final[col].astype(str)

        )


        encoders[col] = encoder



    final["Future_Performance_Category"] = (

        final["Future_Performance_Band"]

        .map({

            "Low":0,
            "Medium":1,
            "High":2

        })

    )


    final.drop(

        columns=[

            "Future_Performance_Band"

        ],

        inplace=True

    )


    final.insert(

        0,

        "Row_ID",

        range(len(final))

    )


    joblib.dump(

        encoders,

        ENCODER_FILE

    )


    joblib.dump(

        [

            c for c in final.columns

            if c not in [

                "Row_ID",
                "Employee ID",
                "Period Year",
                "Period Quarter",
                "Future_Performance_Score",
                "Future_Performance_Category"

            ]

        ],

        FEATURE_FILE

    )


    final.to_excel(

        OUTPUT_FILE,

        index=False

    )


    return final



# ======================================================
# MAIN
# ======================================================

def main():

    print(
"""
====================================
EMPLOYEE FUTURE PERFORMANCE
PREPROCESSING
====================================
"""
    )


    df = load_dataset()

    df = create_future_target(df)

    df = create_history_features(df)

    df = extract_kpis(df)

    df = create_kpi_features(df)

    final = prepare_final_dataset(df)



    print(
        "\nFinal Dataset:",
        final.shape
    )


    print(
        "\nCategory Distribution:"
    )


    print(
        final[
            "Future_Performance_Category"
        ]
        .value_counts()
    )


    print(
        "\nSaved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()