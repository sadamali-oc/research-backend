from pathlib import Path
import json

from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_employee_dataset
from src.validator import validate_dataset
from src.preprocessing import preprocess_data
from src.performance_score import generate_performance_score
from src.model_training import train_performance_model

from src.fairness_engine import run_fairness_analysis



DATASET_PATH = Path(
    "dataset/employee_dataset.csv"
)



def encode_for_ml(df):

    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns


    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )


    return df





def main():


    # ==============================
    # LOAD DATASET
    # ==============================

    df = load_employee_dataset(
        DATASET_PATH
    )


    print("\n===== ORIGINAL DATASET =====")
    print(df.head())



    # ==============================
    # VALIDATION
    # ==============================

    validate_dataset(df)



    # ==============================
    # CREATE FAIRNESS DATA COPY
    # ==============================

    fairness_data = df.copy()



    print("\n===== FAIRNESS DATA SAMPLE =====")

    print(
        fairness_data[
            [
                "Gender",
                "Ethnicity"
            ]
        ].head()
    )



    # ==============================
    # PREPROCESS FOR ML
    # ==============================

    processed_data = preprocess_data(
        df.copy()
    )


    print("\n===== PREPROCESSED DATA =====")

    print(
        processed_data.head()
    )



    # ==============================
    # GENERATE PERFORMANCE SCORE
    # ==============================

    processed_data = generate_performance_score(
        processed_data
    )



    print("\n===== PERFORMANCE SCORE =====")

    print(
        processed_data[
            [
                "Performance Score",
                "Performance Category"
            ]
        ].head()
    )



    # ==============================
    # CREATE FAIRNESS SCORE DATA
    # ==============================

    fairness_processed = preprocess_data(
        fairness_data.copy()
    )


    fairness_processed = generate_performance_score(
        fairness_processed
    )



    # Restore original demographic values

    fairness_processed["Gender"] = (
        fairness_data["Gender"]
        .values
    )


    fairness_processed["Ethnicity"] = (
        fairness_data["Ethnicity"]
        .values
    )



    print("\n===== FAIRNESS READY DATA =====")

    print(
        fairness_processed[
            [
                "Gender",
                "Ethnicity",
                "Performance Score"
            ]
        ].head()
    )



    # ==============================
    # ENCODE ML DATA
    # ==============================

    ml_data = encode_for_ml(
        processed_data
    )



    # ==============================
    # SAVE PROCESSED DATASET
    # ==============================

    ml_data.to_csv(
        "dataset/processed_employee_performance.csv",
        index=False
    )



    # ==============================
    # MODEL TRAINING
    # ==============================

    train_performance_model(
        ml_data
    )



    # ==============================
    # FAIRNESS ANALYSIS
    # ==============================

    print("\n===== FAIRNESS ANALYSIS =====")



    fairness_results = run_fairness_analysis(
        fairness_processed
    )


    print(
        fairness_results
    )



    # ==============================
    # SAVE REPORT
    # ==============================

    Path(
        "outputs"
    ).mkdir(
        exist_ok=True
    )


    with open(
        "outputs/fairness_report.json",
        "w"
    ) as file:

        json.dump(
            fairness_results,
            file,
            indent=4
        )



    print(
        "\n===== PIPELINE COMPLETED SUCCESSFULLY ====="
    )



if __name__ == "__main__":

    main()