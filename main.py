from pathlib import Path

from src.data_loader import load_employee_dataset
from src.validator import validate_dataset
from src.preprocessing import preprocess_data
from src.performance_score import generate_performance_score
from src.model_training import train_performance_model

from src.bias_detection import (
    analyze_gender_bias,
    analyze_ethnicity_bias,
    demographic_parity
)

from src.fairness_metrics import calculate_gender_fairness


DATASET_PATH = Path(
    "dataset/employee_dataset.csv"
)


def main():

    # ==============================
    # LOAD DATASET
    # ==============================

    df = load_employee_dataset(DATASET_PATH)

    print("\n===== ORIGINAL DATASET =====")
    print(df.head())

    print("\n===== COLUMN NAMES =====")
    print(df.columns.tolist())


    # ==============================
    # VALIDATION
    # ==============================

    validate_dataset(df)


    # ==============================
    # PREPROCESSING
    # ==============================

    df = preprocess_data(df)

    print("\n===== PREPROCESSED DATASET =====")
    print(df.head())

    print("\n===== DATA TYPES AFTER PREPROCESSING =====")
    print(df.dtypes)



    # ==============================
    # PERFORMANCE SCORE GENERATION
    # ==============================

    df = generate_performance_score(df)


    print("\n===== FINAL DATASET WITH PERFORMANCE SCORE =====")

    print(
        df[
            [
                "Performance Score",
                "Performance Category"
            ]
        ].head(10)
    )


    print("\n===== SCORE STATISTICS =====")

    print(
        df["Performance Score"].describe()
    )


    print("\n===== PERFORMANCE CATEGORY DISTRIBUTION =====")

    print(
        df["Performance Category"].value_counts()
    )



    # ==============================
    # SAVE PROCESSED DATASET
    # ==============================

    output_path = Path(
        "dataset/processed_employee_performance.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nProcessed dataset saved at: {output_path}"
    )



    # ==============================
    # MODEL TRAINING
    # ==============================

    train_performance_model(df)



    # ==============================
    # BIAS DETECTION
    # ==============================

    print("\n===== BIAS DETECTION RESULTS =====")


    analyze_gender_bias(df)


    analyze_ethnicity_bias(df)


    demographic_parity(df)



    # ==============================
    # FAIRNESS ANALYSIS
    # ==============================

    print("\n===== FAIRNESS METRICS =====")

    calculate_gender_fairness(df)



    print(
        "\n===== PIPELINE COMPLETED SUCCESSFULLY ====="
    )



if __name__ == "__main__":
    main()