import logging
import joblib
import pandas as pd
import numpy as np
import shap

from pathlib import Path


# ======================================================
# LOGGING SETUP
# ======================================================

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("employee_prediction")


# ======================================================
# CONFIGURATION
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR /
    "data" /
    "performance_future_features.xlsx"
)

MODEL_FOLDER = BASE_DIR / "models"

RESULT_FOLDER = BASE_DIR / "results"
RESULT_FOLDER.mkdir(exist_ok=True)


REGRESSOR_FILE = (
    MODEL_FOLDER /
    "random_forest_regressor.pkl"
)

CLASSIFIER_FILE = (
    MODEL_FOLDER /
    "random_forest_classifier.pkl"
)


FEATURE_FILE = (
    RESULT_FOLDER /
    "future_feature_names.pkl"
)


OUTPUT_FILE = (
    RESULT_FOLDER /
    "employee_performance_predictions.xlsx"
)



NEXT_QUARTER = {

    "Q1": "Q2",
    "Q2": "Q3",
    "Q3": "Q4",
    "Q4": "Q1"

}


BAND_MAPPING = {

    0: "Low",
    1: "Medium",
    2: "High"

}



# ======================================================
# LOAD MODELS
# ======================================================

def load_models():

    logger.info(
        "Loading models..."
    )

    regressor = joblib.load(
        REGRESSOR_FILE
    )

    classifier = joblib.load(
        CLASSIFIER_FILE
    )


    return regressor, classifier



# ======================================================
# LOAD DATA
# ======================================================

def load_data():

    logger.info(
        "Loading employee data..."
    )


    df = pd.read_excel(
        DATA_FILE
    )


    logger.info(
        "Dataset shape: %s",
        df.shape
    )


    return df



# ======================================================
# GET LATEST RECORD
# ======================================================

def get_latest_quarter_per_employee(df):

    df = df.sort_values(
        [
            "Employee ID",
            "Period Year",
            "Period Quarter"
        ]
    )


    latest = (
        df.groupby(
            "Employee ID",
            as_index=False
        )
        .tail(1)
        .copy()
    )


    return latest



# ======================================================
# FUTURE PERIOD
# ======================================================

def compute_predicted_period(row):

    year = int(
        row["Period Year"]
    )

    quarter = row["Period Quarter"]


    if quarter == "Q4":

        return f"{year + 1}-Q1"


    return (
        f"{year}-{NEXT_QUARTER[quarter]}"
    )



# ======================================================
# RECOMMENDATION GENERATION
# ======================================================

def generate_recommendation(
        factors,
        band
):

    if band == "High":

        return (
            "Maintain current performance "
            "and continue effective practices."
        )


    recommendations = {

        "Task_Completion_Rate":
            "Improve task planning and completion efficiency.",


        "On_Time_Delivery_Rate":
            "Improve deadline management and delivery consistency.",


        "Bug_Resolution_Rate":
            "Focus on faster bug identification and resolution.",


        "Code_Quality_Score":
            "Improve coding standards and perform regular code reviews.",


        "System_Reliability":
            "Reduce system issues and improve reliability.",


        "Sprint_Velocity_Raw":
            "Improve sprint productivity and workload management.",


        "Rework_Score":
            "Reduce repeated work through better quality checking."

    }


    factor_list = factors.split(", ")


    suggestions = []


    for factor in factor_list:

        if factor in recommendations:

            suggestions.append(
                recommendations[factor]
            )


    if suggestions:

        return " ".join(
            suggestions[:2]
        )


    return (
        "Monitor performance and improve "
        "identified influencing factors."
    )

# ======================================================
# SHAP TOP 5 FEATURE EXPLANATION
# ======================================================

def get_influential_factors(
        model,
        X,
        features,
        predictions,
        top_n=5
):

    logger.info(
        "Generating SHAP explanations..."
    )


    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        X
    )


    n_samples = len(X)

    n_features = len(features)



    # ==================================================
    # HANDLE DIFFERENT SHAP OUTPUT FORMATS
    # ==================================================

    if isinstance(shap_values, list):

        # Older SHAP versions
        # Shape:
        # (number_of_classes, samples, features)

        stacked = np.array(
            shap_values
        )


        per_sample_values = np.array(
            [
                stacked[
                    predictions[i],
                    i,
                    :
                ]

                for i in range(n_samples)
            ]
        )


    else:

        # New SHAP versions

        arr = np.array(
            shap_values
        )


        if arr.ndim == 3:


            # Format:
            # (samples, features, classes)

            if arr.shape[1] == n_features:


                per_sample_values = np.array(
                    [
                        arr[
                            i,
                            :,
                            predictions[i]
                        ]

                        for i in range(n_samples)
                    ]
                )


            # Format:
            # (samples, classes, features)

            elif arr.shape[2] == n_features:


                per_sample_values = np.array(
                    [
                        arr[
                            i,
                            predictions[i],
                            :
                        ]

                        for i in range(n_samples)
                    ]
                )


            else:

                raise ValueError(
                    f"Unexpected SHAP shape: {arr.shape}"
                )


        else:

            per_sample_values = arr



    logger.info(
        "SHAP values generated for %d employees",
        n_samples
    )



    # ==================================================
    # GET TOP 5 FEATURES
    # ==================================================

    top_features = []


    for i in range(n_samples):


        importance = np.abs(
            per_sample_values[i]
        )


        indexes = np.argsort(
            importance
        )[-top_n:][::-1]



        factors = [

            features[index]

            for index in indexes

        ]


        top_features.append(
            ", ".join(factors)
        )


    return top_features





# ======================================================
# MAIN PREDICTION FUNCTION
# ======================================================

def predict():


    logger.info(
        "Starting employee performance prediction..."
    )


    # Load models

    regressor, classifier = load_models()



    # Load dataset

    df = load_data()



    # Select latest employee record

    df = get_latest_quarter_per_employee(
        df
    )



    logger.info(
        "Preparing prediction features..."
    )



    features = joblib.load(
        FEATURE_FILE
    )



    X = df[
        features
    ]


    X = X.fillna(
        X.median()
    )



    # ==================================================
    # REGRESSION PREDICTION
    # ==================================================

    predicted_scores = (

        regressor.predict(
            X
        )

    )



    # ==================================================
    # CLASSIFICATION PREDICTION
    # ==================================================

    predicted_classes = (

        classifier.predict(
            X
        )

    )



    probabilities = (

        classifier.predict_proba(
            X
        )

    )


    confidence = (

        probabilities.max(
            axis=1
        )

    )



    predicted_bands = [

        BAND_MAPPING[value]

        for value in predicted_classes

    ]



    # ==================================================
    # SHAP EXPLANATION
    # ==================================================

    influential_factors = get_influential_factors(

        classifier,

        X,

        features,

        predicted_classes,

        top_n=5

    )



    recommendations = [

        generate_recommendation(
            factors,
            band
        )

        for factors, band

        in zip(
            influential_factors,
            predicted_bands
        )

    ]

   # ==================================================
    # CREATE OUTPUT
    # ==================================================

    output = pd.DataFrame({

        "employee_id":
            df["Employee ID"],

        "based_on_period":
            df["Period Year"].astype(str)
            + "-"
            + df["Period Quarter"],

        "predicted_period":
            df.apply(
                compute_predicted_period,
                axis=1
            ),

        "predicted_performance_score":
            predicted_scores.round(2),

        "predicted_performance_band":
            predicted_bands,

        "model_confidence":
            confidence.round(3),

        "top_5_influential_factors":
            influential_factors,

        "recommendation":
            recommendations

    })


    # ==================================================
    # SAVE OUTPUT EXCEL
    # ==================================================

    try:

        output.to_excel(
            OUTPUT_FILE,
            index=False
        )

        logger.info(
            "Prediction file saved: %s",
            OUTPUT_FILE
        )


    except PermissionError:

        backup_file = (
            RESULT_FOLDER /
            "employee_performance_predictions_backup.xlsx"
        )


        output.to_excel(
            backup_file,
            index=False
        )


        logger.warning(
            "File locked. Saved backup: %s",
            backup_file
        )



# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":

    predict()