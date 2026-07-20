import joblib
import pandas as pd
import numpy as np
import shap

from pathlib import Path


# ======================================================
# CONFIGURATION
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_FILE = (
    BASE_DIR /
    "data" /
    "performance_future_features.xlsx"
)


MODEL_FOLDER = (
    BASE_DIR /
    "models"
)


RESULT_FOLDER = (
    BASE_DIR /
    "results"
)

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



# ======================================================
# LOAD MODELS
# ======================================================

def load_models():

    print("\nLoading models...")


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

    print("\nLoading employee data...")


    df = pd.read_excel(
        DATA_FILE
    )


    print(
        "Dataset:",
        df.shape
    )


    return df




# ======================================================
# GET LATEST QUARTER
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


    quarter = (
        row["Period Quarter"]
    )


    if quarter == "Q4":

        return (
            f"{year+1}-Q1"
        )


    return (

        f"{year}-"
        f"{NEXT_QUARTER[quarter]}"

    )




# ======================================================
# RECOMMENDATIONS
# ======================================================

def generate_recommendation(
        factor,
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



    return recommendations.get(

        factor,

        "Monitor performance and improve identified areas."

    )





# ======================================================
# SHAP EXPLANATION
# ======================================================

def get_influential_factors(
        model,
        X,
        features,
        predictions
):


    print("\nGenerating explanations...")


    explainer = shap.TreeExplainer(
        model
    )


    shap_values = (
        explainer.shap_values(X)
    )


    print(
        "SHAP generated"
    )


    factors = []



    # Multi-class Random Forest handling

    if isinstance(
        shap_values,
        list
    ):


        shap_values = np.array(
            shap_values
        )


        # Select predicted class explanation

        final_values = []


        for i in range(len(X)):


            class_index = predictions[i]


            values = shap_values[

                class_index,

                i

            ]


            final_values.append(
                values
            )


        shap_values = np.array(
            final_values
        )



    else:


        shap_values = np.array(
            shap_values
        )



    # Remove extra dimension

    if len(shap_values.shape) == 3:

        shap_values = shap_values[:,:,0]




    for i in range(len(X)):


        values = np.abs(
            shap_values[i]
        )


        index = np.argmax(
            values
        )


        if index < len(features):

            factors.append(
                features[index]
            )

        else:

            factors.append(
                "Unknown"
            )


    return factors





# ======================================================
# MAIN PREDICTION
# ======================================================

def predict():


    regressor, classifier = load_models()


    df = load_data()


    df = get_latest_quarter_per_employee(
        df
    )


    print(
        "\nPreparing features..."
    )


    features = joblib.load(
        FEATURE_FILE
    )


    X = df[features]



    # Regression prediction

    predicted_scores = (

        regressor.predict(X)

    )



    # Classification prediction

    predicted_classes = (

        classifier.predict(X)

    )



    probabilities = (

        classifier.predict_proba(X)

    )


    confidence = (

        probabilities.max(axis=1)

    )



    band_mapping = {

        0:"Low",
        1:"Medium",
        2:"High"

    }



    predicted_bands = [

        band_mapping[x]

        for x in predicted_classes

    ]



    # SHAP explanation

    influential_factors = get_influential_factors(

        classifier,

        X,

        features,

        predicted_classes

    )



    recommendations = [

        generate_recommendation(

            factor,

            band

        )

        for factor, band

        in zip(
            influential_factors,
            predicted_bands
        )

    ]



    # Final output


    output = pd.DataFrame({


        "employee_id":

        df["Employee ID"],



        "based_on_period":

        df["Period Year"].astype(str)

        +

        "-"

        +

        df["Period Quarter"],



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



        "most_influential_factor":

        influential_factors,



        "recommendation":

        recommendations

    })




    output.to_excel(

        OUTPUT_FILE,

        index=False

    )



    print(

        "\nPrediction completed!"

    )


    print(

        "Rows:",

        len(output)

    )


    print(

        "Saved:",

        OUTPUT_FILE

    )





if __name__ == "__main__":

    predict()