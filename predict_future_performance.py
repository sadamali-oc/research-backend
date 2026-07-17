import joblib
import pandas as pd

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

DATA_FILE = Path(
    "data/performance_future_features.xlsx"
)

CLASS_MODEL = Path(
    "results/best_model.pkl"
)

SCORE_MODEL = Path(
    "results/future_score_model.pkl"
)

OUTPUT_FILE = Path(
    "results/future_performance_prediction.xlsx"
)



# ==========================================
# LOAD MODELS
# ==========================================

print("\nLoading models...")


classification_model = joblib.load(
    CLASS_MODEL
)


regression_model = joblib.load(
    SCORE_MODEL
)



# ==========================================
# LOAD DATA
# ==========================================

print(
    "Loading dataset..."
)


df = pd.read_excel(
    DATA_FILE
)



# ==========================================
# PREPARE FEATURES
# ==========================================

X = df.drop(

    columns=[

        "Row_ID",

        "Future_Performance_Score",

        "Future_Performance_Category"

    ],

    errors="ignore"

)



X = X.fillna(
    X.median()
)



# ==========================================
# PREDICT FUTURE SCORE
# ==========================================

print(
    "Predicting future scores..."
)


future_scores = regression_model.predict(

    X

)



# ==========================================
# PREDICT CATEGORY
# ==========================================

categories = classification_model.predict(

    X

)


probabilities = classification_model.predict_proba(

    X

)


confidence = (

    probabilities.max(axis=1)

    * 100

)



category_mapping = {

    0:"Low",

    1:"Medium",

    2:"High"

}


future_category = [

    category_mapping[x]

    for x in categories

]



# ==========================================
# CURRENT PERFORMANCE LEVEL
# ==========================================

def get_level(score):

    if score < 50:

        return "Low"

    elif score < 75:

        return "Medium"

    else:

        return "High"



current_scores = df[

    "Current_Performance_Score"

]



current_levels = [

    get_level(score)

    for score in current_scores

]



# ==========================================
# CREATE FINAL OUTPUT
# ==========================================


result = pd.DataFrame({


    "Employee_ID":

        df["Row_ID"],



    "Current_Performance_Score":

        current_scores.round(2),



    "Current_Performance_Level":

        current_levels,



    "Predicted_Future_Score":

        future_scores.round(2),



    "Predicted_Future_Performance_Level":

        future_category,



    "Prediction_Confidence":

        confidence.round(2)

})




result.to_excel(

    OUTPUT_FILE,

    index=False

)



print(
"""
====================================
PREDICTION COMPLETED

Saved:
results/future_performance_prediction.xlsx

====================================
"""
)



print(

    result.head(10)

)