import pandas as pd
import joblib
import shap

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_FILE = Path(
    "results/future_score_model.pkl"
)

DATA_FILE = Path(
    "data/performance_future_features.xlsx"
)

OUTPUT_FILE = Path(
    "results/employee_50_shap_explanation.xlsx"
)


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading Random Forest model...")

model = joblib.load(
    MODEL_FILE
)

print(
    "Model:",
    type(model).__name__
)



# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_excel(
    DATA_FILE
)


# Select only 50 employees

df = df.head(50)



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
# SHAP EXPLANATION
# ==========================================

print(
    "\nCalculating SHAP values..."
)


explainer = shap.TreeExplainer(
    model
)


shap_values = explainer.shap_values(
    X
)



# ==========================================
# CREATE EMPLOYEE REPORT
# ==========================================

results = []


predictions = model.predict(X)



for i in range(len(X)):


    employee_shap = pd.DataFrame({

        "Feature":
            X.columns,

        "Impact":
            shap_values[i]

    })


    positive = employee_shap.sort_values(

        "Impact",

        ascending=False

    ).head(3)



    negative = employee_shap.sort_values(

        "Impact",

        ascending=True

    ).head(3)



    results.append({

        "Employee_ID":
            df.iloc[i]["Row_ID"],


        "Current_Performance_Score":
            df.iloc[i]["Previous_Performance_Score"],


        "Predicted_Future_Score":
            round(predictions[i],2),


        "Positive_Factors":
            ", ".join(
                positive["Feature"]
            ),


        "Negative_Factors":
            ", ".join(
                negative["Feature"]
            )

    })



# ==========================================
# SAVE EXCEL
# ==========================================

result_df = pd.DataFrame(
    results
)


result_df.to_excel(

    OUTPUT_FILE,

    index=False

)


print(
"""
====================================
SHAP COMPLETED

50 Employee Explanation Created:

results/employee_50_shap_explanation.xlsx

====================================
"""
)