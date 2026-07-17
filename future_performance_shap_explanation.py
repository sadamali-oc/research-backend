import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

DATA_FILE = Path(
    "data/performance_future_features.xlsx"
)


REGRESSION_MODEL = Path(
    "results/future_score_model.pkl"
)


OUTPUT_FOLDER = Path(
    "results/shap"
)

OUTPUT_FOLDER.mkdir(
    exist_ok=True
)



# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")


model = joblib.load(

    REGRESSION_MODEL

)



# ==========================================
# LOAD DATA
# ==========================================


df = pd.read_excel(

    DATA_FILE

)



X = df.drop(

    columns=[

        "Row_ID",

        "Future_Performance_Score",

        "Future_Performance_Category"

    ]

)



X = X.fillna(

    X.median()

)



print(

"Records:",

len(X)

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
# GLOBAL IMPORTANCE
# ==========================================


plt.figure(

    figsize=(10,8)

)


shap.summary_plot(

    shap_values,

    X,

    show=False

)


plt.tight_layout()


plt.savefig(

    OUTPUT_FOLDER /

    "future_score_feature_importance.png",

    dpi=300

)


plt.close()



# ==========================================
# SAVE SHAP VALUES
# ==========================================


shap_df = pd.DataFrame(

    shap_values,

    columns=X.columns

)



shap_df.to_excel(

    OUTPUT_FOLDER /

    "future_score_shap_values.xlsx",

    index=False

)



# ==========================================
# SINGLE EMPLOYEE EXPLANATION
# ==========================================


employee_index = 0


explanation = pd.DataFrame({

    "Feature":

        X.columns,


    "Impact":

        shap_values[employee_index]

})



explanation["Absolute_Impact"] = (

    explanation["Impact"]
    .abs()

)



explanation = explanation.sort_values(

    "Absolute_Impact",

    ascending=False

)



explanation.to_excel(

    OUTPUT_FOLDER /

    "employee_0_explanation.xlsx",

    index=False

)



print(
"""
====================================
SHAP COMPLETED

Generated:

future_score_feature_importance.png

future_score_shap_values.xlsx

employee_0_explanation.xlsx

====================================
"""
)