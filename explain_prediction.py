import pandas as pd
import numpy as np
import joblib
import shap
import os

import matplotlib.pyplot as plt
import seaborn as sns


from reportlab.lib.pagesizes import landscape, A4

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors



print("\n====================================")
print("EMPLOYEE PERFORMANCE EXPLANATION SYSTEM")
print("====================================\n")



# =====================================================
# 1. LOAD MODEL
# =====================================================

model = joblib.load(
    "results/best_model.pkl"
)

print("Model loaded")



# =====================================================
# 2. LOAD DATA
# =====================================================

df = pd.read_excel(
    "data/performance_features.xlsx"
)


print("Dataset loaded")
print("Shape:", df.shape)



target = "Performance_Category"


employee_ids = df["Employee ID"]



X = df.drop(
    columns=[
        target,
        "Employee ID",
        "Row_ID"
    ],
    errors="ignore"
)



feature_names = X.columns.tolist()



print("\nFeatures:")
print(feature_names)



# =====================================================
# 3. SELECT EMPLOYEES
# =====================================================

selected_indices = []


for category in [0,1,2]:

    selected = df[
        df[target] == category
    ].head(5).index


    selected_indices.extend(
        selected
    )



X_selected = X.loc[
    selected_indices
]



print(
    "\nSelected employees:",
    len(X_selected)
)



# =====================================================
# 4. SHAP EXPLAINER
# =====================================================

print("\nCreating SHAP Explainer...")


explainer = shap.Explainer(
    model.predict_proba,
    X.values[:100]
)


print("SHAP Ready")



# =====================================================
# 5. GENERATE SHAP EXPLANATIONS
# =====================================================


mapping = {

    0:"Low Performance",
    1:"Medium Performance",
    2:"High Performance"

}



reports = []



for index,row in X_selected.iterrows():


    employee_id = employee_ids.loc[index]


    employee = row.to_frame().T



    prediction = model.predict(
        employee.values
    )


    probability = model.predict_proba(
        employee.values
    )



    predicted_class = int(
        prediction[0]
    )



    confidence = (

        np.max(probability)
        *
        100

    )



    shap_values = explainer(
        employee.values
    )



    values = np.array(
        shap_values.values
    )



    if len(values.shape) == 3:

        shap_value = values[
            0,
            :,
            predicted_class
        ]

    else:

        shap_value = values[0]



    importance = pd.DataFrame({

        "Feature": feature_names,

        "SHAP_Value": shap_value,

        "Impact": np.abs(shap_value)

    })



    importance = importance.sort_values(

        "Impact",

        ascending=False

    )



    top5 = importance.head(5)



    rank = 1



    for _,item in top5.iterrows():


        reports.append({

            "Employee ID":
            employee_id,


            "Prediction":
            mapping[predicted_class],


            "Confidence (%)":
            round(confidence,2),


            "Rank":
            rank,


            "Influential Factor":
            item["Feature"],


            "SHAP Impact":
            round(
                item["Impact"],
                4
            ),


            "Direction":

            "Positive"

            if item["SHAP_Value"] > 0

            else

            "Negative"

        })


        rank += 1



# =====================================================
# 6. SAVE EMPLOYEE REPORT
# =====================================================


os.makedirs(
    "results",
    exist_ok=True
)



report_df = pd.DataFrame(
    reports
)



report_df.to_excel(

    "results/employee_individual_factors.xlsx",

    index=False

)



print(
    "Employee explanation Excel saved"
)

# =====================================================
# 7. ROLE-WISE MOST INFLUENTIAL FACTORS
# =====================================================


print("\nGenerating role-wise analysis...")


original_df = pd.read_excel(
    "data/dataset.xlsx"
)



print("\nDataset columns:")
print(original_df.columns.tolist())



# Change this if your column name is different
role_column = "Job Role"



chart_files = []



if role_column in original_df.columns:


    # ---------------------------------------------
    # Merge role information
    # ---------------------------------------------


    merged = report_df.merge(

        original_df[
            [
                "Employee ID",
                role_column
            ]
        ],

        on="Employee ID",

        how="left"

    )



    print("\nRole mapping preview:")
    print(
        merged[
            [
                "Employee ID",
                role_column
            ]
        ].head()
    )



    # Remove employees without roles

    merged = merged.dropna(
        subset=[role_column]
    )



    if len(merged) > 0:



        # -----------------------------------------
        # Calculate role-wise SHAP importance
        # -----------------------------------------


        role_results = (

            merged

            .groupby(
                [
                    role_column,
                    "Influential Factor"
                ]
            )

            [
                "SHAP Impact"
            ]

            .mean()

            .reset_index()

        )



        role_results = role_results.sort_values(

            "SHAP Impact",

            ascending=False

        )



        role_results.to_excel(

            "results/role_xai_importance.xlsx",

            index=False

        )



        print(
            "Role importance Excel saved"
        )



        # -----------------------------------------
        # Create role charts
        # -----------------------------------------


        for role in role_results[role_column].unique():



            role_data = (

                role_results[

                    role_results[role_column] == role

                ]

                .head(5)

            )



            plt.figure(
                figsize=(8,5)
            )



            plt.barh(

                role_data["Influential Factor"],

                role_data["SHAP Impact"]

            )



            plt.xlabel(
                "Average SHAP Impact"
            )



            plt.title(

                str(role)
                +
                "\nMost Influential Performance Factors"

            )



            plt.gca().invert_yaxis()



            plt.tight_layout()



            filename = (

                "results/"

                +

                str(role)
                .replace(
                    " ",
                    "_"
                )

                +

                "_Factors.png"

            )



            plt.savefig(

                filename,

                dpi=300,

                bbox_inches="tight"

            )



            plt.close()



            chart_files.append(
                filename
            )



            print(
                "Chart created:",
                filename
            )



        # -----------------------------------------
        # Role heatmap
        # -----------------------------------------


        heatmap_data = (

            role_results

            .pivot(

                index=role_column,

                columns="Influential Factor",

                values="SHAP Impact"

            )

        )



        plt.figure(
            figsize=(12,6)
        )



        sns.heatmap(

            heatmap_data,

            annot=True,

            fmt=".3f"

        )



        plt.title(
            "Role Based Influential Factors Heatmap"
        )



        plt.tight_layout()



        heatmap_file = (

            "results/Role_Factor_Heatmap.png"

        )



        plt.savefig(

            heatmap_file,

            dpi=300,

            bbox_inches="tight"

        )



        plt.close()



        chart_files.append(
            heatmap_file
        )



    else:

        print(
            "No matching roles found after merge"
        )



else:

    print(
        "Role column not found:",
        role_column
    )



print("\nGenerated charts:")

for chart in chart_files:

    print(chart)


