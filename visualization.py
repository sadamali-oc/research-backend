import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix



# =====================================================
# SETTINGS
# =====================================================

plt.style.use("seaborn-v0_8-whitegrid")

os.makedirs(
    "results",
    exist_ok=True
)



print("\nSTARTING THESIS STYLE VISUALIZATION\n")



# =====================================================
# LOAD FILES
# =====================================================


comparison_file = "results/model_comparison.xlsx"

feature_file = "results/feature_importance.xlsx"

dataset_file = "data/performance_features.xlsx"

model_file = "results/best_model.pkl"



comparison = pd.read_excel(
    comparison_file
)


features = pd.read_excel(
    feature_file
)


data = pd.read_excel(
    dataset_file
)



# =====================================================
# CONVERT VALUES TO %
# =====================================================


comparison_percent = comparison.copy()


for col in [
    "Accuracy",
    "F1 Score",
    "ROC-AUC",
    "CV Mean"
]:

    if col in comparison_percent.columns:

        comparison_percent[col] = (
            comparison_percent[col]*100
        ).round(2)



features["Importance (%)"] = (

    features["Importance"]*100

).round(2)



features = features.sort_values(
    "Importance (%)",
    ascending=False
)



# =====================================================
# PDF
# =====================================================


pdf_path = (
    "results/Performance_Evaluation_Thesis_Report.pdf"
)



with PdfPages(pdf_path) as pdf:



    # =================================================
    # PAGE 1
    # MODEL COMPARISON TABLE
    # =================================================


    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.axis("off")


    table_data = comparison_percent.copy()


    for col in table_data.columns:

        if col in [
            "Accuracy",
            "F1 Score",
            "ROC-AUC",
            "CV Mean"
        ]:

            table_data[col] = (
                table_data[col].astype(str)
                +"%"
            )


    table=ax.table(

        cellText=table_data.values,

        colLabels=table_data.columns,

        loc="center"

    )


    table.auto_set_font_size(False)

    table.set_fontsize(10)

    table.scale(
        1,
        2
    )


    plt.title(

        "Table 1: Performance Comparison of Machine Learning Algorithms",

        fontsize=15,

        fontweight="bold"

    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()




    # =================================================
    # PAGE 2
    # ACCURACY GRAPH
    # =================================================


    fig,ax=plt.subplots(
        figsize=(9,5)
    )


    temp=comparison_percent.sort_values(
        "Accuracy"
    )


    bars=ax.bar(

        temp["Algorithm"],

        temp["Accuracy"]

    )


    ax.set_title(

        "Figure 1: Accuracy Comparison of Prediction Models",

        fontsize=14,

        fontweight="bold"

    )


    ax.set_ylabel(
        "Accuracy (%)"
    )


    ax.set_ylim(
        0,
        100
    )


    for bar in bars:

        ax.text(

            bar.get_x()+bar.get_width()/2,

            bar.get_height()+1,

            f"{bar.get_height():.1f}%",

            ha="center",

            fontweight="bold"

        )


    plt.xticks(
        rotation=20
    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()




    # =================================================
    # PAGE 3
    # F1 AND ROC
    # =================================================


    fig,ax=plt.subplots(
        figsize=(10,6)
    )


    x=np.arange(
        len(comparison_percent)
    )


    width=0.25


    ax.bar(
        x-width,
        comparison_percent["F1 Score"],
        width,
        label="F1 Score"
    )


    ax.bar(
        x,
        comparison_percent["ROC-AUC"],
        width,
        label="ROC-AUC"
    )


    ax.bar(
        x+width,
        comparison_percent["CV Mean"],
        width,
        label="Cross Validation"
    )


    ax.set_xticks(x)

    ax.set_xticklabels(
        comparison_percent["Algorithm"],
        rotation=20
    )


    ax.set_ylabel(
        "Percentage (%)"
    )


    ax.set_ylim(
        0,
        100
    )


    ax.legend()



    ax.set_title(

        "Figure 2: Evaluation Metrics Comparison",

        fontsize=14,

        fontweight="bold"

    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # =================================================
    # PAGE 4
    # FEATURE IMPORTANCE
    # =================================================


    top_features = features.head(10)



    fig,ax=plt.subplots(
        figsize=(10,6)
    )


    ax.barh(

        top_features["Feature"][::-1],

        top_features["Importance (%)"][::-1]

    )


    ax.set_xlabel(
        "Importance (%)"
    )


    ax.set_title(

        "Figure 3: Most Influential KPI Factors Affecting Performance Prediction",

        fontsize=14,

        fontweight="bold"

    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # =================================================
    # PAGE 5
    # FEATURE TABLE
    # =================================================


    fig,ax=plt.subplots(
        figsize=(10,5)
    )

    ax.axis("off")


    ft=top_features[
        [
            "Feature",
            "Importance (%)"
        ]
    ]


    table=ax.table(

        cellText=ft.values,

        colLabels=ft.columns,

        loc="center"

    )


    table.scale(
        1,
        2
    )


    plt.title(

        "Table 2: Ranking of Important Performance Factors",

        fontsize=15,

        fontweight="bold"

    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # =================================================
    # PAGE 6
    # PERFORMANCE DISTRIBUTION
    # =================================================


    counts=data[
        "Performance_Category"
    ].value_counts()



    fig,ax=plt.subplots(
        figsize=(8,5)
    )


    bars=ax.bar(

        counts.index.astype(str),

        counts.values

    )


    ax.set_title(

        "Figure 4: Employee Performance Category Distribution",

        fontsize=14,

        fontweight="bold"

    )


    ax.set_ylabel(
        "Number of Employees"
    )


    for b in bars:

        ax.text(

            b.get_x()+b.get_width()/2,

            b.get_height()+5,

            str(int(b.get_height())),

            ha="center"

        )



    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # =================================================
    # PAGE 7
    # FINAL SUMMARY
    # =================================================


    best = comparison.sort_values(
        "Accuracy",
        ascending=False
    ).iloc[0]



    fig,ax=plt.subplots(
        figsize=(8,6)
    )


    ax.axis("off")



    summary=f"""

AI EMPLOYEE PERFORMANCE PREDICTION MODEL

Dataset:
1000 Employee Records

Input:
18 Structured KPI Features

Best Algorithm:
{best['Algorithm']}

Accuracy:
{best['Accuracy']*100:.2f}%

F1 Score:
{best['F1 Score']*100:.2f}%

ROC-AUC:
{best['ROC-AUC']*100:.2f}%

Cross Validation:
{best['CV Mean']*100:.2f}%


Most Influential Factors:

1. {features.iloc[0]['Feature']}
2. {features.iloc[1]['Feature']}
3. {features.iloc[2]['Feature']}

"""


    ax.text(

        0.1,
        0.8,

        summary,

        fontsize=12,

        verticalalignment="top"

    )


    plt.title(

        "Research Model Summary",

        fontsize=16,

        fontweight="bold"

    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




print(
"""
=====================================

THESIS STYLE PDF GENERATED

File:

results/Performance_Evaluation_Thesis_Report.pdf


Pages:

1. Model comparison table
2. Accuracy comparison
3. Evaluation metrics
4. Feature importance chart
5. Feature ranking table
6. Performance distribution
7. Research summary

=====================================
"""
)