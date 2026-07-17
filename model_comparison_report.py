import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages



# ======================================================
# SETTINGS
# ======================================================


RESULT_PATH = Path("results")


# Create model comparison folder

MODEL_COMPARISON_PATH = (
    RESULT_PATH /
    "model_comparison"
)


MODEL_COMPARISON_PATH.mkdir(
    exist_ok=True
)


# Excel input/output file

COMPARISON_FILE = (
    MODEL_COMPARISON_PATH /
    "future_model_comparison.xlsx"
)


# PDF output file

PDF_FILE = (
    MODEL_COMPARISON_PATH /
    "Model_Comparison_Report.pdf"
)


# ======================================================
# LOAD RESULTS
# ======================================================

print(
"""
====================================
MODEL COMPARISON PDF REPORT
====================================
"""
)


if not COMPARISON_FILE.exists():

    raise FileNotFoundError(
        "future_model_comparison.xlsx not found. Run model_comparison.py first."
    )


df = pd.read_excel(
    COMPARISON_FILE
)


print(df)



# ======================================================
# BEST MODEL
# ======================================================


best = df.sort_values(
    "F1 Score",
    ascending=False
).iloc[0]



# ======================================================
# COLORS
# ======================================================

algorithm_colors = {

    "Decision Tree":
        "steelblue",

    "Random Forest":
        "seagreen",

    "Gradient Boosting":
        "darkorange"

}



colors = [
    algorithm_colors.get(
        algo,
        "gray"
    )
    for algo in df["Algorithm"]
]



# ======================================================
# FUNCTION FOR BAR CHART
# ======================================================


def create_bar_chart(
        metric,
        title,
        ylabel
):


    fig,ax = plt.subplots(
        figsize=(8,5)
    )


    values = df[metric] * 100


    bars=ax.bar(

        df["Algorithm"],

        values,

        color=colors

    )


    for bar,value in zip(
        bars,
        values
    ):

        ax.text(

            bar.get_x()+bar.get_width()/2,

            bar.get_height()+0.5,

            f"{value:.2f}%",

            ha="center",

            fontsize=9

        )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title
    )


    plt.xticks(
        rotation=30
    )


    plt.tight_layout()


    return fig





# ======================================================
# CREATE PDF
# ======================================================


with PdfPages(PDF_FILE) as pdf:



    # ==========================================
    # SUMMARY PAGE
    # ==========================================


    fig,ax = plt.subplots(
        figsize=(8,6)
    )


    ax.axis("off")


    summary=f"""

AI FUTURE EMPLOYEE PERFORMANCE
PREDICTION SYSTEM


MODEL COMPARISON REPORT


Algorithms Compared:

✓ Decision Tree

✓ Random Forest

✓ Gradient Boosting



Best Performing Model:

{best['Algorithm']}



Evaluation Results:


Accuracy:
{best['Accuracy']*100:.2f}%


F1 Score:
{best['F1 Score']*100:.2f}%


ROC-AUC:
{best['ROC-AUC']*100:.2f}%


"""


    ax.text(

        0.05,

        0.95,

        summary,

        fontsize=12,

        verticalalignment="top"

    )


    plt.title(
        "Research Model Comparison Summary"
    )


    pdf.savefig(
        fig,
        dpi=300
    )


    plt.close()




    # ==========================================
    # TABLE
    # ==========================================


    fig,ax = plt.subplots(
        figsize=(10,5)
    )


    ax.axis("off")


    table=ax.table(

        cellText=df.round(4).values,

        colLabels=df.columns,

        loc="center"

    )


    table.auto_set_font_size(False)

    table.set_fontsize(9)

    table.scale(
        1,
        2
    )


    plt.title(
        "Algorithm Performance Comparison"
    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # ==========================================
    # ACCURACY
    # ==========================================


    fig=create_bar_chart(

        "Accuracy",

        "Accuracy Comparison",

        "Accuracy (%)"

    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # ==========================================
    # F1 SCORE
    # ==========================================


    fig=create_bar_chart(

        "F1 Score",

        "F1 Score Comparison",

        "F1 Score (%)"

    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # ==========================================
    # ROC-AUC
    # ==========================================


    fig=create_bar_chart(

        "ROC-AUC",

        "ROC-AUC Comparison",

        "ROC-AUC (%)"

    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # ==========================================
    # TRAINING TIME
    # ==========================================


    fig,ax=plt.subplots(
        figsize=(8,5)
    )


    bars=ax.bar(

        df["Algorithm"],

        df["Training Time"],

        color=colors

    )


    for bar,value in zip(
        bars,
        df["Training Time"]
    ):

        ax.text(

            bar.get_x()+bar.get_width()/2,

            bar.get_height(),

            f"{value:.2f}s",

            ha="center",

            fontsize=9

        )


    ax.set_ylabel(
        "Seconds"
    )


    ax.set_title(
        "Training Execution Time Comparison"
    )


    plt.xticks(
        rotation=30
    )


    plt.tight_layout()


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()




    # ==========================================
    # CURVE / LINE CHART
    # ==========================================


    fig,ax=plt.subplots(
        figsize=(8,5)
    )


    metrics=[

        "Accuracy",

        "F1 Score",

        "ROC-AUC"

    ]


    for i,row in df.iterrows():

        ax.plot(

            metrics,

            row[metrics]*100,

            marker="o",

            label=row["Algorithm"]

        )



    ax.set_ylabel(
        "Performance (%)"
    )


    ax.set_title(
        "Algorithm Performance Curve"
    )


    ax.legend()


    ax.grid(
        True
    )


    pdf.savefig(
        fig,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


print(
"""
====================================
PDF GENERATED SUCCESSFULLY
====================================

Saved:

results/model_comparison/Model_Comparison_Report.pdf


Contains:

✓ Research Summary
✓ Best Model Result
✓ Comparison Table
✓ Accuracy Comparison Chart
✓ F1 Score Comparison Chart
✓ ROC-AUC Comparison Chart
✓ Training Time Comparison Chart
✓ Performance Curve


====================================
"""
)