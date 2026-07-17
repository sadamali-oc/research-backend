import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = Path(
    "results/future_performance_prediction.xlsx"
)

OUTPUT_FOLDER = Path(
    "results/visualizations"
)

OUTPUT_FOLDER.mkdir(
    exist_ok=True
)


SUMMARY_FILE = Path(
    "results/performance_change_summary.xlsx"
)



# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading prediction results...")

df = pd.read_excel(
    INPUT_FILE
)


print(df.head())



# ==========================================
# PERFORMANCE CHANGE CALCULATION
# ==========================================

df["Performance_Change"] = (

    df["Predicted_Future_Score"]

    -

    df["Current_Performance_Score"]

)



df["Performance_Trend"] = df["Performance_Change"].apply(

    lambda x:

    "Improved" if x > 0

    else

    "Declined"

)



# Save updated analysis

df.to_excel(

    SUMMARY_FILE,

    index=False

)



# Statistics

improved = (

    df["Performance_Trend"]
    ==
    "Improved"

).sum()


declined = (

    df["Performance_Trend"]
    ==
    "Declined"

).sum()



average_change = (

    df["Performance_Change"]
    .mean()

)



print("\nPerformance Summary")

print(
"Improved Employees:",
improved
)

print(
"Declined Employees:",
declined
)

print(
"Average Change:",
round(average_change,2)
)




# ==========================================
# CHART 1
# CURRENT VS FUTURE SCORE
# ==========================================


# Use sample for readability

sample = df.head(50)



plt.figure(
    figsize=(12,6)
)



plt.plot(

    sample.index,

    sample["Current_Performance_Score"],

    marker="o",

    label="Current Performance"

)



plt.plot(

    sample.index,

    sample["Predicted_Future_Score"],

    marker="o",

    label="Predicted Future Performance"

)



plt.axhline(

    df["Current_Performance_Score"].mean(),

    linestyle="--",

    label="Average Current Score"

)



plt.title(

    "Current Employee Performance vs AI Predicted Future Performance",

    fontsize=14

)


plt.xlabel(
    "Employee Sample"
)


plt.ylabel(
    "Performance Score"
)


plt.ylim(
    0,
    100
)


plt.legend()

plt.grid()



plt.savefig(

    OUTPUT_FOLDER /

    "current_vs_future_score.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()




# ==========================================
# CHART 2
# PERFORMANCE CHANGE DISTRIBUTION
# ==========================================


plt.figure(

    figsize=(10,6)

)



plt.hist(

    df["Performance_Change"],

    bins=25

)



plt.axvline(

    0,

    linestyle="--",

    label="No Change"

)



plt.axvline(

    average_change,

    linestyle="--",

    label="Average Change"

)



plt.title(

    "Distribution of Future Performance Change",

    fontsize=14

)



plt.xlabel(

    "Future Score - Current Score"

)



plt.ylabel(

    "Number of Employees"

)



plt.legend()

plt.grid()



plt.savefig(

    OUTPUT_FOLDER /

    "performance_change_distribution.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()




# ==========================================
# CHART 3
# PERFORMANCE LEVEL TRANSITION
# ==========================================


transition = pd.crosstab(

    df["Current_Performance_Level"],

    df["Predicted_Future_Performance_Level"]

)



ax = transition.plot(

    kind="bar",

    figsize=(10,6)

)



plt.title(

    "Employee Performance Level Transition",

    fontsize=14

)



plt.xlabel(

    "Current Performance Level"

)



plt.ylabel(

    "Number of Employees"

)



plt.xticks(

    rotation=0

)



plt.legend(

    title="Predicted Future Level"

)



plt.grid()



plt.savefig(

    OUTPUT_FOLDER /

    "performance_level_transition.png",

    dpi=300,

    bbox_inches="tight"

)



plt.close()




# ==========================================
# CHART 4
# TOP IMPROVEMENT EMPLOYEES
# ==========================================


top_improvement = df.sort_values(

    "Performance_Change",

    ascending=False

).head(10)



plt.figure(

    figsize=(10,6)

)



plt.bar(

    top_improvement["Employee_ID"].astype(str),

    top_improvement["Performance_Change"]

)



plt.title(

    "Top 10 Predicted Performance Improvements"

)



plt.xlabel(

    "Employee ID"

)



plt.ylabel(

    "Score Improvement"

)



plt.xticks(

    rotation=45

)


plt.grid()



plt.savefig(

    OUTPUT_FOLDER /

    "top_performance_improvement.png",

    dpi=300,

    bbox_inches="tight"

)



plt.close()




print(
"""
====================================
VISUALIZATION COMPLETED

Generated:

✓ current_vs_future_score.png
✓ performance_change_distribution.png
✓ performance_level_transition.png
✓ top_performance_improvement.png

Saved:
results/visualizations/

Summary:
results/performance_change_summary.xlsx

====================================
"""
)