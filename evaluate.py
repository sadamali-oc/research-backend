import pandas as pd
import matplotlib.pyplot as plt
import os


print("\nSTARTING MODEL EVALUATION MODULE\n")


# =====================================================
# 1. LOAD MODEL RESULTS
# =====================================================

file_path = "results/model_comparison.xlsx"


if not os.path.exists(file_path):

    print("Model comparison file not found!")
    exit()



comparison = pd.read_excel(file_path)



print("\nMODEL PERFORMANCE COMPARISON")

print(comparison)



# =====================================================
# 2. FIND BEST MODEL
# =====================================================


best_model = comparison.loc[
    comparison["F1 Score"].idxmax()
]


print("\nBEST PERFORMING MODEL")

print(
    "Algorithm:",
    best_model["Algorithm"]
)

print(
    "Accuracy:",
    round(best_model["Accuracy"],4)
)

print(
    "F1 Score:",
    round(best_model["F1 Score"],4)
)



# =====================================================
# 3. CREATE COMPARISON CHART
# =====================================================


os.makedirs(
    "results/charts",
    exist_ok=True
)



plt.figure(
    figsize=(8,5)
)


plt.bar(

    comparison["Algorithm"],

    comparison["Accuracy"]

)


plt.title(
    "Machine Learning Model Accuracy Comparison"
)


plt.xlabel(
    "Algorithm"
)


plt.ylabel(
    "Accuracy"
)



plt.ylim(
    0,
    1
)



plt.xticks(
    rotation=45
)



plt.tight_layout()



plt.savefig(

    "results/charts/model_accuracy_comparison.png"

)



plt.close()



# =====================================================
# 4. SAVE SUMMARY
# =====================================================


summary = pd.DataFrame({

    "Best Model":[
        best_model["Algorithm"]
    ],

    "Accuracy":[
        best_model["Accuracy"]
    ],

    "F1 Score":[
        best_model["F1 Score"]
    ]

})



summary.to_excel(

    "results/best_model_summary.xlsx",

    index=False

)



print("\nEvaluation completed successfully")

print(
    "Charts saved in results/charts/"
)