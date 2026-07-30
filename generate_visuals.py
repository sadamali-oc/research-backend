import json

from src.visualizer import (
    plot_gender_bias,
    plot_ethnicity_bias,
    plot_fairness_metrics
)



# Load fairness report

with open(
    "outputs/fairness_report.json",
    "r"
) as file:

    data = json.load(file)



# Generate charts


plot_gender_bias(
    data["gender_bias"]["group_scores"]
)


plot_ethnicity_bias(
    data["ethnicity_bias"]["group_scores"]
)


plot_fairness_metrics(
    {
        "demographic_parity":
            data["demographic_parity"],

        "disparate_impact":
            data["fairness_metrics"]["disparate_impact"]
    }
)



print(
    "Visualization completed successfully"
)