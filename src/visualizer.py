import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Create output folder for charts

CHART_FOLDER = Path(
    "outputs/charts"
)

CHART_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)



def plot_gender_bias(gender_data):

    """
    Generate gender performance comparison chart
    """

    groups = list(
        gender_data.keys()
    )

    scores = list(
        gender_data.values()
    )


    plt.figure(
        figsize=(7,4)
    )


    sns.barplot(
        x=groups,
        y=scores
    )


    plt.title(
        "Average Performance Score by Gender"
    )


    plt.xlabel(
        "Gender Group"
    )


    plt.ylabel(
        "Average Performance Score"
    )


    plt.tight_layout()


    path = CHART_FOLDER / "gender_bias.png"


    plt.savefig(
        path
    )


    plt.close()


    return path





def plot_ethnicity_bias(ethnicity_data):

    """
    Generate ethnicity performance comparison chart
    """


    groups = list(
        ethnicity_data.keys()
    )


    scores = list(
        ethnicity_data.values()
    )



    plt.figure(
        figsize=(7,4)
    )


    sns.barplot(
        x=groups,
        y=scores
    )


    plt.title(
        "Average Performance Score by Ethnicity"
    )


    plt.xlabel(
        "Ethnicity Group"
    )


    plt.ylabel(
        "Average Performance Score"
    )


    plt.tight_layout()



    path = CHART_FOLDER / "ethnicity_bias.png"



    plt.savefig(
        path
    )


    plt.close()


    return path





def plot_fairness_metrics(metrics):

    """
    Generate fairness metric comparison chart
    """


    metric_names = [
        "Demographic Parity",
        "Disparate Impact"
    ]


    values = [
        metrics["demographic_parity"]["parity_ratio"],
        metrics["disparate_impact"]["disparate_impact_ratio"]
    ]



    plt.figure(
        figsize=(7,4)
    )


    sns.barplot(
        x=metric_names,
        y=values
    )


    plt.axhline(
        y=0.8,
        linestyle="--"
    )


    plt.title(
        "Fairness Metrics"
    )


    plt.ylabel(
        "Score"
    )


    plt.ylim(
        0,
        1
    )


    plt.tight_layout()



    path = CHART_FOLDER / "fairness_metrics.png"



    plt.savefig(
        path
    )


    plt.close()


    return path