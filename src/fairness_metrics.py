import pandas as pd


def calculate_gender_fairness(df):

    print("\n===== GENDER FAIRNESS METRICS =====")


    # Create binary positive outcome
    # High Performer = Positive outcome

    df = df.copy()

    df["Positive Outcome"] = (
        df["Performance Category"]
        .apply(
            lambda x: 1 if x == "High Performer" else 0
        )
    )


    # Remove groups with no valid records

    gender_counts = (
        df["Gender"]
        .value_counts()
    )

    valid_groups = (
        gender_counts[
            gender_counts >= 10
        ]
        .index
    )


    fairness_df = df[
        df["Gender"]
        .isin(valid_groups)
    ]


    # Positive outcome rate

    positive_rates = (
        fairness_df
        .groupby("Gender")["Positive Outcome"]
        .mean()
    )


    print("\nPositive Outcome Rate by Gender:")

    print(positive_rates)



    # --------------------------------
    # Demographic Parity Difference
    # --------------------------------

    demographic_parity_difference = (
        positive_rates.max()
        -
        positive_rates.min()
    )


    # --------------------------------
    # Disparate Impact Ratio
    # --------------------------------

    min_rate = positive_rates.min()

    max_rate = positive_rates.max()


    if max_rate == 0:

        disparate_impact_ratio = 0

    else:

        disparate_impact_ratio = (
            min_rate / max_rate
        )



    # --------------------------------
    # Equal Opportunity Difference
    # --------------------------------

    # Using High Performer as positive class

    true_positive_rates = (
        fairness_df
        .groupby("Gender")
        ["Positive Outcome"]
        .mean()
    )


    equal_opportunity_difference = (
        true_positive_rates.max()
        -
        true_positive_rates.min()
    )



    print(
        f"\nDemographic Parity Difference: "
        f"{demographic_parity_difference:.4f}"
    )


    print(
        f"Disparate Impact Ratio: "
        f"{disparate_impact_ratio:.4f}"
    )


    print(
        f"Equal Opportunity Difference: "
        f"{equal_opportunity_difference:.4f}"
    )


    return {

        "Positive Outcome Rate": positive_rates.to_dict(),

        "Demographic Parity Difference":
            demographic_parity_difference,

        "Disparate Impact Ratio":
            disparate_impact_ratio,

        "Equal Opportunity Difference":
            equal_opportunity_difference
    }



def calculate_ethnicity_fairness(df):

    print("\n===== ETHNICITY FAIRNESS ANALYSIS =====")


    df = df.copy()


    df["Positive Outcome"] = (
        df["Performance Category"]
        .apply(
            lambda x:
            1 if x == "High Performer"
            else 0
        )
    )


    ethnicity_counts = (
        df["Ethnicity"]
        .value_counts()
    )


    valid_groups = (
        ethnicity_counts[
            ethnicity_counts >= 10
        ]
        .index
    )


    fairness_df = df[
        df["Ethnicity"]
        .isin(valid_groups)
    ]


    positive_rates = (
        fairness_df
        .groupby("Ethnicity")
        ["Positive Outcome"]
        .mean()
    )


    print(
        "\nPositive Outcome Rate by Ethnicity:"
    )

    print(
        positive_rates
    )


    return positive_rates