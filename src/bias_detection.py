import pandas as pd
from sklearn.metrics import confusion_matrix


def analyze_gender_bias(df):

    print("\n===== GENDER BIAS ANALYSIS =====")


    gender_report = (
        df.groupby("Gender")
        ["Performance Category"]
        .value_counts(normalize=True)
        .unstack()
    )


    print(gender_report)


    return gender_report



def analyze_ethnicity_bias(df):

    print("\n===== ETHNICITY BIAS ANALYSIS =====")


    ethnicity_report = (
        df.groupby("Ethnicity")
        ["Performance Category"]
        .value_counts(normalize=True)
        .unstack()
    )


    print(ethnicity_report)


    return ethnicity_report



def demographic_parity(df):

    print("\n===== DEMOGRAPHIC PARITY =====")


    result = (
        df.groupby("Gender")
        ["Performance Score"]
        .mean()
    )


    print(result)


    return result