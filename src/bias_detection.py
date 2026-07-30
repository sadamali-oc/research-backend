import pandas as pd



def analyze_gender_bias(df):


    if (
        "Gender" not in df.columns
        or "Performance Score" not in df.columns
    ):

        return {
            "error": "Required columns not available"
        }



    # Remove unknown demographic values

    filtered_df = df[
        df["Gender"].astype(str)
        != "Unknown"
    ]



    gender_scores = (

        filtered_df
        .groupby("Gender")
        ["Performance Score"]
        .mean()

    )



    if len(gender_scores) < 2:

        return {
            "error": "Insufficient gender groups"
        }



    bias_gap = (

        gender_scores.max()
        -
        gender_scores.min()

    )



    result = {


        "average_score_by_gender":
            gender_scores.to_dict(),


        "bias_gap":
            round(
                float(bias_gap),
                2
            ),


        "status":

            "Potential Bias"
            if bias_gap > 10
            else
            "Fair"

    }



    return result





def analyze_ethnicity_bias(df):


    if (
        "Ethnicity" not in df.columns
        or "Performance Score" not in df.columns
    ):

        return {
            "error": "Required columns not available"
        }



    # Remove unknown demographic values

    filtered_df = df[
        df["Ethnicity"].astype(str)
        != "Unknown"
    ]



    ethnicity_scores = (

        filtered_df
        .groupby("Ethnicity")
        ["Performance Score"]
        .mean()

    )



    if len(ethnicity_scores) < 2:

        return {
            "error": "Insufficient ethnicity groups"
        }



    bias_gap = (

        ethnicity_scores.max()
        -
        ethnicity_scores.min()

    )



    result = {


        "average_score_by_ethnicity":
            ethnicity_scores.to_dict(),


        "bias_gap":
            round(
                float(bias_gap),
                2
            ),


        "status":

            "Potential Bias"
            if bias_gap > 10
            else
            "Fair"

    }



    return result