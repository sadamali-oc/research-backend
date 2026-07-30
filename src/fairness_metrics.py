import pandas as pd



def demographic_parity(data):


    if (
        "Gender" not in data.columns
        or "Performance Category" not in data.columns
    ):

        return None



    filtered_data = data[
        data["Gender"].astype(str)
        != "Unknown"
    ]



    selection_rate = (

        filtered_data
        .groupby("Gender")
        ["Performance Category"]
        .apply(
            lambda x:
            (
                x == "High Performer"
            ).mean()
        )

    )



    if len(selection_rate) < 2:

        return None



    parity_ratio = (

        selection_rate.min()
        /
        selection_rate.max()

    )



    return {

        "selection_rate":
            selection_rate.to_dict(),

        "parity_ratio":
            round(
                float(parity_ratio),
                3
            ),

        "status":

            "Fair"
            if parity_ratio >= 0.8
            else
            "Potential Bias"

    }





def disparate_impact(data):


    result = demographic_parity(
        data
    )



    if result is None:

        return None



    return {

        "disparate_impact_ratio":
            result["parity_ratio"],

        "status":
            result["status"]

    }





def calculate_fairness_metrics(data):


    return {


        "demographic_parity":

            demographic_parity(
                data
            ),



        "disparate_impact":

            disparate_impact(
                data
            )

    }