from src.bias_detection import (
    analyze_gender_bias,
    analyze_ethnicity_bias
)

from src.fairness_metrics import (
    calculate_fairness_metrics
)



def run_fairness_analysis(data):


    results = {}



    # ==========================
    # Bias Detection
    # ==========================

    results["gender_bias"] = (
        analyze_gender_bias(data)
    )


    results["ethnicity_bias"] = (
        analyze_ethnicity_bias(data)
    )



    # ==========================
    # Fairness Metrics
    # ==========================

    results["fairness_metrics"] = (
        calculate_fairness_metrics(data)
    )



    return results