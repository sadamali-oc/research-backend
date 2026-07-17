import sys
import time
import subprocess

from pathlib import Path
from datetime import datetime



# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_NAME = (
    "AI FUTURE EMPLOYEE PERFORMANCE "
    "PREDICTION SYSTEM"
)



PIPELINE_STEPS = [

    (
        "DATA PREPROCESSING MODULE",
        "preprocess.py"
    ),


    (
        "CLASSIFICATION MODEL TRAINING",
        "train.py"
    ),


    (
        "FUTURE SCORE REGRESSION MODEL TRAINING",
        "train_future_score_model.py"
    ),


    (
        "FUTURE PERFORMANCE PREDICTION MODULE",
        "predict_future_performance.py"
    ),


    (
        "MODEL COMPARISON EXCEL GENERATION",
        "model_comparison.py"
    ),


    (
        "MODEL COMPARISON PDF REPORT",
        "model_comparison_report.py"
    ),


    (
        "PERFORMANCE VISUALIZATION MODULE",
        "visualization.py"
    ),


    (
        "CLASSIFICATION XAI MODULE",
        "explain_prediction.py"
    ),


    (
        "FUTURE SCORE SHAP EXPLANATION",
        "future_performance_shap_explanation.py"
    )

]



REQUIRED_FOLDERS = [

    "data",

    "results",

    "models"

]



LOG_FILE = (
    "results/pipeline_log.txt"
)



# ==========================================
# FUNCTIONS
# ==========================================


def print_header(message):

    print("\n" + "=" * 60)

    print(message)

    print("=" * 60)




def create_folders():

    for folder in REQUIRED_FOLDERS:

        Path(folder).mkdir(

            exist_ok=True

        )




def write_log(message):

    Path("results").mkdir(

        exist_ok=True

    )


    with open(

        LOG_FILE,

        "a",

        encoding="utf-8"

    ) as file:


        file.write(

            f"{datetime.now()} : {message}\n"

        )




def run_script(number, title, script):


    print(

        f"\n[{number}/{len(PIPELINE_STEPS)}] {title}"

    )


    print("-" * 60)



    script_path = Path(script)



    if not script_path.exists():


        error = (

            f"{script} not found"

        )


        print(

            "ERROR:",

            error

        )


        write_log(error)


        sys.exit(1)




    start_time = time.time()



    try:


        subprocess.run(

            [

                sys.executable,

                str(script_path)

            ],

            check=True

        )



        duration = (

            time.time()

            -

            start_time

        )



        message = (

            f"{script} completed "

            f"in {duration:.2f}s"

        )



        print(

            "\n✓",

            message

        )



        write_log(message)




    except subprocess.CalledProcessError as error:



        message = (

            f"{script} failed "

            f"Exit Code={error.returncode}"

        )



        print(

            "\n✗",

            message

        )



        write_log(message)



        sys.exit(

            error.returncode

        )





def check_outputs():


    expected_files = [


        # Dataset

        "data/performance_future_features.xlsx",



        # Classification model

        "results/best_model.pkl",

        "results/final_model_metrics.xlsx",



        # Regression model

        "results/future_score_model.pkl",

        "results/future_score_metrics.xlsx",



        # Prediction

        "results/future_performance_prediction.xlsx",



        # Comparison

        "results/model_comparison/future_model_comparison.xlsx",

        "results/model_comparison/Model_Comparison_Report.pdf",



        # Classification XAI

        "results/employee_future_explanation.xlsx",



        # Regression SHAP

        "results/shap/future_score_feature_importance.png",

        "results/shap/future_score_shap_values.xlsx",

        "results/shap/employee_0_explanation.xlsx"



    ]



    print(

        "\nChecking Generated Files"

    )


    print("-" * 60)



    for file in expected_files:



        if Path(file).exists():


            print(

                "✓",

                file

            )


        else:


            print(

                "⚠ Missing:",

                file

            )





def display_results():


    print(
"""
Generated Outputs:

✓ Future Performance Feature Dataset

✓ Classification Prediction Model

✓ Future Score Regression Model

✓ Future Performance Score Prediction

✓ Model Comparison Analysis

✓ Model Comparison PDF Report

✓ Performance Visualization

✓ Classification XAI Explanation

✓ Future Score SHAP Explanation

✓ Pipeline Execution Log

"""
    )





# ==========================================
# MAIN PIPELINE
# ==========================================


def main():


    create_folders()



    print_header(

        PROJECT_NAME

    )



    write_log(

        "PIPELINE STARTED"

    )



    start = time.time()



    print(

        "Starting AI Performance Prediction Pipeline..."

    )




    for index, (title, script) in enumerate(


        PIPELINE_STEPS,


        start=1


    ):



        run_script(

            index,

            title,

            script

        )





    total_time = (

        time.time()

        -

        start

    )





    print_header(

        "SYSTEM EXECUTION COMPLETED"

    )




    print(

        f"Total Execution Time: {total_time:.2f} seconds"

    )




    write_log(

        f"PIPELINE COMPLETED {total_time:.2f}s"

    )




    check_outputs()



    display_results()





if __name__ == "__main__":

    main()