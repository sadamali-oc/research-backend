import sys
import time
import subprocess

from pathlib import Path
from datetime import datetime


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_NAME = "AI FUTURE EMPLOYEE PERFORMANCE PREDICTION SYSTEM"


PIPELINE_STEPS = [

    (
        "DATA PREPROCESSING MODULE",
        "preprocessing/preprocess.py"
    ),

    (
        "MODEL TRAINING MODULE",
        "models/train.py"
    ),

    (
        "FUTURE PERFORMANCE PREDICTION MODULE",
        "prediction/predict.py"
    ),

    (
        "MODEL EVALUATION MODULE",
        "evaluation/evaluate_model.py"
    )

]


REQUIRED_FOLDERS = [
    "data",
    "models",
    "results"
]


LOG_FILE = "results/pipeline_log.txt"



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

        print(
            "ERROR:",
            script,
            "not found"
        )

        write_log(
            f"{script} missing"
        )

        sys.exit(1)



    start_time = time.time()


    try:

        subprocess.run(
            [
                sys.executable,
                "-u",
                str(script_path)
            ],
            check=True
        )


        duration = time.time() - start_time


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

        "data/performance_future_features.xlsx",

        "models/random_forest_classifier.pkl",

        "models/random_forest_regressor.pkl",

        "results/employee_performance_predictions.xlsx",

        "results/Future_Performance_Evaluation_Report.pdf"

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



    total_time = time.time() - start



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