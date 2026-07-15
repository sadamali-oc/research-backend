import os
import subprocess
import sys


print("====================================")
print("Behavior Pattern Identification Model")
print("Starting Execution Pipeline")
print("====================================")


# Project root directory

project_path = os.path.dirname(os.path.abspath(__file__))


# Source folder

src_path = os.path.join(project_path, "src")



def run_stage(file_name, stage_name):

    print(f"\nRunning {stage_name} Stage...")

    file_path = os.path.join(src_path, file_name)

    try:

        subprocess.run(
            [sys.executable, file_path],
            check=True
        )

        print(f"{stage_name} Completed Successfully")

    except Exception as e:

        print(f"{stage_name} Failed")
        print(e)



# ==============================
# Pipeline Execution
# ==============================


run_stage(
    "preprocess.py",
    "Preprocessing"
)



run_stage(
    "feature_engineering.py",
    "Feature Engineering"
)



run_stage(
    "behaviour_pattern_identification_engine.py",
    "Behavioral Scoring"
)



run_stage(
    "random_forest.py",
    "Multi-Output Random Forest"
)



run_stage(
    "prediction.py",
    "Prediction"
)



run_stage(
    "visualization.py",
    "Visualization"
)



print("\n====================================")
print("Behavior Pattern Identification Completed Successfully")
print("====================================")