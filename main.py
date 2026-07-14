import sys
import time
import subprocess


def run_step(title, script_name):
    print(f"\n{title}")
    print("-" * 44)
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"{script_name} failed!")
        sys.exit(result.returncode)


def main():
    print("\n============================================")
    print(" AI EMPLOYEE PERFORMANCE PREDICTION SYSTEM ")
    print("============================================\n")

    start_time = time.time()

    steps = [
        ("[1/5] STARTING DATA PREPROCESSING MODULE", "preprocess.py"),
        ("[2/5] STARTING MODEL TRAINING MODULE", "train.py"),
        ("[3/5] STARTING MODEL EVALUATION MODULE", "evaluate.py"),
        ("[4/5] STARTING VISUALIZATION MODULE", "visualization.py"),
        ("[5/5] STARTING EMPLOYEE EXPLANATION MODULE", "explain_prediction.py"),
    ]

    for title, script in steps:
        run_step(title, script)
        print(f"\n{script} completed successfully")

    end_time = time.time()

    print("\n============================================")
    print(" SYSTEM EXECUTION COMPLETED ")
    print("============================================")
    print(f"\nTotal Time: {end_time - start_time:.2f} seconds")
    print("""
Generated Outputs:

✓ Processed Dataset
✓ Trained ML Models
✓ Model Comparison
✓ Feature Importance
✓ Performance Evaluation PDF Report
✓ Employee Performance Explanation Report
""")


if __name__ == "__main__":
    main()