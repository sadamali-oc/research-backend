import pandas as pd
from pathlib import Path


print("""
====================================
FUTURE PERFORMANCE EXPLANATION
====================================
""")


# ===============================
# CONFIGURATION
# ===============================

DATA_FILE = Path(
    "data/performance_future_features.xlsx"
)

RESULT_PATH = Path(
    "results"
)

RESULT_PATH.mkdir(
    exist_ok=True
)


OUTPUT_FILE = (
    RESULT_PATH /
    "employee_future_explanation.xlsx"
)



# ===============================
# LOAD DATA
# ===============================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )


df = pd.read_excel(DATA_FILE)


print(
    "Dataset:",
    df.shape
)


print("\nAvailable Columns:")
print(df.columns.tolist())



# ===============================
# CREATE EMPLOYEE ID
# ===============================

print("\nChecking Employee ID...")


if "Employee ID" not in df.columns:

    if "Row_ID" in df.columns:

        print(
            "Creating Employee ID from Row_ID..."
        )

        df.rename(
            columns={
                "Row_ID": "Employee ID"
            },
            inplace=True
        )

    else:

        print(
            "Employee ID not found. Creating new IDs..."
        )

        df["Employee ID"] = [
            f"E{i+1:04d}"
            for i in range(len(df))
        ]



# ===============================
# CREATE SCORE CHANGE
# ===============================

print("\nChecking Score Change feature...")


if "Score_Change" not in df.columns:

    print(
        "Creating Score_Change..."
    )


    df["Score_Change"] = (

        df["Future_Performance_Score"]

        -

        df["Previous_Performance_Score"]

    )


else:

    print(
        "Score_Change already exists."
    )



# ===============================
# CREATE PREVIOUS PERFORMANCE BAND
# ===============================

print("\nCreating Previous Performance Band...")


if "Previous_Performance_Band" not in df.columns:


    def create_band(score):

        if score < 50:

            return "Low"

        elif score < 75:

            return "Medium"

        else:

            return "High"



    df["Previous_Performance_Band"] = (

        df["Previous_Performance_Score"]

        .apply(create_band)

    )


else:

    print(
        "Previous Performance Band already exists."
    )



# ===============================
# VALIDATE REQUIRED COLUMNS
# ===============================

required_columns = [

    "Employee ID",

    "Previous_Performance_Score",

    "Previous_Performance_Band",

    "Future_Performance_Score",

    "Future_Performance_Category",

    "Score_Change"

]


missing = [

    col

    for col in required_columns

    if col not in df.columns

]


if missing:

    raise Exception(

        f"""
Missing columns:

{missing}

Please check your dataset.
"""

    )



# ===============================
# CREATE REPORT TABLE
# ===============================


report = df[

    required_columns

].copy()



# ===============================
# RENAME COLUMNS
# ===============================


report.columns = [

    "Employee ID",

    "Previous Score",

    "Previous Band",

    "Future Score",

    "Future Category",

    "Score Change"

]



# ===============================
# ROUND VALUES
# ===============================


report["Previous Score"] = (

    report["Previous Score"]

    .round(2)

)


report["Future Score"] = (

    report["Future Score"]

    .round(2)

)


report["Score Change"] = (

    report["Score Change"]

    .round(2)

)



# ===============================
# PERFORMANCE MOVEMENT
# ===============================


def performance_change(value):

    if value > 0:

        return "Improved"

    elif value < 0:

        return "Declined"

    else:

        return "No Change"



report["Performance Movement"] = (

    report["Score Change"]

    .apply(performance_change)

)



# ===============================
# ADD CATEGORY LABELS
# ===============================


category_map = {

    0: "Low",

    1: "Medium",

    2: "High"

}


report["Future Category"] = (

    report["Future Category"]

    .map(category_map)

    .fillna(report["Future Category"])

)



# ===============================
# SAVE EXCEL REPORT
# ===============================


report.to_excel(

    OUTPUT_FILE,

    index=False

)



print("""
====================================
EXCEL CREATED SUCCESSFULLY

File:
results/employee_future_explanation.xlsx


Contains:

✓ Employee ID
✓ Previous Performance Score
✓ Previous Performance Band
✓ Future Performance Score
✓ Future Performance Category
✓ Score Change
✓ Performance Movement

====================================
""")