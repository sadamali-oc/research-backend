import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


print("\n====================================")
print("ROLE BASED EMPLOYEE PERFORMANCE PREPROCESSING")
print("====================================\n")


# =====================================================
# 1. LOAD DATASET
# =====================================================

file_path = "data/dataset.xlsx"

print("Loading dataset...")


df = pd.read_excel(
    file_path,
    header=1
)


df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


df = df.dropna(
    axis=0,
    how="all"
)


df = df[
    df["Employee ID"].notna()
]


df.reset_index(
    drop=True,
    inplace=True
)


print("Dataset loaded")
print("Shape:", df.shape)



# =====================================================
# 2. SAVE EMPLOYEE INFORMATION
# =====================================================


os.makedirs(
    "data",
    exist_ok=True
)


employee_info = pd.DataFrame({

    "Employee ID": df["Employee ID"],

    "Row_ID": range(len(df))

})


employee_info.to_excel(
    "data/employee_information.xlsx",
    index=False
)


print("Employee information saved")



# =====================================================
# 3. HANDLE MISSING VALUES
# =====================================================


for col in df.columns:

    if pd.api.types.is_numeric_dtype(df[col]):

        df[col] = df[col].fillna(
            df[col].median()
        )

    else:

        df[col] = df[col].fillna(
            "Unknown"
        )


print("Missing values handled")



# =====================================================
# 4. KPI EXTRACTION
# =====================================================


print("\nExtracting KPI metrics...")


def get_metric(row, metric):

    for i in range(1,8):

        name_col = f"Metric {i} Name"

        if i >= 6:

            value_col = f"Metric {i} Value (Scale)"

        else:

            value_col = f"Metric {i} Value"


        if str(row[name_col]).strip() == metric:

            return row[value_col]


    return 0



metrics = [

"tasks_assigned",
"tasks_completed",
"tasks_on_time",

"bug_count",
"bugs_fixed_count",

"code_quality (1-10)",
"rework_count",

"total_test_cases_executed",
"total_pass_test_cases",

"total_bugs_detected",
"defect_reopened_count",

"document_quality (1-5)",

"total_deployments",
"successful_deployments",

"automated_pipeline_stages",
"total_pipeline_stages",

"system_downtime (hrs)",

"milestone_completion_rate",
"stakeholder_satisfaction(1-5)",

"sprint_velocity"

]



for metric in metrics:

    df[metric] = df.apply(

        lambda row:get_metric(row,metric),

        axis=1

    )


print("KPI extraction completed")



# =====================================================
# 5. CREATE KPI FEATURES
# =====================================================


print("\nCreating KPI features...")


def percentage(a,b):

    if b == 0:

        return 0

    return min((a/b)*100,100)



df["Task_Completion_Rate"] = df.apply(

lambda x: percentage(
x["tasks_completed"],
x["tasks_assigned"]
),

axis=1

)



df["On_Time_Delivery_Rate"] = df.apply(

lambda x: percentage(
x["tasks_on_time"],
x["tasks_completed"]
),

axis=1

)



def bug_rate(row):

    if row["bug_count"] == 0:

        return 100

    return min(

        row["bugs_fixed_count"]
        /
        row["bug_count"]
        *
        100,

        100

    )



df["Bug_Resolution_Rate"] = df.apply(
bug_rate,
axis=1
)



df["Code_Quality_Score"] = (

df["code_quality (1-10)"]
/
10
*
100

)



df["Rework_Score"] = (

100 -
(df["rework_count"]/10*100)

).clip(0,100)



df["Test_Pass_Rate"] = df.apply(

lambda x:percentage(
x["total_pass_test_cases"],
x["total_test_cases_executed"]
),

axis=1

)



df["Defect_Quality_Score"] = (

100 -

(
df["defect_reopened_count"]
/
df["total_bugs_detected"].replace(0,1)
*
100
)

).clip(0,100)



df["Documentation_Quality"] = (

df["document_quality (1-5)"]
/
5
*
100

)



df["Deployment_Success_Rate"] = df.apply(

lambda x:percentage(
x["successful_deployments"],
x["total_deployments"]
),

axis=1

)



df["Pipeline_Automation_Rate"] = df.apply(

lambda x:percentage(
x["automated_pipeline_stages"],
x["total_pipeline_stages"]
),

axis=1

)



df["System_Reliability"] = (

100 -

(df["system_downtime (hrs)"]/50*100)

).clip(0,100)



df["Milestone_Completion"] = (

df["milestone_completion_rate"]
/
10
*
100

)



df["Stakeholder_Satisfaction"] = (

df["stakeholder_satisfaction(1-5)"]
/
5
*
100

)



df["Sprint_Velocity"] = (

df["sprint_velocity"]
/
df["sprint_velocity"].max()
*
100

)



print("KPI features created")


# =====================================================
# 6. PERFORMANCE SCORE (STRONG KPI FACTORS ONLY)
# =====================================================

print("\nCalculating KPI based performance score...")


df["Performance_Score"] = (

    0.20 * df["Task_Completion_Rate"] +

    0.15 * df["On_Time_Delivery_Rate"] +

    0.15 * df["Bug_Resolution_Rate"] +

    0.10 * df["Code_Quality_Score"] +

    0.10 * df["Test_Pass_Rate"] +

    0.10 * df["Defect_Quality_Score"] +

    0.10 * df["System_Reliability"] +

    0.05 * df["Deployment_Success_Rate"] +

    0.03 * df["Sprint_Velocity"] +

    0.02 * df["Rework_Score"]

)


print(df["Performance_Score"].describe())


# =====================================================
# 7. CREATE TARGET LABEL
# =====================================================

print("\nCreating performance categories...")


low_threshold = df["Performance_Score"].quantile(0.33)

high_threshold = df["Performance_Score"].quantile(0.67)



def create_label(score):

    if score <= low_threshold:
        return "Low Performance"

    elif score >= high_threshold:
        return "High Performance"

    else:
        return "Medium Performance"



df["Performance_Category"] = df["Performance_Score"].apply(create_label)



print("\nPerformance Distribution")

print(df["Performance_Category"].value_counts())


# =====================================================
# 8. SELECT FEATURES
# =====================================================

features=[

"Employee ID",

"Avg Response Time (hrs)",

"Duration (Weeks)",

"Relative Effort (Story Pts)",

"Team Size",

"Project Complexity",

"No-Pay Leave",

"Task_Completion_Rate",

"On_Time_Delivery_Rate",

"Bug_Resolution_Rate",

"Code_Quality_Score",

"Rework_Score",

"Test_Pass_Rate",

"Defect_Quality_Score",

"Documentation_Quality",

"Deployment_Success_Rate",

"Pipeline_Automation_Rate",

"System_Reliability",

"Sprint_Velocity",

"Performance_Category"

]

final = df[features].copy()



# =====================================================
# 9. ENCODING
# =====================================================


os.makedirs(
"results",
exist_ok=True
)



encoders={}



for col in ["Project Complexity"]:


    le = LabelEncoder()


    final[col] = le.fit_transform(
        final[col].astype(str)
    )


    encoders[col]=le



joblib.dump(
encoders,
"results/feature_encoders.pkl"
)



target_mapping={

"Low Performance":0,

"Medium Performance":1,

"High Performance":2

}



final["Performance_Category"] = (

final["Performance_Category"]
.map(target_mapping)

)



joblib.dump(
target_mapping,
"results/label_encoder.pkl"
)



print("\nTarget Encoding")

print(target_mapping)



# =====================================================
# 10. SAVE FINAL DATASET
# =====================================================


final.insert(

0,

"Row_ID",

range(len(final))

)



final.to_excel(

"data/performance_features.xlsx",

index=False

)



print("\nFinal Dataset Saved")

print(final.shape)

print(final.head())



# =====================================================
# 11. VISUALIZATION
# =====================================================


plt.figure(
figsize=(7,5)
)


df["Performance_Category"].value_counts().plot(
kind="bar"
)


plt.title(
"Role Based Employee Performance Distribution"
)


plt.xlabel(
"Performance Category"
)


plt.ylabel(
"Employees"
)


plt.tight_layout()


plt.savefig(

"results/performance_distribution.png",

dpi=300

)


plt.close()



print("\nVisualization saved")



print("\n====================================")
print("ROLE BASED PREPROCESSING COMPLETE")
print("====================================")