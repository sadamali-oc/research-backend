import pandas as pd
import os
import joblib


# =====================================================
# 1. Load Trained Random Forest Model
# =====================================================

model_path = os.path.join(
    "models",
    "behavior_random_forest.pkl"
)


model = joblib.load(model_path)


print("Random Forest Model Loaded Successfully")



# =====================================================
# 2. Load New Employee Data
# =====================================================
input_path = os.path.join(
    "dataset",
    "prediction",
    "new_employee_data.csv"
)


new_employee_df = pd.read_csv(input_path)


print("\nNew Employee Data Loaded Successfully")

print(new_employee_df.head())



# =====================================================
# 3. Encode Type of Learning
# =====================================================


learning_mapping = {

    "None observed": 0,

    "Self-learning": 1,

    "Knowledge sharing": 2,

    "On-the-job learning": 3,

    "Formal training": 4,

    "Online courses / certifications": 5,

    "Research & innovation": 6

}



new_employee_df["Type_of_Learning"] = (

    new_employee_df["Type of Learning"]
    .map(learning_mapping)
    .fillna(0)

)



# =====================================================
# 4. Prepare Input Features
# =====================================================


X_new = pd.DataFrame()



X_new["Punctuality"] = (
    new_employee_df["Punctuality (1-5)"]
)


X_new["Adherence_Level"] = (
    new_employee_df["Adherence Level"]
)


X_new["Response_Time_Level"] = (
    new_employee_df["Response Time Level"]
)


X_new["Meetings_Attended"] = (
    new_employee_df["No. of Meetings Attended"]
)


X_new["No_of_Subordinates"] = (
    new_employee_df["No. of Subordinates"]
)


X_new["Decision_Contribution"] = (
    new_employee_df["Decision Contribution (1-5)"]
)


X_new["Learning_Hours"] = (
    new_employee_df["Learning Hours / Month"]
)


X_new["Type_of_Learning"] = (
    new_employee_df["Type_of_Learning"]
)


X_new["Team_Engagement"] = (
    new_employee_df["Team Engagement Frequency"]
)


X_new["Storypoint_Ratio"] = (
    new_employee_df["Completed Storypoint Ratio"]
)



print("\nPrepared Input Data")

print(X_new.head())



# =====================================================
# 5. Generate Predictions
# =====================================================


print("\nGenerating Behavioral Predictions...")


predictions = model.predict(X_new)



# =====================================================
# 6. Create Prediction Result Dataset
# =====================================================


prediction_result = pd.DataFrame(

    predictions,

    columns=[

        "Predicted_Productivity",

        "Predicted_Communication",

        "Predicted_Leadership",

        "Predicted_Learning",

        "Predicted_Collaboration"

    ]

)



# Round values

prediction_result = prediction_result.round(2)



# =====================================================
# 7. Calculate Overall Behavior Score
# =====================================================


prediction_result["Overall_Behavior_Score"] = (

    prediction_result[

        [

            "Predicted_Productivity",

            "Predicted_Communication",

            "Predicted_Leadership",

            "Predicted_Learning",

            "Predicted_Collaboration"

        ]

    ]

    .mean(axis=1)

    .round(2)

)



# =====================================================
# 8. Assign Overall Behavior Level
# =====================================================


def classify_behavior(score):

    if score < 40:

        return "Low"

    elif score < 70:

        return "Moderate"

    else:

        return "High"



prediction_result["Overall_Behavior_Level"] = (

    prediction_result["Overall_Behavior_Score"]

    .apply(classify_behavior)

)



# =====================================================
# 9. Save Prediction Results
# =====================================================

output_path = os.path.join(
    "results",
    "predicted_behavior.csv"
)


os.makedirs(

    os.path.dirname(output_path),

    exist_ok=True

)



prediction_result.to_csv(

    output_path,

    index=False

)



print("\n====================================")

print("Prediction Completed Successfully")

print("====================================")


print("\nPredicted Behavioral Patterns:")

print(prediction_result)



print("\nSaved File:")

print(output_path)