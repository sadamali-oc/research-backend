import pandas as pd
import os


# =====================================
# 1. Load Engineered Behavior Dataset
# =====================================

input_path = os.path.join(
    "dataset",
    "processed",
    "engineered_behavior.csv"
)


df = pd.read_csv(input_path)


print("Engineered dataset loaded")
print(df.head())



# =====================================
# 2. Behavioral Classification Function
# =====================================


def classify_behavior(score):

    if score < 40:
        return "Low"

    elif score < 70:
        return "Moderate"

    else:
        return "High"



# =====================================
# 3. Create Behavioral Levels
# =====================================


df["Productivity_Level"] = (
    df["Productivity"]
    .apply(classify_behavior)
)


df["Communication_Level"] = (
    df["Communication"]
    .apply(classify_behavior)
)


df["Leadership_Level"] = (
    df["Leadership"]
    .apply(classify_behavior)
)


df["Learning_Level"] = (
    df["Learning"]
    .apply(classify_behavior)
)


df["Collaboration_Level"] = (
    df["Collaboration"]
    .apply(classify_behavior)
)



# =====================================
# 4. Calculate Overall Behavior Score
# =====================================


df["Overall_Behavior_Score"] = (

    df[
        [
            "Productivity",
            "Communication",
            "Leadership",
            "Learning",
            "Collaboration"
        ]
    ]
    .mean(axis=1)

)



# =====================================
# 5. Overall Behavior Category
# =====================================


df["Overall_Behavior_Level"] = (

    df["Overall_Behavior_Score"]
    .apply(classify_behavior)

)



# =====================================
# 6. Save Scored Dataset
# =====================================


output_path = os.path.join(
    "dataset",
    "processed",
    "behavior_scored.csv"
)



os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)



df.to_csv(
    output_path,
    index=False
)



print("\n================================")
print("Behavior Scoring Completed")
print("================================")


print(df.head())


print("\nSaved File:")
print(output_path)
