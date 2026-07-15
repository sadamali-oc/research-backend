import pandas as pd
import os


# =====================================
# 1. Load Preprocessed Dataset
# =====================================

input_path = os.path.join(
    "dataset",
    "processed",
    "behaviour_processed.csv"
)


df = pd.read_csv(input_path)


print("Dataset Loaded Successfully")
print(df.head())



# =====================================
# 2. Productivity Behavior
#
# Based on:
# - Punctuality
# - Adherence Level
#
# Both are rated 1-5
# =====================================


df["Productivity"] = (

    (
        (df["Punctuality (1-5)"] / 5) * 100
        +
        (df["Adherence Level"] / 5) * 100
    )

    / 2

)



# =====================================
# 3. Communication Behavior
#
# Based on:
# - Response Time Level
# - Meetings Attended
# =====================================


response_score = (

    df["Response Time Level"] / 5

) * 100



meeting_score = (

    df["No. of Meetings Attended"]
    /
    df["No. of Meetings Attended"].max()

) * 100



df["Communication"] = (

    response_score +
    meeting_score

) / 2



# =====================================
# 4. Leadership Behavior
#
# Based on:
# - Number of Subordinates
# - Decision Contribution
# =====================================


subordinate_score = (

    df["No. of Subordinates"]
    /
    df["No. of Subordinates"].max()

) * 100



decision_score = (

    df["Decision Contribution (1-5)"]
    /
    5

) * 100



df["Leadership"] = (

    subordinate_score +
    decision_score

) / 2



# =====================================
# 5. Learning Behavior
#
# Based on:
# - Learning Hours
# - Type of Learning
# =====================================


learning_hour_score = (

    df["Learning Hours / Month"]
    /
    df["Learning Hours / Month"].max()

) * 100



# Learning type importance mapping

learning_mapping = {

    "None observed": 0,

    "Self-learning": 60,

    "Knowledge sharing": 70,

    "On-the-job learning": 75,

    "Formal training": 85,

    "Online courses / certifications": 90,

    "Research & innovation": 100

}



df["Learning_Type_Score"] = (

    df["Type of Learning"]
    .map(learning_mapping)

)



df["Learning"] = (

    learning_hour_score +
    df["Learning_Type_Score"]

) / 2




# =====================================
# 6. Collaboration Behavior
#
# Based on:
# - Team Engagement Frequency
# - Completed Storypoint Ratio
# =====================================


engagement_score = (

    df["Team Engagement Frequency"]
    /
    df["Team Engagement Frequency"].max()

) * 100



storypoint_score = (

    df["Completed Storypoint Ratio"]

) * 100



df["Collaboration"] = (

    engagement_score +
    storypoint_score

) / 2




# =====================================
# 7. Create Final Behavioral Dataset
# =====================================


behavior_df = df[
    [
        "Productivity",
        "Communication",
        "Leadership",
        "Learning",
        "Collaboration"
    ]
]



# Round percentages

behavior_df = behavior_df.round(2)



# =====================================
# 8. Save Engineered Dataset
# =====================================


output_path = os.path.join(
    "dataset",
    "processed",
    "engineered_behavior.csv"
)



os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)



behavior_df.to_csv(
    output_path,
    index=False
)



print("\n================================")
print("Feature Engineering Completed")
print("================================")


print("\nGenerated Behavioral Patterns:")
print(behavior_df.head())


print("\nSaved File:")
print(output_path)
