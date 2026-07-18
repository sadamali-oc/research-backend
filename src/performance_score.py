import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_performance_score(df):

    # ==========================
    # Normalize productivity metrics
    # ==========================
    scaler = MinMaxScaler()

    productivity_features = [
        "Deadline Adherence Rate (%)",
        "Completed Storypoint Ratio",
        "Relative Effort (Story Pts)"
    ]

    df[productivity_features] = scaler.fit_transform(
        df[productivity_features]
    )


    # ==========================
    # Self Assessment Score
    # ==========================
    self_assessment = (
        df["Punctuality (1-5)"] +
        df["Problem Solving (1-5)"] +
        df["Leadership (1-5)"] +
        df["Collaboration (1-5)"] +
        df["Communication (1-5)"]
    ) / 25


    # ==========================
    # Work Productivity Score
    # ==========================
    productivity = (
        df["Deadline Adherence Rate (%)"] +
        df["Completed Storypoint Ratio"] +
        df["Relative Effort (Story Pts)"]
    ) / 3


    # ==========================
    # Behaviour Score
    # ==========================
    behaviour = (
        (df["Decision Contribution (1-5)"] / 5) +
        (df["Team Engagement Frequency"] / 5) +
        (df["Language Proficiency"] / 5)
    ) / 3


    # ==========================
    # Learning & Development Score
    # ==========================
    learning = (
        df["Learning Hours / Month"] /
        df["Learning Hours / Month"].max()
    )


    # ==========================
    # Final Weighted Performance Score
    # ==========================
    df["Performance Score"] = (
        (self_assessment * 0.35) +
        (productivity * 0.30) +
        (behaviour * 0.20) +
        (learning * 0.15)
    ) * 100


    # Keep score between 0-100
    df["Performance Score"] = (
        df["Performance Score"]
        .clip(0, 100)
    )


    # ==========================
    # Performance Classification
    # ==========================
    df["Performance Category"] = pd.cut(
        df["Performance Score"],
        bins=[-1, 50, 65, 101],
        labels=[
            "Low Performer",
            "Medium Performer",
            "High Performer"
        ],
        include_lowest=True
    )


    # Convert category to string
    df["Performance Category"] = (
        df["Performance Category"]
        .astype(str)
    )


    return df