import pandas as pd
import os


# ==============================
# Load Dataset
# ==============================

input_path = os.path.join(
    "dataset",
    "raw",
    "Data Set.csv"
)


df = pd.read_csv(
    input_path,
    header=1
)


print("Dataset loaded successfully")
print("Original Shape:", df.shape)



# ==============================
# Clean Column Names
# ==============================

# Remove extra spaces from column names

df.columns = df.columns.str.strip()


# Remove unnamed columns

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


print("\nAvailable Columns:")
print(df.columns.tolist())



# ==============================
# Select Behavior Pattern Columns
# ==============================

behavior_columns = [

    'Punctuality (1-5)',
    'Adherence Level',
    'Response Time Level',
    'No. of Meetings Attended',
    'No. of Subordinates',
    'Decision Contribution (1-5)',
    'Learning Hours / Month',
    'Type of Learning',
    'Team Engagement Frequency',
    'Completed Storypoint Ratio'

]


# Create behavior dataset

behavior_df = df[behavior_columns].copy()


print("\nBehavior Dataset Created")
print("Shape:", behavior_df.shape)



# ==============================
# Handle Missing Values
# ==============================

behavior_df = behavior_df.fillna(0)



# ==============================
# Remove Duplicate Rows
# ==============================

behavior_df = behavior_df.drop_duplicates()



print("\nAfter Cleaning:")
print(behavior_df.head())

print("\nDataset Information:")
print(behavior_df.info())



# ==============================
# Save Processed Dataset
# ==============================


output_path = os.path.join(
    "dataset",
    "processed",
    "behaviour_processed.csv"
)


# Create processed folder if not exists

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)



behavior_df.to_csv(
    output_path,
    index=False
)


print("\n====================================")
print("Preprocessing completed successfully")
print("Saved file:")
print(output_path)
print("====================================")
