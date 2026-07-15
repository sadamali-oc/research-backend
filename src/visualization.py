import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# =====================================================
# 1. Load Prediction Dataset
# =====================================================

input_path = r"results\predicted_behavior.csv"


df = pd.read_csv(input_path)


print("Prediction dataset loaded successfully")

print(df.head())


# Add employee IDs
df["Employee_ID"] = range(1, len(df) + 1)



# =====================================================
# 2. Create Visualization Folder
# =====================================================


output_folder = r"results\visualizations"


os.makedirs(
    output_folder,
    exist_ok=True
)



# =====================================================
# 3. Average Behavioral Pattern Visualization
# =====================================================


behavior_columns = [

    "Predicted_Productivity",

    "Predicted_Communication",

    "Predicted_Leadership",

    "Predicted_Learning",

    "Predicted_Collaboration"

]


average_scores = df[behavior_columns].mean()



plt.figure(figsize=(8,5))


plt.bar(

    [
        "Productivity",
        "Communication",
        "Leadership",
        "Learning",
        "Collaboration"
    ],

    average_scores.values

)


plt.xlabel("Behavior Categories")

plt.ylabel("Average Percentage (%)")


plt.title(
    "Average Employee Behavioral Pattern"
)


plt.xticks(rotation=45)


plt.ylim(0,100)


plt.tight_layout()



plt.savefig(

    os.path.join(

        output_folder,

        "average_behavior_pattern.png"

    )

)


plt.close()



print("Average behavior chart created")



# =====================================================
# 4. Overall Behavior Level Distribution
# =====================================================


level_count = df["Overall_Behavior_Level"].value_counts()



plt.figure(figsize=(6,5))


plt.bar(

    level_count.index,

    level_count.values

)


plt.xlabel("Behavior Level")


plt.ylabel("Number of Employees")


plt.title(

    "Overall Employee Behavioral Level Distribution"

)


plt.tight_layout()



plt.savefig(

    os.path.join(

        output_folder,

        "behavior_level_distribution.png"

    )

)


plt.close()



print("Behavior distribution chart created")



# =====================================================
# 5. Employee Selection for Radar Chart
# =====================================================


print("\nAvailable Employees:")

print(
    "Employee 1 to Employee",
    len(df)
)



while True:

    try:

        employee_number = int(
            input(
                "\nEnter Employee Number for Radar Chart: "
            )
        )


        if employee_number >= 1 and employee_number <= len(df):

            break

        else:

            print(
                "Invalid employee number. Try again."
            )


    except:

        print(
            "Please enter a valid number."
        )




# Select employee


employee_data = df.loc[

    df["Employee_ID"] == employee_number

].iloc[0]



# =====================================================
# 6. Radar Chart Creation
# =====================================================



values = [

    employee_data["Predicted_Productivity"],

    employee_data["Predicted_Communication"],

    employee_data["Predicted_Leadership"],

    employee_data["Predicted_Learning"],

    employee_data["Predicted_Collaboration"]

]



categories = [

    "Productivity",

    "Communication",

    "Leadership",

    "Learning",

    "Collaboration"

]



# Close radar shape

values += values[:1]


angles = np.linspace(

    0,

    2*np.pi,

    len(categories)+1

)



plt.figure(figsize=(7,7))


ax = plt.subplot(

    111,

    polar=True

)



ax.plot(

    angles,

    values,

    linewidth=2

)



ax.fill(

    angles,

    values,

    alpha=0.25

)



ax.set_xticks(

    angles[:-1]

)



ax.set_xticklabels(

    categories

)



ax.set_ylim(

    0,

    100

)



plt.title(

    f"Employee {employee_number} Behavioral Profile"

)



plt.tight_layout()



radar_path = os.path.join(

    output_folder,

    f"employee_{employee_number}_radar_chart.png"

)



plt.savefig(

    radar_path

)



plt.close()



print(
    f"Radar chart created for Employee {employee_number}"
)



# =====================================================
# 7. Save Average Scores
# =====================================================


average_df = pd.DataFrame(

    {

        "Behavior":

        [

            "Productivity",

            "Communication",

            "Leadership",

            "Learning",

            "Collaboration"

        ],


        "Average_Score":

        average_scores.values

    }

)



average_df.to_csv(

    os.path.join(

        output_folder,

        "average_behavior_scores.csv"

    ),

    index=False

)



# =====================================================
# 8. Completion Message
# =====================================================


print("\n====================================")

print("Visualization Completed Successfully")

print("====================================")


print("\nGenerated Radar Chart:")

print(radar_path)


print("\nAll files saved inside:")

print(output_folder)