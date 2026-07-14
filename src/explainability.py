import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os



# =====================================================
# 1. Load Trained Model
# =====================================================


model_path = r"C:\Users\Mihi\Desktop\Behaviour Pattern Identification Model\models\behavior_random_forest.pkl"


model = joblib.load(model_path)


print("Random Forest model loaded successfully")



# =====================================================
# 2. Define Input Features
# =====================================================


features = [

    "Punctuality",

    "Adherence_Level",

    "Response_Time_Level",

    "Meetings_Attended",

    "No_of_Subordinates",

    "Decision_Contribution",

    "Learning_Hours",

    "Type_of_Learning",

    "Team_Engagement",

    "Storypoint_Ratio"

]



# =====================================================
# 3. Extract Feature Importance
# =====================================================


importance_values = []


for estimator in model.estimators_:

    importance_values.append(

        estimator.feature_importances_

    )



# Average importance from all output models

average_importance = (

    sum(importance_values)

    /

    len(importance_values)

)



importance_df = pd.DataFrame(

    {

        "Feature": features,

        "Importance": average_importance

    }

)



importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)



print("\nFeature Importance Ranking")

print(importance_df)



# =====================================================
# 4. Save Importance Result
# =====================================================


output_folder = r"C:\Users\Mihi\Desktop\Behaviour Pattern Identification Model\results\explainability"


os.makedirs(

    output_folder,

    exist_ok=True

)



importance_df.to_csv(

    os.path.join(

        output_folder,

        "feature_importance.csv"

    ),

    index=False

)



# =====================================================
# 5. Visualization
# =====================================================


plt.figure(figsize=(10,6))


plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]

)


plt.xlabel(

    "Importance Score"

)


plt.ylabel(

    "Employee Attributes"

)


plt.title(

    "Random Forest Feature Importance"

)


plt.gca().invert_yaxis()


plt.tight_layout()



plt.savefig(

    os.path.join(

        output_folder,

        "feature_importance_chart.png"

    )

)



plt.close()



print("\n===================================")

print("Explainability Completed Successfully")

print("===================================")


print("\nSaved Location:")

print(output_folder)