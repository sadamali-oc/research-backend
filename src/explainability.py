import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os



# =====================================================
# 1. Load Trained XGBoost Model
# =====================================================


model_path = os.path.join(

    "models",

    "xgboost_behavior_model.pkl"

)



model = joblib.load(model_path)



print("XGBoost model loaded successfully")





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
# 3. Extract XGBoost Feature Importance
# =====================================================


importance_values = []



# MultiOutputRegressor contains multiple XGBRegressor models

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





# =====================================================
# 4. Create Explainability Dataset
# =====================================================


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





print("\nXGBoost Feature Importance Ranking")


print(importance_df)





# =====================================================
# 5. Save Importance Results
# =====================================================


output_folder = os.path.join(

    "Results",

    "explainability"

)



os.makedirs(

    output_folder,

    exist_ok=True

)




importance_df.to_csv(

    os.path.join(

        output_folder,

        "xgboost_feature_importance.csv"

    ),

    index=False

)





# =====================================================
# 6. Create Feature Importance Chart
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

    "XGBoost Feature Importance"

)



plt.gca().invert_yaxis()



plt.tight_layout()




plt.savefig(

    os.path.join(

        output_folder,

        "xgboost_feature_importance_chart.png"

    )

)



plt.close()





# =====================================================
# Completion Message
# =====================================================


print("\n===================================")

print("XGBoost Explainability Completed Successfully")

print("===================================")



print("\nSaved Location:")

print(output_folder)