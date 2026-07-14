import pandas as pd
import os
import joblib


from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# =====================================================
# 1. Load Datasets
# =====================================================


behavior_path = r"C:\Users\Mihi\Desktop\Behaviour Pattern Identification Model\dataset\processed\behavior_scored.csv"

raw_behavior_path = r"C:\Users\Mihi\Desktop\Behaviour Pattern Identification Model\dataset\processed\behaviour_processed.csv"



behavior_df = pd.read_csv(behavior_path)

original_df = pd.read_csv(raw_behavior_path)



print("Datasets Loaded Successfully")


print("Behavior scored rows:", len(behavior_df))

print("Original behavior rows:", len(original_df))



# =====================================================
# 2. Encode Type of Learning
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



original_df["Type_of_Learning"] = (

    original_df["Type of Learning"]
    .map(learning_mapping)

)



# =====================================================
# 3. Create Input Features (X)
# =====================================================


X = pd.DataFrame()



X["Punctuality"] = original_df["Punctuality (1-5)"]


X["Adherence_Level"] = original_df["Adherence Level"]


X["Response_Time_Level"] = original_df["Response Time Level"]


X["Meetings_Attended"] = original_df["No. of Meetings Attended"]


X["No_of_Subordinates"] = original_df["No. of Subordinates"]


X["Decision_Contribution"] = original_df["Decision Contribution (1-5)"]


X["Learning_Hours"] = original_df["Learning Hours / Month"]


X["Type_of_Learning"] = original_df["Type_of_Learning"]


X["Team_Engagement"] = original_df["Team Engagement Frequency"]


X["Storypoint_Ratio"] = original_df["Completed Storypoint Ratio"]




# =====================================================
# 4. Create Output Variables (Y)
# =====================================================


Y = behavior_df[

    [

        "Productivity",

        "Communication",

        "Leadership",

        "Learning",

        "Collaboration"

    ]

]



print("\nBefore Cleaning")

print("X shape:", X.shape)

print("Y shape:", Y.shape)



# =====================================================
# 5. Remove Missing Values
# =====================================================


combined_data = pd.concat(

    [X, Y],

    axis=1

)



print("\nMissing Values:")

print(combined_data.isnull().sum())



# Remove rows containing NaN

combined_data = combined_data.dropna()



# Separate X and Y again

X = combined_data[

    [

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

]



Y = combined_data[

    [

        "Productivity",

        "Communication",

        "Leadership",

        "Learning",

        "Collaboration"

    ]

]



print("\nAfter Cleaning")

print("X shape:", X.shape)

print("Y shape:", Y.shape)



# =====================================================
# 6. Split Training and Testing Data
# =====================================================


X_train, X_test, Y_train, Y_test = train_test_split(

    X,

    Y,

    test_size=0.2,

    random_state=42

)



print("\nTraining data:", X_train.shape)

print("Testing data:", X_test.shape)




# =====================================================
# 7. Create Multi Output Random Forest
# =====================================================


rf_model = MultiOutputRegressor(

    RandomForestRegressor(

        n_estimators=100,

        random_state=42

    )

)



# =====================================================
# 8. Train Model
# =====================================================


print("\nTraining Random Forest Model...")


rf_model.fit(

    X_train,

    Y_train

)



print("Training Completed Successfully")




# =====================================================
# 9. Prediction
# =====================================================


Y_pred = rf_model.predict(X_test)



prediction_df = pd.DataFrame(

    Y_pred,

    columns=[

        "Predicted_Productivity",

        "Predicted_Communication",

        "Predicted_Leadership",

        "Predicted_Learning",

        "Predicted_Collaboration"

    ]

)



results_path = r"C:\Users\Mihi\Desktop\Behaviour Pattern Identification Model\results"



os.makedirs(

    results_path,

    exist_ok=True

)



prediction_df.to_csv(

    os.path.join(results_path, "predictions.csv"),

    index=False

)



print("\nPrediction file saved")




# =====================================================
# 10. Evaluation
# =====================================================


evaluation_results = []



for i, column in enumerate(Y.columns):


    mae = mean_absolute_error(

        Y_test.iloc[:, i],

        Y_pred[:, i]

    )


    mse = mean_squared_error(

        Y_test.iloc[:, i],

        Y_pred[:, i]

    )


    r2 = r2_score(

        Y_test.iloc[:, i],

        Y_pred[:, i]

    )


    evaluation_results.append(

        [

            column,

            mae,

            mse,

            r2

        ]

    )




evaluation_df = pd.DataFrame(

    evaluation_results,

    columns=[

        "Behavior",

        "MAE",

        "MSE",

        "R2_Score"

    ]

)



evaluation_df.to_csv(

    os.path.join(results_path, "evaluation_metrics.csv"),

    index=False

)



print("\nEvaluation Results")

print(evaluation_df)




# =====================================================
# 11. Save Model
# =====================================================


model_folder = r"C:\Users\Mihi\Desktop\Behaviour Pattern Identification Model\models"



os.makedirs(

    model_folder,

    exist_ok=True

)



model_path = os.path.join(

    model_folder,

    "behavior_random_forest.pkl"

)



joblib.dump(

    rf_model,

    model_path

)



print("\nModel Saved Successfully")

print(model_path)



print("\n================================")

print("Random Forest Stage Completed")

print("================================")