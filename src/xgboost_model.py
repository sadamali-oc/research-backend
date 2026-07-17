import pandas as pd
import os
import joblib


from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor



# =====================================================
# Project Path (Relative Path)
# =====================================================


project_path = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



dataset_path = os.path.join(
    project_path,
    "dataset",
    "processed",
    "behavior_scored.csv"
)


original_path = os.path.join(
    project_path,
    "dataset",
    "processed",
    "behaviour_processed.csv"
)



results_path = os.path.join(
    project_path,
    "Results"
)


models_path = os.path.join(
    project_path,
    "models"
)



os.makedirs(
    results_path,
    exist_ok=True
)


os.makedirs(
    models_path,
    exist_ok=True
)



# =====================================================
# Load Dataset
# =====================================================


df = pd.read_csv(dataset_path)


original_df = pd.read_csv(original_path)


print("Dataset Loaded Successfully")



# =====================================================
# Encode Learning Type
# =====================================================


learning_mapping = {


    "None observed":0,

    "Self-learning":1,

    "Knowledge sharing":2,

    "On-the-job learning":3,

    "Formal training":4,

    "Online courses / certifications":5,

    "Research & innovation":6

}




# =====================================================
# Prepare Input Features
# =====================================================


X = pd.DataFrame()



X["Punctuality"] = (
    original_df["Punctuality (1-5)"]
)



X["Adherence_Level"] = (
    original_df["Adherence Level"]
)



X["Response_Time_Level"] = (
    original_df["Response Time Level"]
)



X["Meetings_Attended"] = (
    original_df["No. of Meetings Attended"]
)



X["No_of_Subordinates"] = (
    original_df["No. of Subordinates"]
)



X["Decision_Contribution"] = (
    original_df["Decision Contribution (1-5)"]
)



X["Learning_Hours"] = (
    original_df["Learning Hours / Month"]
)



X["Type_of_Learning"] = (

    original_df["Type of Learning"]
    .map(learning_mapping)

)



X["Team_Engagement"] = (
    original_df["Team Engagement Frequency"]
)



X["Storypoint_Ratio"] = (
    original_df["Completed Storypoint Ratio"]
)




# =====================================================
# Output Targets
# =====================================================


Y = df[

    [

    "Productivity",

    "Communication",

    "Leadership",

    "Learning",

    "Collaboration"

    ]

]



# Remove missing values


combined = pd.concat(
    [
        X,
        Y
    ],
    axis=1
)



combined.dropna(
    inplace=True
)



X = combined[X.columns]


Y = combined[Y.columns]



print("\nData Prepared")

print("Input Shape:",X.shape)

print("Output Shape:",Y.shape)



# =====================================================
# Train Test Split
# =====================================================


X_train, X_test, Y_train, Y_test = train_test_split(

    X,

    Y,

    test_size=0.2,

    random_state=42

)



print("\nTraining Data:",X_train.shape)

print("Testing Data:",X_test.shape)




# =====================================================
# XGBoost Multi Output Regression
# =====================================================


xgb_model = MultiOutputRegressor(

    XGBRegressor(

        n_estimators=200,

        learning_rate=0.05,

        max_depth=5,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        objective="reg:squarederror"

    )

)



print("\nTraining XGBoost Model...")



xgb_model.fit(

    X_train,

    Y_train

)



print("XGBoost Training Completed")




# =====================================================
# Prediction
# =====================================================


prediction = xgb_model.predict(
    X_test
)




prediction_df = pd.DataFrame(

    prediction,

    columns=[

        "Predicted_Productivity",

        "Predicted_Communication",

        "Predicted_Leadership",

        "Predicted_Learning",

        "Predicted_Collaboration"

    ]

)



prediction_df.to_csv(

    os.path.join(

        results_path,

        "xgboost_predictions.csv"

    ),

    index=False

)



# =====================================================
# Evaluation
# =====================================================


evaluation_results=[]



for index,column in enumerate(Y.columns):


    evaluation_results.append(

        [

            column,

            mean_absolute_error(

                Y_test.iloc[:,index],

                prediction[:,index]

            ),


            mean_squared_error(

                Y_test.iloc[:,index],

                prediction[:,index]

            ),



            r2_score(

                Y_test.iloc[:,index],

                prediction[:,index]

            )

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



print("\nEvaluation Results")

print(evaluation_df)



evaluation_df.to_csv(

    os.path.join(

        results_path,

        "xgboost_evaluation_metrics.csv"

    ),

    index=False

)




# =====================================================
# Save Model
# =====================================================


model_file = os.path.join(

    models_path,

    "xgboost_behavior_model.pkl"

)



joblib.dump(

    xgb_model,

    model_file

)



print("\nXGBoost Model Saved Successfully")

print(model_file)



print("\n===================================")

print("XGBoost Multi Output Regression Completed")

print("===================================")