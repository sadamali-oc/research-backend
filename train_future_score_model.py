import time
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)


# ==========================================
# CONFIGURATION
# ==========================================

DATA_FILE = Path(
    "data/performance_future_features.xlsx"
)

RESULT_FOLDER = Path(
    "results"
)

RESULT_FOLDER.mkdir(
    exist_ok=True
)


MODEL_FILE = RESULT_FOLDER / "future_score_model.pkl"

METRIC_FILE = RESULT_FOLDER / "future_score_metrics.xlsx"

FEATURE_FILE = RESULT_FOLDER / "score_feature_names.pkl"



# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("\nLoading dataset...")

    df = pd.read_excel(
        DATA_FILE
    )

    print(
        "Dataset:",
        df.shape
    )

    return df



# ==========================================
# PREPARE DATA
# ==========================================

def prepare_data(df):

    print(
        "\nPreparing features..."
    )


    drop_columns = [

        "Row_ID",

        "Future_Performance_Score",

        "Future_Performance_Category"

    ]


    X = df.drop(

        columns=[

            c for c in drop_columns

            if c in df.columns

        ]

    )


    y = df[

        "Future_Performance_Score"

    ]



    # Handle missing values

    X = X.fillna(

        X.median()

    )



    joblib.dump(

        X.columns.tolist(),

        FEATURE_FILE

    )


    return X, y




# ==========================================
# TRAIN MODEL
# ==========================================

def main():


    print(
"""
====================================
FUTURE PERFORMANCE SCORE MODEL
====================================
"""
    )



    df = load_data()



    X, y = prepare_data(
        df
    )



    print(
        "\nTarget:"
    )

    print(
        y.describe()
    )



    # ==============================
    # TRAIN TEST SPLIT
    # ==============================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )



    print(

        "\nTraining:",

        X_train.shape

    )



    print(

        "Testing:",

        X_test.shape

    )




    # ==============================
    # RANDOM FOREST REGRESSOR
    # ==============================


    model = RandomForestRegressor(

        n_estimators=300,

        max_depth=None,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1

    )



    start = time.time()



    model.fit(

        X_train,

        y_train

    )



    training_time = time.time() - start




    # ==============================
    # PREDICTION
    # ==============================


    prediction = model.predict(

        X_test

    )




    # ==============================
    # EVALUATION
    # ==============================


    mae = mean_absolute_error(

        y_test,

        prediction

    )



    rmse = root_mean_squared_error(

        y_test,

        prediction

    )



    r2 = r2_score(

        y_test,

        prediction

    )



    cv_scores = cross_val_score(

        model,

        X,

        y,

        cv=5,

        scoring="r2"

    )



    cv_mean = cv_scores.mean()




    print(
"""
MODEL PERFORMANCE
"""
    )



    print(

        "MAE:",

        round(mae,4)

    )


    print(

        "RMSE:",

        round(rmse,4)

    )


    print(

        "R2:",

        round(r2,4)

    )


    print(

        "CV R2:",

        round(cv_mean,4)

    )


    print(

        "Training Time:",

        round(training_time,4)

    )





    # ==============================
    # SAVE METRICS
    # ==============================


    metrics = pd.DataFrame({

        "Metric":[

            "MAE",

            "RMSE",

            "R2",

            "CV R2",

            "Training Time"

        ],


        "Value":[

            mae,

            rmse,

            r2,

            cv_mean,

            training_time

        ]

    })



    metrics.to_excel(

        METRIC_FILE,

        index=False

    )




    # ==============================
    # SAVE MODEL
    # ==============================


    joblib.dump(

        model,

        MODEL_FILE

    )



    print(
"""
====================================
MODEL SAVED SUCCESSFULLY

Model:
results/future_score_model.pkl

Metrics:
results/future_score_metrics.xlsx

Features:
results/score_feature_names.pkl

====================================
"""
    )




if __name__ == "__main__":

    main()