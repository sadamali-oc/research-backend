# ======================================================
# RANDOM FOREST TRAINING
# Employee Future Performance Prediction
# ======================================================

import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_FILE = (
    BASE_DIR /
    "data" /
    "performance_future_features.xlsx"
)


MODEL_FOLDER = (
    BASE_DIR /
    "models"
)

RESULT_FOLDER = (
    BASE_DIR /
    "results"
)


MODEL_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)



REGRESSOR_FILE = (
    MODEL_FOLDER /
    "random_forest_regressor.pkl"
)


CLASSIFIER_FILE = (
    MODEL_FOLDER /
    "random_forest_classifier.pkl"
)



# ======================================================
# LOAD DATA
# ======================================================

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



# ======================================================
# PREPARE DATA
# ======================================================

def prepare_data(df):

    print("\nPreparing features...")


    features = joblib.load(
        RESULT_FOLDER /
        "future_feature_names.pkl"
    )


    X = df[features]


    y_score = df[
        "Future_Performance_Score"
    ]


    y_band = df[
        "Future_Performance_Category"
    ]


    groups = df[
        "Employee ID"
    ]


    return X, y_score, y_band, groups



# ======================================================
# TRAIN REGRESSOR
# ======================================================

def train_regressor(
        X_train,
        X_test,
        y_train,
        y_test
):

    print(
        "\nTraining Random Forest Regressor..."
    )


    model = RandomForestRegressor(

        n_estimators=200,

        random_state=42,

        n_jobs=-1

    )


    model.fit(

        X_train,

        y_train

    )


    prediction = model.predict(
        X_test
    )


    mae = mean_absolute_error(

        y_test,

        prediction

    )


    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            prediction

        )

    )


    r2 = r2_score(

        y_test,

        prediction

    )


    print("\nRegression Results")

    print(
        "MAE:",
        round(mae,3)
    )

    print(
        "RMSE:",
        round(rmse,3)
    )

    print(
        "R2:",
        round(r2,3)
    )


    joblib.dump(

        model,

        REGRESSOR_FILE

    )


    return model



# ======================================================
# TRAIN CLASSIFIER
# ======================================================

def train_classifier(
        X_train,
        X_test,
        y_train,
        y_test
):

    print(
        "\nTraining Random Forest Classifier..."
    )


    model = RandomForestClassifier(

        n_estimators=200,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1

    )


    model.fit(

        X_train,

        y_train

    )


    prediction = model.predict(

        X_test

    )


    accuracy = accuracy_score(

        y_test,

        prediction

    )


    precision = precision_score(

        y_test,

        prediction,

        average="weighted"

    )


    recall = recall_score(

        y_test,

        prediction,

        average="weighted"

    )


    f1 = f1_score(

        y_test,

        prediction,

        average="weighted"

    )


    print("\nClassification Results")

    print(
        "Accuracy:",
        round(accuracy,3)
    )

    print(
        "Precision:",
        round(precision,3)
    )

    print(
        "Recall:",
        round(recall,3)
    )

    print(
        "F1 Score:",
        round(f1,3)
    )


    print(
        classification_report(
            y_test,
            prediction
        )
    )


    joblib.dump(

        model,

        CLASSIFIER_FILE

    )


    return model



# ======================================================
# MAIN
# ======================================================

def main():

    print(
"""
====================================
RANDOM FOREST MODEL TRAINING
====================================
"""
    )


    df = load_data()


    X, y_score, y_band, groups = prepare_data(df)



    # Employee-based split
    splitter = GroupShuffleSplit(

        test_size=0.2,

        random_state=42

    )


    train_index, test_index = next(

        splitter.split(
            X,
            y_score,
            groups
        )

    )


    X_train = X.iloc[train_index]

    X_test = X.iloc[test_index]


    score_train = y_score.iloc[train_index]

    score_test = y_score.iloc[test_index]


    band_train = y_band.iloc[train_index]

    band_test = y_band.iloc[test_index]



    print(
        "\nTraining employees:",
        groups.iloc[train_index].nunique()
    )


    print(
        "Testing employees:",
        groups.iloc[test_index].nunique()
    )



    train_regressor(

        X_train,
        X_test,
        score_train,
        score_test

    )


    train_classifier(

        X_train,
        X_test,
        band_train,
        band_test

    )


    print(
        "\nModels saved successfully"
    )



if __name__ == "__main__":

    main()