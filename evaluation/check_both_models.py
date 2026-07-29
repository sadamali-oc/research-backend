# ======================================================
# FINAL MODEL VALIDATION
# Employee Future Performance Prediction
# Classification + Regression
# ======================================================


import joblib
import pandas as pd
import numpy as np

from pathlib import Path


from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)


from sklearn.metrics import (

    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,


    # Regression
    mean_absolute_error,
    mean_squared_error,
    r2_score

)



# ======================================================
# PATH CONFIGURATION
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



CLASSIFIER_FILE = (
    MODEL_FOLDER /
    "random_forest_classifier.pkl"
)



REGRESSOR_FILE = (
    MODEL_FOLDER /
    "random_forest_regressor.pkl"
)



FEATURE_FILE = (
    RESULT_FOLDER /
    "future_feature_names.pkl"
)



ENCODER_FILE = (
    MODEL_FOLDER /
    "feature_encoders.pkl"
)





# ======================================================
# LOAD DATA AND MODELS
# ======================================================


def load_resources():


    print("\nLoading dataset...")


    df = pd.read_excel(
        DATA_FILE
    )


    print(
        "Dataset Shape:",
        df.shape
    )



    print("\nLoading models...")


    classifier = joblib.load(
        CLASSIFIER_FILE
    )


    regressor = joblib.load(
        REGRESSOR_FILE
    )


    feature_names = joblib.load(
        FEATURE_FILE
    )



    print(
        "Classifier:",
        type(classifier).__name__
    )


    print(
        "Regressor:",
        type(regressor).__name__
    )


    return (
        df,
        classifier,
        regressor,
        feature_names
    )





# ======================================================
# FEATURE PREPARATION
# ======================================================


def prepare_features(
        df,
        feature_names
):


    print("\nPreparing features...")



    X = df[feature_names].copy()



    # Handle categorical columns

    categorical_columns = X.select_dtypes(
        include=["object","string"]
    ).columns



    if len(categorical_columns) > 0:


        print(
            "Encoding:",
            list(categorical_columns)
        )


        for col in categorical_columns:


            X[col] = pd.factorize(
                X[col]
            )[0]



    # Missing values

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )


    return X





# ======================================================
# CLASSIFICATION VALIDATION
# ======================================================


def validate_classifier(
        model,
        X,
        y
):


    print("\n")
    print("="*50)
    print(" RANDOM FOREST CLASSIFICATION VALIDATION ")
    print("="*50)



    X_train,X_test,y_train,y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    model.fit(
        X_train,
        y_train
    )



    prediction = model.predict(
        X_test
    )



    probability = model.predict_proba(
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


    roc = roc_auc_score(
        y_test,
        probability,
        multi_class="ovr"
    )



    print("\nTesting Results")

    print(
        "Accuracy:",
        round(accuracy,4)
    )


    print(
        "Precision:",
        round(precision,4)
    )


    print(
        "Recall:",
        round(recall,4)
    )


    print(
        "F1 Score:",
        round(f1,4)
    )


    print(
        "ROC-AUC:",
        round(roc,4)
    )



    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            prediction
        )
    )



    print(
        "Confusion Matrix"
    )


    print(
        confusion_matrix(
            y_test,
            prediction
        )
    )



    # Cross Validation

    cv = cross_val_score(

        model,

        X,

        y,

        cv=5,

        scoring="accuracy"

    )


    print(
        "\nCross Validation Accuracy:"
    )


    print(
        cv
    )


    print(
        "Average CV Accuracy:",
        round(cv.mean(),4)
    )







# ======================================================
# REGRESSION VALIDATION
# ======================================================


def validate_regressor(
        model,
        X,
        y
):


    print("\n")
    print("="*50)
    print(" RANDOM FOREST REGRESSION VALIDATION ")
    print("="*50)



    X_train,X_test,y_train,y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

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



    mse = mean_squared_error(

        y_test,

        prediction

    )


    rmse = np.sqrt(
        mse
    )


    r2 = r2_score(

        y_test,

        prediction

    )



    print("\nRegression Results")


    print(
        "MAE:",
        round(mae,4)
    )


    print(
        "MSE:",
        round(mse,4)
    )


    print(
        "RMSE:",
        round(rmse,4)
    )


    print(
        "R2 Score:",
        round(r2,4)
    )



    # Regression CV


    cv = cross_val_score(

        model,

        X,

        y,

        cv=5,

        scoring="r2"

    )



    print(
        "\nCross Validation R2:"
    )


    print(
        cv
    )


    print(
        "Average CV R2:",
        round(cv.mean(),4)
    )






# ======================================================
# MAIN
# ======================================================


def main():


    print(
"""
================================================
EMPLOYEE FUTURE PERFORMANCE MODEL VALIDATION
================================================
"""
    )



    (
        df,
        classifier,
        regressor,
        feature_names

    ) = load_resources()



    X = prepare_features(

        df,

        feature_names

    )



    y_class = df[
        "Future_Performance_Category"
    ]



    y_reg = df[
        "Future_Performance_Score"
    ]




    validate_classifier(

        classifier,

        X,

        y_class

    )



    validate_regressor(

        regressor,

        X,

        y_reg

    )



    print(
"""
================================================
MODEL VALIDATION COMPLETED
================================================
"""
    )





if __name__ == "__main__":

    main()