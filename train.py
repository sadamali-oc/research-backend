import time
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)



# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = Path(
    "data/performance_future_features.xlsx"
)


RESULT_PATH = Path(
    "results"
)

RESULT_PATH.mkdir(
    exist_ok=True
)



MODEL_FILE = (
    RESULT_PATH /
    "best_model.pkl"
)


ENCODER_FILE = (
    RESULT_PATH /
    "feature_encoders.pkl"
)


FEATURE_FILE = (
    RESULT_PATH /
    "feature_names.pkl"
)


METRIC_FILE = (
    RESULT_PATH /
    "final_model_metrics.xlsx"
)


PREDICTION_FILE = (
    RESULT_PATH /
    "classification_predictions.xlsx"
)





# ==========================================
# LOAD DATA
# ==========================================

def load_prepare():


    print("\nLoading dataset...")


    df = pd.read_excel(

        DATA_PATH

    )


    print(

        "Dataset:",

        df.shape

    )



    # Remove target leakage

    drop_columns = [

        "Row_ID",

        "Employee ID",

        "Future_Performance_Score",

        "Future_Performance_Category",

        "Actual_Future_Band"

    ]



    X = df.drop(

        columns=[

            c for c in drop_columns

            if c in df.columns

        ]

    )



    y = df[

        "Future_Performance_Category"

    ]





    # ==================================
    # ENCODING
    # ==================================


    encoders = {}


    categorical_columns = X.select_dtypes(

        include="object"

    ).columns



    print(

        "\nCategorical Features:"

    )


    print(

        list(categorical_columns)

    )



    for col in categorical_columns:


        encoder = LabelEncoder()



        X[col] = encoder.fit_transform(

            X[col].astype(str)

        )



        encoders[col] = encoder




    # ==================================
    # MISSING VALUES
    # ==================================


    print(

        "\nMissing Values Before:",

        X.isnull().sum().sum()

    )



    X = X.fillna(

        X.median()

    )



    print(

        "Missing Values After:",

        X.isnull().sum().sum()

    )




    # ==================================
    # SAVE FEATURES
    # ==================================


    joblib.dump(

        encoders,

        ENCODER_FILE

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
========================================
FUTURE PERFORMANCE CATEGORY MODEL
(Random Forest Classifier)
========================================
"""
    )



    X, y = load_prepare()



    print(

        "\nTarget Distribution:"

    )


    print(

        y.value_counts()

    )





    # ==================================
    # TRAIN TEST SPLIT
    # ==================================


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    print(

        "\nTraining:",

        X_train.shape

    )


    print(

        "Testing:",

        X_test.shape

    )





    # ==================================
    # RANDOM FOREST
    # ==================================


    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=10,

        min_samples_leaf=3,

        max_features="sqrt",

        class_weight="balanced",

        random_state=42,

        n_jobs=-1

    )




    start_time = time.time()



    model.fit(

        X_train,

        y_train

    )



    training_time = (

        time.time()

        -

        start_time

    )





    # ==================================
    # PREDICTION
    # ==================================


    prediction = model.predict(

        X_test

    )



    probability = model.predict_proba(

        X_test

    )





    # ==================================
    # SAVE PREDICTIONS
    # ==================================


    prediction_output = X_test.copy()



    prediction_output[

        "Actual_Category"

    ] = y_test.values



    prediction_output[

        "Predicted_Category"

    ] = prediction



    prediction_output.to_excel(

        PREDICTION_FILE,

        index=False

    )





    # ==================================
    # EVALUATION
    # ==================================


    accuracy = accuracy_score(

        y_test,

        prediction

    )



    f1 = f1_score(

        y_test,

        prediction,

        average="macro"

    )



    roc = roc_auc_score(

        y_test,

        probability,

        multi_class="ovr"

    )



    cv_scores = cross_val_score(

        model,

        X,

        y,

        cv=5,

        scoring="f1_macro"

    )



    cv_mean = cv_scores.mean()





    print(

        "\nClassification Report"

    )


    print(

        classification_report(

            y_test,

            prediction

        )

    )




    print(

        "Accuracy:",

        accuracy

    )


    print(

        "F1 Score:",

        f1

    )


    print(

        "ROC-AUC:",

        roc

    )


    print(

        "CV F1 Mean:",

        cv_mean

    )


    print(

        "Training Time:",

        training_time

    )





    # ==================================
    # SAVE METRICS
    # ==================================


    metrics = pd.DataFrame({


        "Metric":[

            "Accuracy",

            "F1 Score",

            "ROC-AUC",

            "CV F1 Mean",

            "Training Time"

        ],


        "Value":[

            accuracy,

            f1,

            roc,

            cv_mean,

            training_time

        ]

    })



    metrics.to_excel(

        METRIC_FILE,

        index=False

    )





    # ==================================
    # SAVE MODEL
    # ==================================


    joblib.dump(

        model,

        MODEL_FILE

    )





    print(
"""
========================================
MODEL SAVED SUCCESSFULLY

Model:
results/best_model.pkl

Metrics:
results/final_model_metrics.xlsx

Predictions:
results/classification_predictions.xlsx

Features:
results/feature_names.pkl

========================================
"""
    )





if __name__ == "__main__":

    main()