# ======================================================
# RANDOM FOREST MODEL VALIDATION
# Employee Future Performance Prediction
# ======================================================

import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ======================================================
# PATHS
# ======================================================

CLASSIFICATION_MODEL = Path(
    "models/random_forest_classifier.pkl"
)

REGRESSION_MODEL = Path(
    "models/random_forest_regressor.pkl"
)

DATA_FILE = Path(
    "data/performance_future_features.xlsx"
)


# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_excel(DATA_FILE)

print("\nDataset Shape:")
print(df.shape)



# ======================================================
# TARGET COLUMNS
# ======================================================

CLASS_TARGET = "Future_Performance_Category"

REG_TARGET = "Future_Performance_Score"



# ======================================================
# REMOVE NON-MODEL FEATURES
# ======================================================

REMOVE_COLUMNS = [
    "Row_ID",
    "Employee ID",
    "Period Year",
    "Period Quarter"
]



# ======================================================
# LOAD CLASSIFICATION MODEL
# ======================================================

print("\n======================================")
print(" RANDOM FOREST CLASSIFICATION CHECK ")
print("======================================")


classifier = joblib.load(
    CLASSIFICATION_MODEL
)

print("\nLoaded Model:")
print(classifier)



# ======================================================
# PREPARE CLASSIFICATION DATA
# ======================================================

X_class = df.drop(
    columns=[
        CLASS_TARGET,
        REG_TARGET
    ],
    errors="ignore"
)


X_class = X_class.drop(
    columns=REMOVE_COLUMNS,
    errors="ignore"
)


y_class = df[CLASS_TARGET]



# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)



# ======================================================
# PREDICTION
# ======================================================

train_prediction = classifier.predict(
    X_train
)

test_prediction = classifier.predict(
    X_test
)



# ======================================================
# CLASSIFICATION RESULTS
# ======================================================

print("\nTraining Accuracy:")

print(
    round(
        accuracy_score(
            y_train,
            train_prediction
        ),
        4
    )
)


print("\nTesting Results")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            test_prediction
        ),
        4
    )
)


print(
    "Precision:",
    round(
        precision_score(
            y_test,
            test_prediction,
            average="weighted"
        ),
        4
    )
)


print(
    "Recall:",
    round(
        recall_score(
            y_test,
            test_prediction,
            average="weighted"
        ),
        4
    )
)


print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            test_prediction,
            average="weighted"
        ),
        4
    )
)



print("\nClassification Report")

print(
    classification_report(
        y_test,
        test_prediction
    )
)



print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        test_prediction
    )
)



# ======================================================
# CROSS VALIDATION
# ======================================================

cv_scores = cross_val_score(
    classifier,
    X_class,
    y_class,
    cv=5,
    scoring="accuracy"
)


print("\nCross Validation Scores")

print(cv_scores)

print(
    "Average CV Accuracy:",
    round(
        cv_scores.mean(),
        4
    )
)



# ======================================================
# LOAD REGRESSION MODEL
# ======================================================

print("\n================================")
print(" RANDOM FOREST REGRESSION CHECK ")
print("================================")


regressor = joblib.load(
    REGRESSION_MODEL
)


print("\nLoaded Model:")
print(regressor)



# ======================================================
# PREPARE REGRESSION DATA
# ======================================================

X_reg = df.drop(
    columns=[
        CLASS_TARGET,
        REG_TARGET
    ],
    errors="ignore"
)


X_reg = X_reg.drop(
    columns=REMOVE_COLUMNS,
    errors="ignore"
)


y_reg = df[REG_TARGET]



# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)



# ======================================================
# REGRESSION PREDICTION
# ======================================================

prediction = regressor.predict(
    X_test
)



# ======================================================
# REGRESSION RESULTS
# ======================================================

mae = mean_absolute_error(
    y_test,
    prediction
)


mse = mean_squared_error(
    y_test,
    prediction
)


rmse = mse ** 0.5


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



print("\n================================")
print(" MODEL CHECK COMPLETED ")
print("================================")