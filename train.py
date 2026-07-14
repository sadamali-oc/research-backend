import pandas as pd
import numpy as np
import os
import time
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from sklearn.utils.class_weight import compute_sample_weight


print("\n====================================")
print("AI EMPLOYEE PERFORMANCE MODEL TRAINING")
print("====================================\n")


os.makedirs("results", exist_ok=True)


# =====================================================
# 1. LOAD DATASET
# =====================================================

DATA_PATH = "data/performance_features.xlsx"

df = pd.read_excel(DATA_PATH)

print("Dataset Loaded")
print("Shape:", df.shape)


print("\nTarget Distribution")
print(df["Performance_Category"].value_counts())


# =====================================================
# 2. FEATURE / TARGET SPLIT
# =====================================================

drop_cols = [
    "Row_ID",
    "Employee ID",
    "Performance_Category"
]


X = df.drop(
    columns=[c for c in drop_cols if c in df.columns]
)

y = df["Performance_Category"]


categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical Features")
print(categorical_features)


print("\nNumeric Features")
print(numeric_features)



# =====================================================
# 3. PREPROCESSING
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ],
    remainder="drop"
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)



# Transform data

X_train_proc = preprocessor.fit_transform(X_train)

X_test_proc = preprocessor.transform(X_test)



# =====================================================
# 4. HANDLE CLASS IMBALANCE
# =====================================================


try:

    from imblearn.over_sampling import SMOTE

    print("\nApplying SMOTE")

    smote = SMOTE(
        random_state=42
    )

    X_train_bal, y_train_bal = smote.fit_resample(
        X_train_proc,
        y_train
    )


    print("\nAfter SMOTE")
    print(
        pd.Series(y_train_bal).value_counts()
    )


    USE_SMOTE = True


except ImportError:

    print(
        "\nSMOTE not installed."
        "\nUsing class weights."
    )


    X_train_bal = X_train_proc
    y_train_bal = y_train

    USE_SMOTE = False



sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_bal
)



# =====================================================
# 5. CROSS VALIDATION
# =====================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



def cross_val_scores(
        model,
        X_tr,
        y_tr,
        weights=None):

    from sklearn.base import clone

    scores = []


    for train_idx, val_idx in cv.split(
        X_tr,
        y_tr
    ):

        model_clone = clone(model)


        X_a = X_tr[train_idx]
        X_b = X_tr[val_idx]


        if hasattr(y_tr, "iloc"):

            y_a = y_tr.iloc[train_idx]
            y_b = y_tr.iloc[val_idx]

        else:

            y_a = y_tr[train_idx]
            y_b = y_tr[val_idx]


        if weights is not None:

            model_clone.fit(
                X_a,
                y_a,
                sample_weight=weights[train_idx]
            )

        else:

            model_clone.fit(
                X_a,
                y_a
            )


        scores.append(
            model_clone.score(
                X_b,
                y_b
            )
        )


    return np.array(scores)



# =====================================================
# 6. MODEL EVALUATION FUNCTION
# =====================================================


results = []

best_estimators = {}



def evaluate_model(
        name,
        model,
        X_tr,
        y_tr,
        X_te,
        y_te,
        weights=None):


    print("\n" + "="*35)
    print("Training", name)
    print("="*35)


    start = time.time()


    if weights is not None:

        model.fit(
            X_tr,
            y_tr,
            sample_weight=weights
        )

    else:

        model.fit(
            X_tr,
            y_tr
        )


    train_time = time.time() - start



    predictions = model.predict(
        X_te
    )


    probabilities = model.predict_proba(
        X_te
    )


    print(
        classification_report(
            y_te,
            predictions
        )
    )


    print("Confusion Matrix")

    print(
        confusion_matrix(
            y_te,
            predictions
        )
    )



    accuracy = accuracy_score(
        y_te,
        predictions
    )


    f1 = f1_score(
        y_te,
        predictions,
        average="weighted"
    )


    auc = roc_auc_score(
        y_te,
        probabilities,
        multi_class="ovr"
    )



    cv_result = cross_val_scores(
        model,
        X_tr,
        y_tr,
        weights
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
        auc
    )

    print(
        "CV Mean:",
        cv_result.mean()
    )



    results.append(
        {
            "Algorithm": name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "ROC-AUC": auc,
            "Training Time": train_time,
            "CV Mean": cv_result.mean()
        }
    )


    best_estimators[name] = model



# =====================================================
# 7. DECISION TREE
# =====================================================


dt_grid = {

    "max_depth":
        [4,6,8,10,None],

    "min_samples_leaf":
        [1,5,10,20]

}



dt_search = GridSearchCV(

    DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    dt_grid,

    cv=cv,

    scoring="f1_weighted",

    n_jobs=-1
)



dt_search.fit(
    X_train_bal,
    y_train_bal
)



print(
    "\nBest Decision Tree:",
    dt_search.best_params_
)



evaluate_model(
    "Decision Tree",
    dt_search.best_estimator_,
    X_train_bal,
    y_train_bal,
    X_test_proc,
    y_test
)



# =====================================================
# 8. RANDOM FOREST
# =====================================================


rf_grid = {

    "n_estimators":
        [200,400],

    "max_depth":
        [8,12,None],

    "min_samples_leaf":
        [1,5,10]

}



rf_search = GridSearchCV(

    RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ),

    rf_grid,

    cv=cv,

    scoring="f1_weighted",

    n_jobs=-1
)



rf_search.fit(
    X_train_bal,
    y_train_bal
)



print(
    "\nBest Random Forest:",
    rf_search.best_params_
)



evaluate_model(
    "Random Forest",
    rf_search.best_estimator_,
    X_train_bal,
    y_train_bal,
    X_test_proc,
    y_test
)



# =====================================================
# 9. GRADIENT BOOSTING
# =====================================================


gb_grid = {


    "n_estimators":
        [150,300],


    "learning_rate":
        [0.03,0.1],


    "max_depth":
        [2,3,4],


    "subsample":
        [0.8,1.0]

}



gb_search = GridSearchCV(

    GradientBoostingClassifier(
        random_state=42
    ),

    gb_grid,

    cv=cv,

    scoring="f1_weighted",

    n_jobs=-1
)



gb_search.fit(

    X_train_bal,

    y_train_bal,

    sample_weight=None if USE_SMOTE else sample_weights

)



print(
    "\nBest Gradient Boosting:",
    gb_search.best_params_
)



evaluate_model(

    "Gradient Boosting",

    gb_search.best_estimator_,

    X_train_bal,

    y_train_bal,

    X_test_proc,

    y_test,

    None if USE_SMOTE else sample_weights

)



# =====================================================
# 10. MODEL COMPARISON
# =====================================================


results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "F1 Score",
    ascending=False
)



print("\nMODEL COMPARISON")

print(results_df)



best_name = results_df.iloc[0]["Algorithm"]

best_model = best_estimators[best_name]



print("\nBEST MODEL")

print(best_name)

print(
    "F1 Score:",
    results_df.iloc[0]["F1 Score"]
)



# =====================================================
# 11. SAVE MODEL
# =====================================================


joblib.dump(
    best_model,
    "results/best_model.pkl"
)


joblib.dump(
    preprocessor,
    "results/preprocessor.pkl"
)


results_df.to_excel(
    "results/model_comparison.xlsx",
    index=False
)



print("\nBest model saved")



# =====================================================
# 12. FEATURE IMPORTANCE
# =====================================================


if hasattr(
    best_model,
    "feature_importances_"
):


    importance_df = pd.DataFrame(

        {

            "Feature":
                numeric_features,


            "Importance":
                best_model.feature_importances_

        }

    )


    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )


    print(
        "\nFeature Importance"
    )

    print(
        importance_df
    )



    importance_df.to_excel(
        "results/feature_importance.xlsx",
        index=False
    )



print("\n====================================")
print("MODEL TRAINING COMPLETE")
print("====================================")