import time
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

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


MODEL_COMPARISON_PATH = (
    RESULT_PATH /
    "model_comparison"
)


MODEL_COMPARISON_PATH.mkdir(
    exist_ok=True
)



# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    df = pd.read_excel(
        DATA_PATH
    )


    print(
        "Dataset:",
        df.shape
    )


    return df




# ==========================================
# PREPARE FEATURES
# ==========================================

def prepare_features(df):


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



    encoders = {}



    categorical = X.select_dtypes(
        include="object"
    ).columns



    print(
        "\nCategorical Features:"
    )


    print(
        list(categorical)
    )



    for col in categorical:


        encoder = LabelEncoder()


        X[col] = encoder.fit_transform(

            X[col].astype(str)

        )


        encoders[col] = encoder




    # ======================================
    # HANDLE MISSING VALUES
    # ======================================

    print(
        "\nMissing Values Before:"
    )


    print(
        X.isnull().sum().sum()
    )



    X = X.fillna(

        X.median()

    )



    print(
        "Missing Values After:"
    )


    print(
        X.isnull().sum().sum()
    )



    # Save encoders

    joblib.dump(

        encoders,

        RESULT_PATH /
        "feature_encoders.pkl"

    )



    # Save feature names for XAI

    joblib.dump(

        X.columns.tolist(),

        RESULT_PATH /
        "feature_names.pkl"

    )



    print(

        "\nEncoders saved:"

    )


    print(

        RESULT_PATH /
        "feature_encoders.pkl"

    )


    return X, y




# ==========================================
# MODELS
# ==========================================

def get_models():

    return {


        "Decision Tree":

        DecisionTreeClassifier(

            class_weight="balanced",

            random_state=42

        ),



        "Random Forest":

        RandomForestClassifier(

            n_estimators=200,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1

        ),



        "Gradient Boosting":

        GradientBoostingClassifier(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=4,

            min_samples_leaf=5,

            subsample=0.8,

            random_state=42

        )

    }





# ==========================================
# MAIN
# ==========================================

def main():


    print(
"""
====================================
MODEL COMPARISON
====================================
"""
    )



    df = load_data()



    X, y = prepare_features(
        df
    )



    print(
        "\nFeatures:"
    )


    print(
        X.columns.tolist()
    )



    print(
        "\nTarget Distribution:"
    )


    print(
        y.value_counts()
    )



    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    print(

        "\nTraining Data:",

        X_train.shape

    )


    print(

        "Testing Data:",

        X_test.shape

    )



    models = get_models()



    results = []





    for name, model in models.items():


        print(

            "\nTraining:",

            name

        )



        start = time.time()



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



        training_time = (

            time.time()

            -

            start

        )



        print(

            classification_report(

                y_test,

                prediction

            )

        )



        print(

            "CV F1 Mean:",

            cv_mean

        )



        results.append(

            {

                "Algorithm": name,

                "Accuracy": accuracy,

                "F1 Score": f1,

                "ROC-AUC": roc,

                "CV F1 Mean": cv_mean,

                "Training Time": training_time

            }

        )





    result_df = pd.DataFrame(

        results

    )



    result_df.sort_values(

        [

            "F1 Score",

            "ROC-AUC"

        ],

        ascending=False,

        inplace=True

    )




    print(

        "\nMODEL COMPARISON RESULTS"

    )


    print(

        result_df

    )





    output_file = (

        MODEL_COMPARISON_PATH /

        "future_model_comparison.xlsx"

    )



    result_df.to_excel(

        output_file,

        index=False

    )



    print(
"""
====================================
MODEL COMPARISON FILE SAVED

Location:

results/model_comparison/
future_model_comparison.xlsx

====================================
"""
    )





    best = result_df.iloc[0]



    print(

        "BEST MODEL:",

        best["Algorithm"]

    )


    print(

        f"Accuracy: {best['Accuracy']*100:.2f}%"

    )


    print(

        f"F1 Score: {best['F1 Score']*100:.2f}%"

    )


    print(

        f"ROC-AUC: {best['ROC-AUC']*100:.2f}%"

    )


    print(

        f"CV F1 Mean: {best['CV F1 Mean']*100:.2f}%"

    )





if __name__ == "__main__":

    main()