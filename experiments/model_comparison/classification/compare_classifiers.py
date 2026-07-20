import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import GroupShuffleSplit, cross_val_score
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


# ======================================================
# PATH CONFIGURATION
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[3]


DATA_FILE = (
    BASE_DIR /
    "data" /
    "performance_future_features.xlsx"
)


EXPERIMENT_FOLDER = (
    BASE_DIR /
    "experiments" /
    "model_comparison" /
    "classification"
)

EXPERIMENT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_FOLDER = (
    EXPERIMENT_FOLDER /
    "saved_models"
)

MODEL_FOLDER.mkdir(
    exist_ok=True
)


PDF_FILE = (
    EXPERIMENT_FOLDER /
    "classification_model_comparison.pdf"
)



# ======================================================
# LOAD DATA
# ======================================================

def load_data():

    print("\nLoading dataset...")

    df = pd.read_excel(DATA_FILE)

    print(
        "Dataset:",
        df.shape
    )

    return df



# ======================================================
# PREPARE FEATURES
# ======================================================

def prepare_features(df):

    print("\nPreparing features...")


    drop_columns = [

        "Row_ID",
        "Employee ID",
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
        "Future_Performance_Category"
    ]


    groups = df[
        "Employee ID"
    ]


    categorical = X.select_dtypes(
        include=["object","string"]
    ).columns


    print(
        "Categorical:",
        list(categorical)
    )


    for col in categorical:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )


    X = X.fillna(
        X.median()
    )


    return X, y, groups



# ======================================================
# MODELS
# ======================================================

def get_models():

    return {


        "Decision Tree":

        DecisionTreeClassifier(

            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42

        ),



        "Random Forest":

        RandomForestClassifier(

            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
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



# ======================================================
# PDF REPORT
# ======================================================

def create_pdf(result_df):

    with PdfPages(PDF_FILE) as pdf:


        # Table page

        fig, ax = plt.subplots(
            figsize=(10,4)
        )

        ax.axis("off")


        table = ax.table(

            cellText=result_df.round(4).values,

            colLabels=result_df.columns,

            loc="center"

        )


        table.auto_set_font_size(False)

        table.set_fontsize(8)

        table.scale(
            1,
            2
        )


        pdf.savefig(
            bbox_inches="tight"
        )

        plt.close()



        # Metric charts

        for metric in [
            "Accuracy",
            "F1 Score",
            "ROC-AUC"
        ]:


            fig, ax = plt.subplots(
                figsize=(8,5)
            )


            ax.bar(

                result_df["Algorithm"],

                result_df[metric]

            )


            ax.set_title(
                "Model Comparison - " + metric
            )


            ax.set_ylabel(
                metric
            )


            ax.tick_params(
                axis="x",
                rotation=45
            )


            plt.tight_layout()


            pdf.savefig()

            plt.close()



# ======================================================
# MAIN
# ======================================================

def main():


    print(
"""
====================================
CLASSIFICATION MODEL COMPARISON

Decision Tree
Random Forest
Gradient Boosting

====================================
"""
    )


    df = load_data()


    X,y,groups = prepare_features(df)



    splitter = GroupShuffleSplit(

        test_size=0.2,

        random_state=42

    )


    train_idx,test_idx = next(

        splitter.split(
            X,
            y,
            groups
        )

    )


    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    print(
        "Training Employees:",
        groups.iloc[train_idx].nunique()
    )

    print(
        "Testing Employees:",
        groups.iloc[test_idx].nunique()
    )



    models = get_models()


    results=[]



    for name,model in models.items():


        print(
            "\nTraining:",
            name
        )


        start=time.time()


        model.fit(
            X_train,
            y_train
        )


        pred=model.predict(
            X_test
        )


        prob=model.predict_proba(
            X_test
        )


        accuracy=accuracy_score(
            y_test,
            pred
        )


        f1=f1_score(
            y_test,
            pred,
            average="macro"
        )


        roc=roc_auc_score(
            y_test,
            prob,
            multi_class="ovr"
        )


        cv=cross_val_score(

            model,

            X,

            y,

            cv=5,

            scoring="f1_macro"

        )


        print(
            classification_report(
                y_test,
                pred
            )
        )


        results.append({

            "Algorithm":name,

            "Accuracy":accuracy,

            "F1 Score":f1,

            "ROC-AUC":roc,

            "CV F1 Mean":cv.mean(),

            "Training Time":time.time()-start

        })



        joblib.dump(

            model,

            MODEL_FOLDER /
            f"{name.replace(' ','_')}.pkl"

        )



    result_df=pd.DataFrame(results)


    result_df.sort_values(

        by="F1 Score",

        ascending=False,

        inplace=True

    )


    create_pdf(
        result_df
    )


    print(
"""
====================================
COMPLETED

PDF:
experiments/model_comparison/classification/

Models:
experiments/model_comparison/classification/saved_models/

====================================
"""
    )


    print(result_df)



if __name__=="__main__":
    main()