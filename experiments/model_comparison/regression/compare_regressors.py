import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np



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
    "regression"
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
    "regression_model_comparison.pdf"
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
        "Future_Performance_Score"
    ]


    groups = df[
        "Employee ID"
    ]



    categorical = X.select_dtypes(

        include=[
            "object",
            "string"
        ]

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


    return X,y,groups



# ======================================================
# MODELS
# ======================================================

def get_models():


    return {


        "Decision Tree":

        DecisionTreeRegressor(

            max_depth=10,

            min_samples_leaf=5,

            random_state=42

        ),



        "Random Forest":

        RandomForestRegressor(

            n_estimators=300,

            max_depth=10,

            min_samples_leaf=3,

            random_state=42,

            n_jobs=-1

        ),



        "Gradient Boosting":

        GradientBoostingRegressor(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=4,

            min_samples_leaf=5,

            subsample=0.8,

            random_state=42

        )

    }



# ======================================================
# CREATE PDF
# ======================================================

def create_pdf(result_df):


    with PdfPages(PDF_FILE) as pdf:


        # Table page

        fig,ax=plt.subplots(
            figsize=(10,4)
        )

        ax.axis(
            "off"
        )


        table=ax.table(

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



        # Charts

        for metric in [

            "MAE",

            "RMSE",

            "R2 Score"

        ]:


            fig,ax=plt.subplots(
                figsize=(8,5)
            )


            ax.bar(

                result_df["Algorithm"],

                result_df[metric]

            )


            ax.set_title(

                "Regression Model Comparison - "
                + metric

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
REGRESSION MODEL COMPARISON

Decision Tree
Random Forest
Gradient Boosting

====================================
"""
    )


    df=load_data()


    X,y,groups=prepare_features(df)



    splitter=GroupShuffleSplit(

        test_size=0.2,

        random_state=42

    )


    train_idx,test_idx=next(

        splitter.split(

            X,

            y,

            groups

        )

    )



    X_train=X.iloc[train_idx]

    X_test=X.iloc[test_idx]


    y_train=y.iloc[train_idx]

    y_test=y.iloc[test_idx]



    print(

        "Training Employees:",

        groups.iloc[train_idx].nunique()

    )


    print(

        "Testing Employees:",

        groups.iloc[test_idx].nunique()

    )



    models=get_models()


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


        prediction=model.predict(

            X_test

        )



        mae=mean_absolute_error(

            y_test,

            prediction

        )


        rmse=np.sqrt(

            mean_squared_error(

                y_test,

                prediction

            )

        )


        r2=r2_score(

            y_test,

            prediction

        )



        cv=cross_val_score(

            model,

            X,

            y,

            cv=5,

            scoring="r2"

        )



        print(
            "MAE:",
            mae
        )

        print(
            "RMSE:",
            rmse
        )

        print(
            "R2:",
            r2
        )



        results.append({

            "Algorithm":name,

            "MAE":mae,

            "RMSE":rmse,

            "R2 Score":r2,

            "CV R2 Mean":cv.mean(),

            "Training Time":time.time()-start

        })



        joblib.dump(

            model,

            MODEL_FOLDER /
            f"{name.replace(' ','_')}.pkl"

        )




    result_df=pd.DataFrame(results)



    result_df.sort_values(

        by="R2 Score",

        ascending=False,

        inplace=True

    )



    create_pdf(

        result_df

    )



    print(
"""
====================================
REGRESSION EXPERIMENT COMPLETED

PDF:
experiments/model_comparison/regression/

Models:
experiments/model_comparison/regression/saved_models/

====================================
"""
    )


    print(result_df)



if __name__=="__main__":

    main()