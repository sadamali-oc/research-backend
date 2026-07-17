import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
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

PDF_FILE = RESULT_FOLDER / "Future_Score_Regression_Report.pdf"



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


    # Only fill numeric columns with the median; fillna(X.median())
    # silently drops non-numeric columns from the median Series, which
    # can misalign things if you ever have categorical features here.
    numeric_cols = X.select_dtypes(include="number").columns

    X[numeric_cols] = X[numeric_cols].fillna(
        X[numeric_cols].median()
    )


    joblib.dump(

        X.columns.tolist(),

        FEATURE_FILE

    )


    return X, y

# ==========================================
# CREATE PDF REPORT
# ==========================================

def create_pdf_report(
        model,
        X_test,
        y_test,
        prediction,
        mae,
        mse,
        rmse,
        r2
):

    with PdfPages(PDF_FILE) as pdf:


        # ==============================
        # PAGE 1 - MODEL METRICS
        # ==============================

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.axis("off")


        report = f"""AI FUTURE EMPLOYEE PERFORMANCE
REGRESSION REPORT

Machine Learning Model:
Random Forest Regressor

Performance Metrics

MAE  : {mae:.4f}
MSE  : {mse:.4f}
RMSE : {rmse:.4f}
R2 Score : {r2:.4f}
"""

        # KEY FIX: without transform=ax.transAxes, matplotlib treats
        # (0.1, 0.8) as DATA coordinates. Since this axis has nothing
        # else drawn on it (axis is off), autoscaling produces
        # unpredictable / tiny / off-page text. transAxes pins
        # (0,0)-(1,1) to the figure area regardless of content.
        ax.text(
            0.05,
            0.95,
            report,
            fontsize=13,
            family="monospace",
            ha="left",
            va="top",
            transform=ax.transAxes,
            wrap=True
        )


        pdf.savefig(fig)

        plt.close(fig)



        # ==============================
        # PAGE 2 - ACTUAL VS PREDICTED
        # ==============================


        fig, ax = plt.subplots(
            figsize=(8, 6)
        )


        ax.scatter(
            y_test,
            prediction,
            alpha=0.6,
            edgecolor="k",
            label="Predictions"
        )


        ax.plot(

            [
                y_test.min(),
                y_test.max()
            ],

            [
                y_test.min(),
                y_test.max()
            ],

            color="red",
            linestyle="--",
            label="Perfect Prediction"

        )


        ax.set_title(
            "Actual vs Predicted Future Performance Score"
        )


        ax.set_xlabel(
            "Actual Score"
        )


        ax.set_ylabel(
            "Predicted Score"
        )

        ax.legend()


        pdf.savefig(fig)

        plt.close(fig)



        # ==============================
        # PAGE 3 - ERROR DISTRIBUTION
        # ==============================


        errors = y_test - prediction


        fig, ax = plt.subplots(
            figsize=(8, 6)
        )


        ax.hist(
            errors,
            bins=30,
            edgecolor="black"
        )


        ax.set_title(
            "Prediction Error Distribution"
        )


        ax.set_xlabel(
            "Error"
        )


        ax.set_ylabel(
            "Frequency"
        )


        pdf.savefig(fig)

        plt.close(fig)



        # ==============================
        # PAGE 4 - FEATURE IMPORTANCE
        # ==============================


        importance = pd.DataFrame({

            "Feature":
            X_test.columns,

            "Importance":
            model.feature_importances_

        })


        importance = importance.sort_values(

            "Importance",

            ascending=False

        ).head(10)


        # Sort ascending for barh so the most important feature
        # appears at the TOP of the chart instead of the bottom.
        importance = importance.sort_values(
            "Importance",
            ascending=True
        )


        fig, ax = plt.subplots(
            figsize=(10, 6)
        )


        ax.barh(

            importance["Feature"],

            importance["Importance"]

        )


        ax.set_title(
            "Top Features Influencing Future Performance"
        )


        ax.set_xlabel(
            "Importance"
        )

        fig.tight_layout()


        pdf.savefig(fig)

        plt.close(fig)



    print(
        "\nPDF Generated:",
        PDF_FILE
    )



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



    print("\nTarget Statistics")

    print(
        y.describe()
    )



    # Train Test Split

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




    # Prediction

    prediction = model.predict(
        X_test
    )



    # ==============================
    # METRICS
    # ==============================


    mae = mean_absolute_error(

        y_test,

        prediction

    )


    mse = mean_squared_error(

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


    print("MAE :", round(mae, 4))

    print("MSE :", round(mse, 4))

    print("RMSE:", round(rmse, 4))

    print("R2  :", round(r2, 4))

    print("CV R2:", round(cv_mean, 4))

    print(
        "Training Time:",
        round(training_time, 4)
    )



    # ==============================
    # SAVE METRICS
    # ==============================


    metrics = pd.DataFrame({

        "Metric": [

            "MAE",

            "MSE",

            "RMSE",

            "R2",

            "CV R2",

            "Training Time"

        ],


        "Value": [

            mae,

            mse,

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



    # PDF

    create_pdf_report(

        model,

        X_test,

        y_test,

        prediction,

        mae,

        mse,

        rmse,

        r2

    )



    # Save model


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


PDF:
results/Future_Score_Regression_Report.pdf


Features:
results/score_feature_names.pkl


====================================
"""
    )




if __name__ == "__main__":

    main()