# ======================================================
# FINAL MODEL EVALUATION REPORT
# Employee Future Performance Prediction
# Regression + Classification
# ======================================================

import logging
import joblib
import pandas as pd
import numpy as np
import datetime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import (
    # Regression
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)


# ======================================================
# LOGGING SETUP
# ======================================================

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("model_evaluation")


# ======================================================
# PDF FONT FIX
# ======================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ======================================================
# CONFIGURATION
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "performance_future_features.xlsx"

MODEL_FOLDER = BASE_DIR / "models"

RESULT_FOLDER = BASE_DIR / "results"
RESULT_FOLDER.mkdir(exist_ok=True)

CLASSIFIER_FILE = MODEL_FOLDER / "random_forest_classifier.pkl"
REGRESSOR_FILE = MODEL_FOLDER / "random_forest_regressor.pkl"
FEATURE_FILE = RESULT_FOLDER / "future_feature_names.pkl"

# --- ADDED: same held-out employee IDs saved by train.py ---
TEST_IDS_FILE = RESULT_FOLDER / "test_employee_ids.pkl"
# -------------------------------------------------------------

PDF_FILE = RESULT_FOLDER / "Final_Model_Evaluation_Report.pdf"

CATEGORY_NAMES = {0: "Low", 1: "Medium", 2: "High"}


# ======================================================
# VISUAL SETTINGS
# ======================================================

sns.set_theme(style="whitegrid")

PAGE_SIZE = (11.69, 8.27)

COLOR_PRIMARY = "#2E4374"
COLOR_BLUE = "#3E7CB1"
COLOR_GREEN = "#3E8E5A"
COLOR_RED = "#B5473B"
COLOR_ORANGE = "#D3A029"

BAND_COLORS = {
    "Low": COLOR_RED,
    "Medium": COLOR_ORANGE,
    "High": COLOR_GREEN,
}


# ======================================================
# LOAD DATA
# ======================================================

def load_data():
    logger.info("Loading dataset from %s", DATA_FILE)
    df = pd.read_excel(DATA_FILE)
    logger.info("Dataset shape: %s", df.shape)
    return df


# ======================================================
# LOAD MODELS
# ======================================================

def load_models():
    logger.info("Loading models...")
    classifier = joblib.load(CLASSIFIER_FILE)
    regressor = joblib.load(REGRESSOR_FILE)
    logger.info(
        "Loaded classifier=%s regressor=%s",
        type(classifier).__name__,
        type(regressor).__name__,
    )
    return classifier, regressor


# ======================================================
# PREPARE FEATURES
# ======================================================

def prepare_features(df):
    logger.info("Preparing features...")
    features = joblib.load(FEATURE_FILE)

    X = df[features]
    X = X.fillna(X.median())

    y_class = df["Future_Performance_Category"]
    y_reg = df["Future_Performance_Score"]

    return X, y_class, y_reg


# ======================================================
# REGRESSION EVALUATION (leakage-free: uses saved test IDs)
# ======================================================

def evaluate_regression(model, X, y, df):
    test_ids = joblib.load(TEST_IDS_FILE)
    mask = df["Employee ID"].isin(test_ids)

    X_test = X[mask]
    y_test = y[mask]

    prediction = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
        "R2 Score": r2_score(y_test, prediction),
    }

    logger.info("Regression results: %s", {k: round(v, 4) for k, v in metrics.items()})

    return y_test, prediction, metrics


# ======================================================
# CLASSIFICATION EVALUATION (leakage-free: uses saved test IDs)
# ======================================================

def evaluate_classification(model, X, y, df):
    test_ids = joblib.load(TEST_IDS_FILE)
    mask = df["Employee ID"].isin(test_ids)

    X_test = X[mask]
    y_test = y[mask]

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, average="weighted"),
        "Recall": recall_score(y_test, prediction, average="weighted"),
        "F1 Score": f1_score(y_test, prediction, average="weighted"),
        "ROC-AUC": roc_auc_score(y_test, probability, multi_class="ovr"),
    }

    report = classification_report(y_test, prediction, output_dict=True)

    logger.info("Classification results: %s", {k: round(v, 4) for k, v in metrics.items()})

    return y_test, prediction, probability, metrics, report


# ======================================================
# FEATURE IMPORTANCE
# ======================================================

def get_feature_importance(model, X):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    })

    importance["Importance (%)"] = importance["Importance"] * 100

    return importance.sort_values("Importance (%)", ascending=False).head(10)


# ======================================================
# PDF FOOTER
# ======================================================

def add_footer(fig, page, total_pages):
    date = datetime.date.today().strftime("%B %d, %Y")

    fig.text(0.05, 0.02, f"Generated: {date}", fontsize=8)
    fig.text(
        0.5, 0.02,
        "Employee Future Performance Prediction - Model Evaluation",
        ha="center", fontsize=8,
    )
    fig.text(0.95, 0.02, f"Page {page}/{total_pages}", ha="right", fontsize=8)


# ======================================================
# CREATE PDF REPORT
# ======================================================

def create_pdf(
    regression_metrics,
    classification_metrics,
    y_reg_test,
    reg_prediction,
    y_test,
    prediction,
    probability,
    importance,
    df,
):
    TOTAL_PAGES = 8

    with PdfPages(PDF_FILE) as pdf:

        page = 0

        # PAGE 1 - SUMMARY
        page += 1
        fig = plt.figure(figsize=PAGE_SIZE)

        fig.text(
            0.5, 0.92,
            "EMPLOYEE FUTURE PERFORMANCE",
            fontsize=20, weight="bold", ha="center", color=COLOR_PRIMARY,
        )
        fig.text(
            0.5, 0.87,
            "Final Model Evaluation Report",
            fontsize=13, ha="center", color=COLOR_PRIMARY,
        )
        fig.text(
            0.5, 0.82,
            f"Dataset Records: {len(df)}   |   Classifier: Random Forest   |   Regressor: Random Forest",
            fontsize=10, ha="center", color="dimgray",
        )

        ax_reg = fig.add_axes([0.10, 0.50, 0.35, 0.25])
        ax_reg.axis("off")
        ax_reg.set_title("Regression Results", fontsize=12, weight="bold", color=COLOR_PRIMARY, pad=15)

        reg_rows = [[k, f"{v:.4f}"] for k, v in regression_metrics.items()]
        reg_table = ax_reg.table(
            cellText=reg_rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
        )
        reg_table.auto_set_font_size(False)
        reg_table.set_fontsize(10)
        reg_table.scale(1, 1.8)

        ax_cls = fig.add_axes([0.55, 0.50, 0.35, 0.25])
        ax_cls.axis("off")
        ax_cls.set_title("Classification Results", fontsize=12, weight="bold", color=COLOR_PRIMARY, pad=15)

        cls_rows = [[k, f"{v:.4f}"] for k, v in classification_metrics.items()]
        cls_table = ax_cls.table(
            cellText=cls_rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
        )
        cls_table.auto_set_font_size(False)
        cls_table.set_fontsize(10)
        cls_table.scale(1, 1.8)

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 2 - REGRESSION PERFORMANCE
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        ax.scatter(y_reg_test, reg_prediction, alpha=0.7, color=COLOR_BLUE)
        ax.plot(
            [y_reg_test.min(), y_reg_test.max()],
            [y_reg_test.min(), y_reg_test.max()],
            linestyle="--", color=COLOR_RED,
        )

        ax.set_title("Actual vs Predicted Future Performance Score")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 3 - REGRESSION ERROR DISTRIBUTION
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        residuals = y_reg_test.values - reg_prediction

        ax.hist(residuals, bins=20, color=COLOR_BLUE, edgecolor="white")
        ax.set_title("Regression Residual Error Distribution")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 4 - CLASSIFICATION METRICS
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        names = list(classification_metrics.keys())
        values = list(classification_metrics.values())

        ax.bar(names, values, color=COLOR_PRIMARY)
        ax.set_ylim(0, 1)
        ax.set_title("Classification Performance Metrics")
        plt.xticks(rotation=45)

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 5 - NORMALIZED CONFUSION MATRIX
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        cm_normalized = confusion_matrix(y_test, prediction, normalize="true")

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
            ax=ax
        )

        ax.set_title("Normalized Confusion Matrix")
        ax.set_xlabel("Predicted Performance Category")
        ax.set_ylabel("Actual Performance Category")

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 6 - ROC CURVE
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        for i in range(3):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), probability[:, i])
            score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{CATEGORY_NAMES[i]} AUC={score:.3f}")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

        ax.set_title("ROC Curve - Performance Classification")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 7 - FEATURE IMPORTANCE
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        imp = importance.sort_values("Importance (%)")

        ax.barh(imp["Feature"], imp["Importance (%)"], color=COLOR_GREEN)
        ax.set_title("Top 10 Most Influential Features")
        ax.set_xlabel("Importance (%)")
        plt.tight_layout()

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 8 - DATA DISTRIBUTION
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        counts = df["Future_Performance_Category"].value_counts().sort_index()
        labels = [CATEGORY_NAMES[x] for x in counts.index]
        colors = [BAND_COLORS[label] for label in labels]

        ax.bar(labels, counts.values, color=colors)
        ax.set_title("Future Performance Category Distribution")
        ax.set_ylabel("Number of Employees")

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

    logger.info("PDF generated successfully: %s", PDF_FILE)


# ======================================================
# MAIN FUNCTION
# ======================================================

def main():
    logger.info("=" * 48)
    logger.info("EMPLOYEE FUTURE PERFORMANCE MODEL EVALUATION")
    logger.info("=" * 48)

    df = load_data()
    classifier, regressor = load_models()
    X, y_class, y_reg = prepare_features(df)

    y_reg_test, reg_prediction, regression_metrics = evaluate_regression(
        regressor, X, y_reg, df
    )

    (
        y_test,
        prediction,
        probability,
        classification_metrics,
        report,
    ) = evaluate_classification(classifier, X, y_class, df)

    importance = get_feature_importance(classifier, X)

    create_pdf(
        regression_metrics,
        classification_metrics,
        y_reg_test,
        reg_prediction,
        y_test,
        prediction,
        probability,
        importance,
        df,
    )


if __name__ == "__main__":
    main()