import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import seaborn as sns

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib
matplotlib.use("Agg")  # Prevent GUI errors

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from sklearn.model_selection import (
    train_test_split,
    learning_curve
)

# ======================================================
# CONFIGURATION
# ======================================================

RESULT_PATH = Path("results")

DATA_FILE = Path("data/performance_future_features.xlsx")

MODEL_FILE = RESULT_PATH / "best_model.pkl"
ENCODER_FILE = RESULT_PATH / "feature_encoders.pkl"
FEATURE_FILE = RESULT_PATH / "feature_names.pkl"

PDF_FILE = RESULT_PATH / "Future_Performance_Evaluation_Report.pdf"

CATEGORY_NAMES = {
    0: "Low",
    1: "Medium",
    2: "High"
}

CATEGORY_COLORS = ["tomato", "gold", "seagreen"]


# ======================================================
# LOAD MODEL
# ======================================================

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model file not found")

    model = joblib.load(MODEL_FILE)

    print("Loaded Model:", type(model).__name__)
    return model


# ======================================================
# LOAD DATA
# ======================================================

def prepare_data():
    df = pd.read_excel(DATA_FILE)

    TARGET = "Future_Performance_Category"

    drop_columns = [
        "Row_ID",
        "Employee ID",
        "Future_Performance_Score",
        "Actual_Future_Band",
        TARGET
    ]

    X = df.drop(columns=[c for c in drop_columns if c in df.columns])
    y = df[TARGET]

    print("\nOriginal Features:")
    print(X.columns.tolist())

    # Load encoders
    if ENCODER_FILE.exists():
        encoders = joblib.load(ENCODER_FILE)

        for col, encoder in encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median())

    # Match training features
    if FEATURE_FILE.exists():
        feature_names = joblib.load(FEATURE_FILE)
        X = X[feature_names]

    print("\nFinal Features:")
    print(X.columns.tolist())

    return df, X, y


# ======================================================
# EVALUATION
# ======================================================

def evaluate(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "f1": f1_score(y_test, prediction, average="macro"),
        "roc": roc_auc_score(y_test, probability, multi_class="ovr")
    }

    return X_test, y_test, prediction, probability, metrics


# ======================================================
# FEATURE IMPORTANCE
# ======================================================

def get_feature_importance(model, X):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance["Importance (%)"] = (importance["Importance"] * 100).round(2)

    return importance.sort_values(
        "Importance (%)",
        ascending=False
    ).head(10)


# ======================================================
# PDF REPORT
# ======================================================

def create_pdf(df, model, X, y, y_test, prediction, probability, metrics, importance):

    with PdfPages(PDF_FILE) as pdf:

        # ================= COVER PAGE =================
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        ax.text(0.5, 0.7, "AI FUTURE PERFORMANCE REPORT",
                fontsize=20, ha="center", weight="bold")

        ax.text(0.5, 0.5,
                f"Model: {type(model).__name__}\n"
                f"Accuracy: {metrics['accuracy']:.2f}\n"
                f"F1 Score: {metrics['f1']:.2f}\n"
                f"ROC-AUC: {metrics['roc']:.2f}",
                fontsize=12, ha="center")

        pdf.savefig(fig)
        plt.close()

        # ================= FEATURE IMPORTANCE =================
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(importance.Feature[::-1],
                importance["Importance (%)"][::-1],
                color="skyblue")

        ax.set_title("Top 10 Important Features", fontsize=14)
        ax.set_xlabel("Importance (%)")
        ax.set_ylabel("Features")

        for i, v in enumerate(importance["Importance (%)"][::-1]):
            ax.text(v + 0.5, i, str(v), va='center')

        pdf.savefig(fig)
        plt.close()

        # ================= DISTRIBUTION =================
        counts = df["Future_Performance_Category"].value_counts()
        labels = [CATEGORY_NAMES[i] for i in counts.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, counts.values, color=CATEGORY_COLORS)

        ax.set_title("Employee Performance Distribution")
        ax.set_xlabel("Performance Category")
        ax.set_ylabel("Number of Employees")

        pdf.savefig(fig)
        plt.close()

        # ================= CONFUSION MATRIX =================
        cm = confusion_matrix(y_test, prediction)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=CATEGORY_NAMES.values(),
                    yticklabels=CATEGORY_NAMES.values())

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        pdf.savefig(fig)
        plt.close()

        # ================= ROC CURVE =================
        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(len(model.classes_)):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), probability[:, i])
            ax.plot(fpr, tpr, label=f"{CATEGORY_NAMES[i]}")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        pdf.savefig(fig)
        plt.close()

        # ================= LEARNING CURVE =================
        sizes, train, test = learning_curve(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(sizes, train.mean(axis=1), 'o-', label="Training Score")
        ax.plot(sizes, test.mean(axis=1), 'o-', label="Validation Score")

        ax.fill_between(sizes,
                        train.mean(axis=1) - train.std(axis=1),
                        train.mean(axis=1) + train.std(axis=1),
                        alpha=0.1)

        ax.fill_between(sizes,
                        test.mean(axis=1) - test.std(axis=1),
                        test.mean(axis=1) + test.std(axis=1),
                        alpha=0.1)

        ax.set_title("Learning Curve")
        ax.set_xlabel("Training Size")
        ax.set_ylabel("Accuracy")
        ax.legend()

        pdf.savefig(fig)
        plt.close()

        # ================= MODEL INTERPRETATION PAGE =================
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        interpretation = f"""
MODEL INSIGHTS

• Model Accuracy: {metrics['accuracy']:.2f}
• Model F1 Score: {metrics['f1']:.2f}


"""

        ax.text(0.05, 0.95, interpretation, va="top", fontsize=11)

        pdf.savefig(fig)
        plt.close()
# ======================================================
# MAIN
# ======================================================

def main():

    print("Generating Report...")

    model = load_model()
    df, X, y = prepare_data()

    X_test, y_test, prediction, probability, metrics = evaluate(model, X, y)

    importance = get_feature_importance(model, X)

    create_pdf(
        df,
        model,
        X,
        y,
        y_test,
        prediction,
        probability,
        metrics,
        importance
    )

    print("PDF GENERATED SUCCESSFULLY")


if __name__ == "__main__":
    main()