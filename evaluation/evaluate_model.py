# ======================================================
# RANDOM FOREST MODEL EVALUATION
# Employee Future Performance Prediction
# ======================================================

import joblib
import pandas as pd
import numpy as np
import datetime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)


# ======================================================
# CONFIGURATION
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULT_PATH = BASE_DIR / "results"

DATA_FILE = BASE_DIR / "data" / "performance_future_features.xlsx"

MODEL_FILE = BASE_DIR / "models" / "random_forest_classifier.pkl"

FEATURE_FILE = RESULT_PATH / "future_feature_names.pkl"

PDF_FILE = RESULT_PATH / "Future_Performance_Evaluation_Report.pdf"

CATEGORY_NAMES = {0: "Low", 1: "Medium", 2: "High"}

# ---- Visual theme (applied once, used everywhere) ----

COLOR_PRIMARY = "#2E4374"      # deep navy-blue, headers/titles
COLOR_ACCENT = "#3E7CB1"       # mid blue, bars/lines
COLOR_ACCENT2 = "#81A4CD"      # light blue, secondary series
COLOR_HIGH = "#3E8E5A"         # green
COLOR_MED = "#D3A029"          # amber
COLOR_LOW = "#B5473B"          # red
BAND_COLORS = {"Low": COLOR_LOW, "Medium": COLOR_MED, "High": COLOR_HIGH}
COLOR_GRID = "#D9D9D9"
COLOR_TEXT_MUTED = "#6E6E6E"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "axes.edgecolor": COLOR_GRID,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlecolor": COLOR_PRIMARY,
    "axes.labelcolor": "#333333",
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
})

PAGE_SIZE = (10, 7.5)  # consistent page size across every figure


# ======================================================
# LOAD MODEL
# ======================================================

def load_model():

    print("\nLoading model...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError("Random Forest model not found")

    model = joblib.load(MODEL_FILE)

    print("Model:", type(model).__name__)

    return model


# ======================================================
# LOAD DATA
# ======================================================

def load_data():

    print("\nLoading dataset...")

    df = pd.read_excel(DATA_FILE)

    print("Dataset:", df.shape)

    return df


# ======================================================
# PREPARE FEATURES
# ======================================================

def prepare_features(df):

    print("\nPreparing features...")

    feature_names = joblib.load(FEATURE_FILE)

    X = df[feature_names]

    y = df["Future_Performance_Category"]

    X = X.fillna(X.median(numeric_only=True))

    return X, y


# ======================================================
# EVALUATION
# ======================================================

def evaluate_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, average="weighted"),
        "Recall": recall_score(y_test, prediction, average="weighted"),
        "F1 Score": f1_score(y_test, prediction, average="weighted"),
        "ROC-AUC": roc_auc_score(y_test, probability, multi_class="ovr"),
    }

    print("\nEvaluation Results")
    for key, value in metrics.items():
        print(key, ":", round(value, 3))

    report_text = classification_report(y_test, prediction)
    print(report_text)

    report_dict = classification_report(
        y_test, prediction, output_dict=True
    )

    return y_test, prediction, probability, metrics, report_dict


# ======================================================
# FEATURE IMPORTANCE
# ======================================================

def feature_importance(model, X):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    })

    importance["Importance (%)"] = (importance["Importance"] * 100).round(2)

    return importance.sort_values("Importance (%)", ascending=False).head(10)


# ======================================================
# PDF PAGE HELPERS
# ======================================================

def add_footer(fig, page_num, total_pages):
    """Adds a consistent footer (date + page number) to every page."""

    today = datetime.date.today().strftime("%B %d, %Y")

    fig.text(
        0.05, 0.02,
        f"Generated {today}",
        fontsize=8, color=COLOR_TEXT_MUTED, ha="left"
    )

    fig.text(
        0.95, 0.02,
        f"Page {page_num} of {total_pages}",
        fontsize=8, color=COLOR_TEXT_MUTED, ha="right"
    )

    fig.text(
        0.5, 0.02,
        "Future Employee Performance — Model Evaluation Report",
        fontsize=8, color=COLOR_TEXT_MUTED, ha="center"
    )


def new_page():
    fig = plt.figure(figsize=PAGE_SIZE)
    return fig


# ======================================================
# CREATE PDF REPORT
# ======================================================

def create_report(df, model, y_test, prediction, probability, metrics,
                   importance, report_dict):

    print("\nCreating PDF Report...")

    TOTAL_PAGES = 6
    page = 0

    with PdfPages(PDF_FILE) as pdf:

        # --------------------------------------------
        # PAGE 1 — COVER PAGE
        # --------------------------------------------
        page += 1
        fig = new_page()
        fig.patch.set_facecolor("white")

        # Header band
        fig.add_artist(plt.Rectangle(
            (0, 0.82), 1, 0.18, transform=fig.transFigure,
            color=COLOR_PRIMARY, zorder=0
        ))

        fig.text(
            0.5, 0.90, "Employee Future Performance",
            ha="center", va="center", fontsize=24, fontweight="bold",
            color="white"
        )
        fig.text(
            0.5, 0.85, "Random Forest Model Evaluation Report",
            ha="center", va="center", fontsize=13, color="#DCE4F0"
        )

        fig.text(
            0.5, 0.72,
            f"Model type:  {type(model).__name__}",
            ha="center", fontsize=11, color="#333333"
        )
        fig.text(
            0.5, 0.68,
            f"Test set size:  {len(y_test)} records",
            ha="center", fontsize=11, color="#333333"
        )

        # Metrics summary "cards"
        metric_items = list(metrics.items())
        n = len(metric_items)
        card_w = 0.16
        gap = 0.02
        total_w = n * card_w + (n - 1) * gap
        start_x = 0.5 - total_w / 2

        for i, (key, value) in enumerate(metric_items):
            x0 = start_x + i * (card_w + gap)
            fig.add_artist(plt.Rectangle(
                (x0, 0.42), card_w, 0.16, transform=fig.transFigure,
                facecolor="#F2F5FA", edgecolor=COLOR_ACCENT2, linewidth=1
            ))
            fig.text(
                x0 + card_w / 2, 0.535, f"{value:.3f}",
                ha="center", va="center", fontsize=15, fontweight="bold",
                color=COLOR_PRIMARY
            )
            fig.text(
                x0 + card_w / 2, 0.44, key,
                ha="center", va="center", fontsize=8.5, color=COLOR_TEXT_MUTED
            )

        fig.text(
            0.5, 0.32,
            "This report summarizes classifier performance for predicting\n"
            "an employee's performance band (Low / Medium / High) in the\n"
            "next quarter, based on current-quarter KPIs and history.",
            ha="center", va="center", fontsize=10, color="#444444"
        )

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------
        # PAGE 2 — FEATURE IMPORTANCE
        # --------------------------------------------
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        imp_sorted = importance.sort_values("Importance (%)", ascending=True)
        bars = ax.barh(
            imp_sorted["Feature"], imp_sorted["Importance (%)"],
            color=COLOR_ACCENT, edgecolor="white", height=0.65
        )

        for bar, val in zip(bars, imp_sorted["Importance (%)"]):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color="#333333"
            )

        ax.set_title("Top 10 Most Influential Features", pad=15)
        ax.set_xlabel("Relative Importance (%)")
        ax.set_xlim(0, imp_sorted["Importance (%)"].max() * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(left=0.32, right=0.95, top=0.88, bottom=0.12)

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------
        # PAGE 3 — TARGET DISTRIBUTION
        # --------------------------------------------
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        counts = df["Future_Performance_Category"].value_counts().sort_index()
        labels = [CATEGORY_NAMES[i] for i in counts.index]
        colors = [BAND_COLORS[l] for l in labels]
        total = counts.sum()

        bars = ax.bar(labels, counts.values, color=colors, width=0.55,
                       edgecolor="white")

        for bar, val in zip(bars, counts.values):
            pct = val / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{val}  ({pct:.1f}%)", ha="center", fontsize=10, color="#333333"
            )

        ax.set_title("Future Performance Band — Class Distribution", pad=15)
        ax.set_ylabel("Number of Records")
        ax.set_ylim(0, counts.values.max() * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(top=0.88, bottom=0.12)

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------
        # PAGE 4 — CONFUSION MATRIX
        # --------------------------------------------
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        cm = confusion_matrix(y_test, prediction)
        labels_order = [CATEGORY_NAMES[i] for i in sorted(CATEGORY_NAMES)]

        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=True,
            xticklabels=labels_order, yticklabels=labels_order,
            annot_kws={"fontsize": 12, "fontweight": "bold"},
            linewidths=1, linecolor="white"
        )

        ax.set_title("Confusion Matrix — Predicted vs. Actual", pad=15)
        ax.set_xlabel("Predicted Band")
        ax.set_ylabel("Actual Band")
        fig.subplots_adjust(top=0.88, bottom=0.15, left=0.18)

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------
        # PAGE 5 — ROC CURVES
        # --------------------------------------------
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)

        curve_colors = [COLOR_LOW, COLOR_MED, COLOR_HIGH]

        for i in range(3):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), probability[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=curve_colors[i], linewidth=2.2,
                label=f"{CATEGORY_NAMES[i]}  (AUC = {roc_auc:.3f})"
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA",
                linewidth=1.2, label="Random baseline")

        ax.set_title("ROC Curves — One-vs-Rest by Performance Band", pad=15)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(top=0.88, bottom=0.12)

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------
        # PAGE 6 — CLASSIFICATION REPORT TABLE
        # --------------------------------------------
        page += 1
        fig, ax = plt.subplots(figsize=PAGE_SIZE)
        ax.axis("off")

        ax.set_title("Per-Class Performance Summary", pad=20, loc="center")

        rows = []
        for key in ["0", "1", "2"]:
            if key in report_dict:
                r = report_dict[key]
                rows.append([
                    CATEGORY_NAMES[int(key)],
                    f"{r['precision']:.3f}",
                    f"{r['recall']:.3f}",
                    f"{r['f1-score']:.3f}",
                    int(r["support"]),
                ])

        for avg_key, label in [("macro avg", "Macro Avg"),
                                ("weighted avg", "Weighted Avg")]:
            r = report_dict[avg_key]
            rows.append([
                label,
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['f1-score']:.3f}",
                int(r["support"]),
            ])

        col_labels = ["Band", "Precision", "Recall", "F1-Score", "Support"]

        table = ax.table(
            cellText=rows, colLabels=col_labels,
            cellLoc="center", loc="center", bbox=[0.08, 0.35, 0.84, 0.45]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(COLOR_GRID)
            if r == 0:
                cell.set_facecolor(COLOR_PRIMARY)
                cell.set_text_props(color="white", fontweight="bold")
            elif r > len(rows) - 2:
                cell.set_facecolor("#EEF1F7")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("white" if r % 2 else "#F7F9FC")

        fig.text(
            0.5, 0.24,
            f"Overall Accuracy: {metrics['Accuracy']:.3f}   |   "
            f"Weighted F1: {metrics['F1 Score']:.3f}   |   "
            f"ROC-AUC: {metrics['ROC-AUC']:.3f}",
            ha="center", fontsize=10.5, color=COLOR_PRIMARY, fontweight="bold"
        )

        add_footer(fig, page, TOTAL_PAGES)
        pdf.savefig(fig)
        plt.close(fig)

    print("\nPDF Generated:", PDF_FILE)


# ======================================================
# MAIN
# ======================================================

def main():

    print("""
====================================
MODEL EVALUATION MODULE
====================================
""")

    model = load_model()
    df = load_data()
    X, y = prepare_features(df)

    y_test, prediction, probability, metrics, report_dict = evaluate_model(
        model, X, y
    )

    importance = feature_importance(model, X)

    create_report(
        df, model, y_test, prediction, probability, metrics,
        importance, report_dict
    )


if __name__ == "__main__":
    main()