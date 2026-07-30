import streamlit as st
import pandas as pd
import json

from src.validator import validate_dataset
from src.preprocessing import preprocess_data
from src.performance_score import generate_performance_score
from src.fairness_engine import run_fairness_analysis



# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="AI Employee Fairness Dashboard",
    page_icon="⚖️",
    layout="wide"
)



# ==============================
# TITLE
# ==============================

st.title(
    "⚖️ AI-Based Employee Performance Fairness Dashboard"
)


st.write(
    """
    This dashboard analyzes employee performance appraisal data
    to identify potential bias and evaluate fairness using
    AI-based fairness metrics.
    """
)



# ==============================
# UPLOAD DATASET
# ==============================

st.sidebar.header(
    "Upload Dataset"
)


uploaded_file = st.sidebar.file_uploader(
    "Upload Employee Dataset CSV",
    type=["csv"]
)



if uploaded_file:


    # ==============================
    # LOAD DATA
    # ==============================

    df = pd.read_csv(
        uploaded_file
    )

    st.write(df["Gender"].value_counts())

    st.subheader(
        "📄 Dataset Preview"
    )


    st.dataframe(
        df.head()
    )



    # ==============================
    # VALIDATION
    # ==============================

    st.subheader(
        "✅ Dataset Validation"
    )


    validate_dataset(
        df
    )


    st.success(
        "Dataset validation completed"
    )



    # ==============================
    # PREPROCESS
    # ==============================

    processed_data = preprocess_data(
        df.copy()
    )



    processed_data = generate_performance_score(
        processed_data
    )



    # Restore demographic values

    if "Gender" in df.columns:

        processed_data["Gender"] = (
            df["Gender"]
            .values
        )


    if "Ethnicity" in df.columns:

        processed_data["Ethnicity"] = (
            df["Ethnicity"]
            .values
        )



    # ==============================
    # FAIRNESS ANALYSIS
    # ==============================

    fairness_results = run_fairness_analysis(
        processed_data
    )



    st.divider()



    # ==============================
    # DISPLAY GENDER BIAS
    # ==============================

    st.header(
        "👥 Gender Fairness Analysis"
    )


    gender_result = (
        fairness_results.get(
            "gender_bias",
            {}
        )
    )



    if (
        gender_result
        and
        "error" not in gender_result
    ):


        gender_scores = (
            gender_result.get(
                "average_score_by_gender",
                {}
            )
        )


        gender_df = pd.DataFrame(
            gender_scores.items(),
            columns=[
                "Gender",
                "Average Performance Score"
            ]
        )



        col1, col2 = st.columns(2)



        with col1:

            st.metric(
                "Bias Gap",
                gender_result.get(
                    "bias_gap",
                    "N/A"
                )
            )



        with col2:

            st.metric(
                "Status",
                gender_result.get(
                    "status",
                    "N/A"
                )
            )



        st.bar_chart(
            gender_df.set_index(
                "Gender"
            )
        )


    else:

        st.warning(
            "Gender fairness data unavailable"
        )



    st.divider()



    # ==============================
    # ETHNICITY BIAS
    # ==============================

    st.header(
        "🌍 Ethnicity Fairness Analysis"
    )


    ethnicity_result = (
        fairness_results.get(
            "ethnicity_bias",
            {}
        )
    )



    if (
        ethnicity_result
        and
        "error" not in ethnicity_result
    ):


        ethnicity_scores = (
            ethnicity_result.get(
                "average_score_by_ethnicity",
                {}
            )
        )


        ethnicity_df = pd.DataFrame(

            ethnicity_scores.items(),

            columns=[
                "Ethnicity",
                "Average Performance Score"
            ]

        )



        col1, col2 = st.columns(2)



        with col1:

            st.metric(
                "Bias Gap",
                ethnicity_result.get(
                    "bias_gap",
                    "N/A"
                )
            )



        with col2:

            st.metric(
                "Status",
                ethnicity_result.get(
                    "status",
                    "N/A"
                )
            )



        st.bar_chart(
            ethnicity_df.set_index(
                "Ethnicity"
            )
        )


    else:

        st.warning(
            "Ethnicity fairness data unavailable"
        )



    st.divider()



    # ==============================
    # FAIRNESS METRICS
    # ==============================

    st.header(
        "⚖️ Fairness Metrics"
    )


    metrics = (
        fairness_results.get(
            "fairness_metrics",
            {}
        )
    )



    demographic = metrics.get(
        "demographic_parity"
    )


    disparate = metrics.get(
        "disparate_impact"
    )



    col1, col2 = st.columns(2)



    with col1:


        if demographic:


            st.metric(
                "Demographic Parity Ratio",
                demographic.get(
                    "parity_ratio",
                    "N/A"
                )
            )


        else:

            st.metric(
                "Demographic Parity Ratio",
                "N/A"
            )



    with col2:


        if disparate:


            st.metric(
                "Disparate Impact Ratio",
                disparate.get(
                    "disparate_impact_ratio",
                    "N/A"
                )
            )


        else:

            st.metric(
                "Disparate Impact Ratio",
                "N/A"
            )



    st.success(
        "Fairness analysis completed successfully"
    )



    # ==============================
    # DOWNLOAD REPORT
    # ==============================

    st.divider()


    st.header(
        "📥 Download Report"
    )


    report = json.dumps(
        fairness_results,
        indent=4
    )


    st.download_button(

        label="Download Fairness Report",

        data=report,

        file_name="fairness_report.json",

        mime="application/json"

    )



else:


    st.info(
        "Please upload employee dataset to begin analysis."
    )