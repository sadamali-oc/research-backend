# ======================================================
# AI EMPLOYEE FUTURE PERFORMANCE PREDICTION DASHBOARD
# ======================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ======================================================
# CONFIGURATION
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

RESULT_FILE = (
    BASE_DIR /
    "results" /
    "employee_performance_predictions.xlsx"
)

PDF_FILE = (
    BASE_DIR /
    "results" /
    "Final_Model_Evaluation_Report.pdf"
)


# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="AI Employee Performance Prediction",
    page_icon="🤖",
    layout="wide"
)


# ======================================================
# TITLE
# ======================================================

st.title(
    "🤖 AI-Based Employee Future Performance Prediction System"
)

st.write(
    """
    Interactive dashboard for analysing AI-generated employee
    future performance predictions using Machine Learning and
    Explainable AI insights.
    """
)


# ======================================================
# LOAD PREDICTIONS
# ======================================================

@st.cache_data
def load_data():

    return pd.read_excel(
        RESULT_FILE
    )


try:

    df = load_data()

except:

    st.error(
        "Prediction result file not found."
    )

    st.stop()



# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header(
    "Dashboard Filters"
)


selected_band = st.sidebar.multiselect(

    "Performance Category",

    options=df[
        "predicted_performance_band"
    ].unique(),

    default=df[
        "predicted_performance_band"
    ].unique()

)


employee_id = st.sidebar.text_input(
    "Search Employee ID"
)



filtered_df = df[
    df[
        "predicted_performance_band"
    ].isin(selected_band)
]



if employee_id:

    filtered_df = filtered_df[
        filtered_df[
            "employee_id"
        ]
        .astype(str)
        .str.contains(
            employee_id,
            case=False
        )
    ]



# ======================================================
# SUMMARY DASHBOARD
# ======================================================

st.header(
    "📊 Performance Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Employees",
        len(filtered_df)
    )


with col2:

    st.metric(
        "Average Predicted Score",

        round(
            filtered_df[
                "predicted_performance_score"
            ].mean(),
            2
        )
    )


with col3:

    st.metric(

        "High Performers",

        (
            filtered_df[
                "predicted_performance_band"
            ]
            ==
            "High"
        )
        .sum()

    )


with col4:

    st.metric(

        "Average Confidence",

        str(
            round(
                filtered_df[
                    "model_confidence"
                ]
                .mean()
                *
                100,
                2
            )
        )
        + "%"

    )



# ======================================================
# TABS
# ======================================================


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Predictions",
        "📈 Analytics",
        "🔍 Explainable AI",
        "📄 Reports"
    ]
)



# ======================================================
# TAB 1 - PREDICTIONS
# ======================================================


with tab1:


    st.subheader(
        "Employee Future Performance Predictions"
    )


    st.dataframe(

        filtered_df,

        use_container_width=True

    )



    st.subheader(
        "🏆 Employee Ranking"
    )


    ranking = (

        filtered_df
        .sort_values(

            "predicted_performance_score",

            ascending=False

        )
        .head(10)

    )


    st.dataframe(

        ranking[
            [
                "employee_id",
                "predicted_performance_score",
                "predicted_performance_band",
                "model_confidence"
            ]
        ],

        use_container_width=True

    )



# ======================================================
# TAB 2 - ANALYTICS
# ======================================================


with tab2:


    st.subheader(
        "Performance Category Distribution"
    )


    fig, ax = plt.subplots()


    sns.countplot(

        data=filtered_df,

        x="predicted_performance_band",

        ax=ax

    )


    st.pyplot(fig)



    st.subheader(
        "Performance Score Distribution"
    )


    fig, ax = plt.subplots()


    sns.histplot(

        filtered_df[
            "predicted_performance_score"
        ],

        bins=20,

        kde=True,

        ax=ax

    )


    st.pyplot(fig)



    st.subheader(
        "Confidence Level Distribution"
    )


    fig, ax = plt.subplots()


    sns.histplot(

        filtered_df[
            "model_confidence"
        ],

        bins=20,

        ax=ax

    )


    st.pyplot(fig)




# ======================================================
# TAB 3 - XAI
# ======================================================


with tab3:


    st.subheader(
        "🔍 Explainable AI Prediction Analysis"
    )


    selected_employee = st.selectbox(

        "Select Employee",

        filtered_df[
            "employee_id"
        ].unique()

    )


    employee = filtered_df[

        filtered_df[
            "employee_id"
        ]
        ==
        selected_employee

    ]



    if not employee.empty:


        st.write(
            "### Prediction Result"
        )


        st.info(

            employee[
                "predicted_performance_band"
            ]
            .values[0]

        )


        st.write(
            "Confidence:"
        )


        st.success(

            employee[
                "model_confidence"
            ]
            .values[0]

        )


        st.write(
            "Important Performance Factors:"
        )


        st.warning(

            employee[
                "top_5_influential_factors"
            ]
            .values[0]

        )


        st.write(
            "Recommendation:"
        )


        st.success(

            employee[
                "recommendation"
            ]
            .values[0]

        )




# ======================================================
# TAB 4 - REPORT EXPORT
# ======================================================


with tab4:


    st.subheader(
        "📄 Export Results"
    )


    excel_file = (

        filtered_df
        .to_excel(

            "employee_predictions.xlsx",

            index=False

        )

    )


    with open(
        "employee_predictions.xlsx",
        "rb"
    ) as file:


        st.download_button(

            "Download Prediction Results",

            file,

            file_name=
            "employee_predictions.xlsx"

        )



    if PDF_FILE.exists():


        with open(
            PDF_FILE,
            "rb"
        ) as file:


            st.download_button(

                "Download Evaluation Report",

                file,

                file_name=
                "Final_Model_Evaluation_Report.pdf",

                mime=
                "application/pdf"

            )



st.success(
    "Dashboard loaded successfully."
)