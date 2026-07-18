import streamlit as st

st.set_page_config(
    page_title="AI Employee Performance Appraisal System",
    page_icon="🏢",
    layout="wide"
)

# Redirect to dashboard
st.switch_page("pages/dashboard.py")