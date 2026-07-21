# frontend/app.py
import streamlit as st
import requests

# API Configuration
API_URL = "http://localhost:8000"

# Cache functions with API_URL
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_employees():
    try:
        response = requests.get(f"{API_URL}/api/employees/?limit=1000", timeout=10)
        if response.status_code == 200:
            return response.json().get('employees', [])
    except:
        pass
    return []

@st.cache_data(ttl=300)
def get_employee_history(employee_id):
    try:
        response = requests.get(f"{API_URL}/api/employees/history/{employee_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

@st.cache_data(ttl=300)
def get_employee_predictions(employee_id):
    try:
        response = requests.get(f"{API_URL}/api/predictions/employee/{employee_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

@st.cache_data(ttl=60)
def get_prediction_stats():
    try:
        response = requests.get(f"{API_URL}/api/predictions/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

st.set_page_config(
    page_title="AI Employee Performance Appraisal System",
    page_icon="🏢",
    layout="wide"
)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

# Function to navigate
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# Sidebar navigation
with st.sidebar:
    st.markdown("### 🧠 AI Appraisal System")
    st.markdown("---")

    if st.button("📊 Dashboard", use_container_width=True):
        navigate_to('dashboard')

    if st.button("🎯 Performance Prediction", use_container_width=True):
        navigate_to('performance')

    if st.button("📊 360 Feedback", use_container_width=True):
        navigate_to('feedback_preprocess')

    st.markdown("---")
    st.caption(f"Current Page: {st.session_state.page}")
    st.caption("Version: 2.0.0")

# Page routing
page = st.session_state.page

if page == 'dashboard':
    from pages.dashboard import main as dashboard_page
    dashboard_page()
elif page == 'performance':
    from pages.performance import main as performance_page
    performance_page()
elif page == 'feedback_preprocess':
    from pages.feedback_preprocess import main as feedback_page
    feedback_page()
else:
    from pages.dashboard import main as dashboard_page
    dashboard_page()