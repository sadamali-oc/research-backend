import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

API_URL = "http://localhost:8000"

def main():
    # REMOVED: st.set_page_config() - already set in app.py

    st.markdown("""
        <style>
        .main-header {
            ffont-size: 2.5rem;
            font-weight: 700;
            /* Swapped dark blue for bright neon/sky blue tones that pop on black and white */
            background: linear-gradient(120deg, #63b3ed, #4299e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            padding: 1rem 0;
            border-bottom: 3px solid #4299e1;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: white;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        .metric-card {
            background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            border-left: 5px solid #2b6cb0;
            text-align: center;
            color: #1a202c !important;
        }
        .module-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-top: 4px solid #4299e1;
            color: #1a202c !important;
        }
        .metric-card *, .module-card * {
        color: #1a202c !important;
        }
        .api-status-online {
            background: #c6f6d5;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border-left: 4px solid #48bb78;
            color: #22543d !important; /* Dark green text */
        }
        .api-status-offline {
            background: #fed7d7;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border-left: 4px solid #fc8181;
            color: #744210 !important; /* Dark orange text */
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🏢 AI-Based Employee Performance Appraisal System</div>', unsafe_allow_html=True)

    # Check API connection
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        api_connected = response.status_code == 200
    except:
        api_connected = False

    if api_connected:
        st.markdown('<div class="api-status-online">✅ Backend API: Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status-offline">❌ Backend API: Offline</div>', unsafe_allow_html=True)
        st.warning("Please start the backend server first.")
        return

    # Get data from API
    try:
        emp_response = requests.get(f"{API_URL}/api/employees/?limit=1")
        if emp_response.status_code == 200:
            emp_data = emp_response.json()
            total_employees = emp_data.get('total', 0)
        else:
            total_employees = 0

        quarter_response = requests.get(f"{API_URL}/api/employees/quarters/")
        if quarter_response.status_code == 200:
            quarter_data = quarter_response.json()
            total_quarters = quarter_data.get('total_quarters', 0)
        else:
            total_quarters = 0

        stats_response = requests.get(f"{API_URL}/api/predictions/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
        else:
            stats = {'total_predictions': 0}

    except:
        total_employees = 0
        total_quarters = 0
        stats = {'total_predictions': 0}

    # Metrics Row
    st.markdown('<div class="sub-header">📈 System Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #4299e1;">
                <div class="metric-title">Total Employees</div>
                <div style="font-size: 2rem; font-weight: 700;">{total_employees}</div>
                <div style="color: #718096;">👥 Registered in system</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #48bb78;">
                <div class="metric-title">Quarters Tracked</div>
                <div style="font-size: 2rem; font-weight: 700;">{total_quarters}</div>
                <div style="color: #718096;">📅 Historical data available</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #9f7aea;">
                <div class="metric-title">Predictions Made</div>
                <div style="font-size: 2rem; font-weight: 700;">{stats.get('total_predictions', 0)}</div>
                <div style="color: #718096;">🎯 Performance predictions</div>
            </div>
        """, unsafe_allow_html=True)

    # System Modules
    st.markdown('<div class="sub-header">🧩 System Modules</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="module-card" style="border-top-color: #4299e1;">
                <div style="font-size: 2.5rem;">🎯</div>
                <div style="font-weight: 600; font-size: 1.1rem;">Module 1: Performance Prediction</div>
                <div style="color: #718096; font-size: 0.9rem;">
                    Predicts employee performance using hybrid ML algorithms
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="module-card" style="border-top-color: #ed8936;">
                <div style="font-size: 2.5rem;">⚖️</div>
                <div style="font-weight: 600; font-size: 1.1rem;">Module 3: Bias Detection & Fairness</div>
                <div style="color: #718096; font-size: 0.9rem;">
                    Identifies and mitigates bias in appraisals
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="module-card" style="border-top-color: #48bb78;">
                <div style="font-size: 2.5rem;">🧠</div>
                <div style="font-weight: 600; font-size: 1.1rem;">Module 2: Behavior Pattern Identification</div>
                <div style="color: #718096; font-size: 0.9rem;">
                    Identifies behavioral patterns using NLP
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="module-card" style="border-top-color: #9f7aea;">
                <div style="font-size: 2.5rem;">🤝</div>
                <div style="font-weight: 600; font-size: 1.1rem;">Module 4: Conflict Resolution</div>
                <div style="color: #718096; font-size: 0.9rem;">
                    Resolves appraisal conflicts using opinion dynamics
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🏢 Conceptual AI-Based Employee Performance Appraisal System | Version 2.0.0")

if __name__ == "__main__":
    main()