import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Employee Performance Appraisal System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Main header style */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1a365d, #2b6cb0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem 0;
        border-bottom: 3px solid #2b6cb0;
        margin-bottom: 2rem;
    }
    
    /* Sub-header */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a365d;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07), 0 1px 3px rgba(0,0,0,0.1);
        border-left: 5px solid #2b6cb0;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a202c;
        margin: 0.5rem 0;
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #718096;
    }
    
    /* Module cards */
    .module-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-top: 4px solid #4299e1;
        height: 100%;
        transition: transform 0.2s ease;
    }
    
    .module-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .module-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .module-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #1a202c;
        margin-bottom: 0.5rem;
    }
    
    .module-desc {
        font-size: 0.9rem;
        color: #718096;
        line-height: 1.5;
    }
    
    /* Info boxes */
    .info-box {
        background: #ebf8ff;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #3182ce;
        margin: 1rem 0;
    }
    
    .info-box-title {
        font-weight: 600;
        color: #1a365d;
        margin-bottom: 0.5rem;
    }
    
    /* Feature list */
    .feature-list {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .feature-item {
        padding: 0.25rem 0;
        color: #2d3748;
    }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-active {
        background: #48bb78;
        color: white;
    }
    
    .badge-concept {
        background: #ed8936;
        color: white;
    }
    
    .badge-future {
        background: #4299e1;
        color: white;
    }
    
    /* Sidebar */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #2b6cb0;
        padding: 1rem 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .sidebar-module {
        padding: 0.75rem 0.5rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .sidebar-module:hover {
        background: #ebf8ff;
    }
    
    .sidebar-module-active {
        background: #ebf8ff;
        border-left: 3px solid #2b6cb0;
    }
    
    /* Section divider */
    .section-divider {
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #e2e8f0, transparent);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            🧠 AI Appraisal System
        </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("### 📋 Modules")

    modules = [
        {"icon": "📊", "name": "Dashboard", "active": True},
        {"icon": "🎯", "name": "Performance Prediction", "active": False},
        {"icon": "🧠", "name": "Behavior Pattern", "active": False},
        {"icon": "⚖️", "name": "Bias Detection", "active": False},
        {"icon": "🤝", "name": "Conflict Resolution", "active": False}
    ]

    for module in modules:
        if module["active"]:
            st.markdown(f"""
                <div class="sidebar-module sidebar-module-active">
                    {module['icon']} {module['name']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="sidebar-module">
                    {module['icon']} {module['name']}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # System info
    st.markdown("### ℹ️ System Info")
    st.caption("**Version:** 1.0.0")
    st.caption("**Status:** Conceptual Stage")
    st.caption("**Framework:** AI-Based Appraisal")
    st.caption("**Sector:** Sri Lankan IT")

    st.markdown("---")

    # Research info
    st.markdown("### 📚 Research Context")
    st.caption("""
        This system is designed as part of a research study 
        on AI-based employee performance appraisal in the 
        Sri Lankan IT sector.
    """)

# Main content
st.markdown("""
    <div class="main-header">
        🏢 AI-Based Employee Performance Appraisal System
    </div>
""", unsafe_allow_html=True)

# Context info box
st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📌 Conceptual Framework Overview</div>
        <p style="margin: 0; color: #2d3748;">
            This system presents a conceptual AI-based employee performance appraisal framework 
            designed specifically for the <strong>Sri Lankan IT sector</strong>. The framework 
            integrates four interconnected modules to provide comprehensive, fair, and 
            evidence-based employee evaluations.
        </p>
    </div>
""", unsafe_allow_html=True)

# Key Metrics Row
st.markdown('<div class="sub-header">📈 System Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="metric-card" style="border-left-color: #4299e1;">
            <div class="metric-title">Modules</div>
            <div class="metric-value">4</div>
            <div class="metric-sub">Integrated AI modules</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card" style="border-left-color: #48bb78;">
            <div class="metric-title">Performance Factors</div>
            <div class="metric-value">9</div>
            <div class="metric-sub">Key performance indicators</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card" style="border-left-color: #ed8936;">
            <div class="metric-title">Behavioral Indicators</div>
            <div class="metric-value">7</div>
            <div class="metric-sub">Behavioral metrics</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="metric-card" style="border-left-color: #9f7aea;">
            <div class="metric-title">ML Algorithm</div>
            <div class="metric-value">Random Forest</div>
            <div class="metric-sub">Classification model</div>
        </div>
    """, unsafe_allow_html=True)

# Module Overview
st.markdown('<div class="sub-header">🧩 System Modules</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Module 1
    st.markdown("""
        <div class="module-card" style="border-top-color: #4299e1;">
            <div class="module-icon">🎯</div>
            <div class="module-title">Module 1: Performance Prediction</div>
            <div class="module-desc">
                Predicts employee performance levels using Random Forest classification 
                algorithm based on nine key performance indicators.
            </div>
            <br>
            <span class="badge badge-concept">Conceptual Stage</span>
            <span class="badge badge-active" style="margin-left: 0.5rem;">ML Ready</span>
        </div>
    """, unsafe_allow_html=True)

    # Module 3
    st.markdown("""
        <div class="module-card" style="border-top-color: #ed8936;">
            <div class="module-icon">⚖️</div>
            <div class="module-title">Module 3: Bias Detection & Fairness</div>
            <div class="module-desc">
                Identifies and mitigates bias in appraisals using statistical methods 
                and NLP techniques for fair and transparent evaluations.
            </div>
            <br>
            <span class="badge badge-concept">Conceptual Stage</span>
            <span class="badge badge-future" style="margin-left: 0.5rem;">NLP Ready</span>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Module 2
    st.markdown("""
        <div class="module-card" style="border-top-color: #48bb78;">
            <div class="module-icon">🧠</div>
            <div class="module-title">Module 2: Behavior Pattern Identification</div>
            <div class="module-desc">
                Identifies behavioral patterns using NLP and clustering algorithms 
                from structured and unstructured feedback data.
            </div>
            <br>
            <span class="badge badge-concept">Conceptual Stage</span>
            <span class="badge badge-future" style="margin-left: 0.5rem;">NLP Ready</span>
        </div>
    """, unsafe_allow_html=True)

    # Module 4
    st.markdown("""
        <div class="module-card" style="border-top-color: #9f7aea;">
            <div class="module-icon">🤝</div>
            <div class="module-title">Module 4: Conflict Resolution</div>
            <div class="module-desc">
                Resolves appraisal conflicts using opinion dynamics theory and 
                weighted aggregation for balanced decision-making.
            </div>
            <br>
            <span class="badge badge-concept">Conceptual Stage</span>
            <span class="badge badge-future" style="margin-left: 0.5rem;">Theory Ready</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Module 1 Detailed View
st.markdown('<div class="sub-header">🎯 Module 1: Performance Prediction Module</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
        <div style="background: #f7fafc; padding: 1.5rem; border-radius: 12px;">
            <h4 style="color: #1a365d;">Overview</h4>
            <p style="color: #2d3748; line-height: 1.6;">
                The Performance Prediction Module is designed to predict employee performance 
                levels using structured performance-related data. This module aims to improve 
                consistency, objectivity, and transparency in employee evaluation within 
                Sri Lankan IT organizations.
            </p>
            <br>
            <h4 style="color: #1a365d;">Key Input Attributes</h4>
            <div class="feature-list">
                <div class="feature-item">• KPI Achievement Rate</div>
                <div class="feature-item">• Task Completion Rate</div>
                <div class="feature-item">• Attendance Records</div>
                <div class="feature-item">• On-Time Delivery Rate</div>
                <div class="feature-item">• Work Consistency Score</div>
                <div class="feature-item">• Bug Resolution Rate</div>
                <div class="feature-item">• Code Quality Score</div>
                <div class="feature-item">• Past Performance Scores</div>
                <div class="feature-item">• Workload Level</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #ebf8ff, #bee3f8); padding: 1.5rem; border-radius: 12px;">
            <h4 style="color: #1a365d;">🎯 Output</h4>
            <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <strong>High Performance</strong>
                <div style="height: 4px; background: #48bb78; border-radius: 2px; margin-top: 0.25rem;"></div>
            </div>
            <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <strong>Medium Performance</strong>
                <div style="height: 4px; background: #ed8936; border-radius: 2px; margin-top: 0.25rem;"></div>
            </div>
            <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <strong>Low Performance</strong>
                <div style="height: 4px; background: #fc8181; border-radius: 2px; margin-top: 0.25rem;"></div>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #2d3748;">
                <span class="badge badge-concept">Algorithm: Random Forest</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Technology Stack
st.markdown('<div class="sub-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)

tech_col1, tech_col2, tech_col3, tech_col4, tech_col5 = st.columns(5)

tech_stack = [
    {"name": "Python", "icon": "🐍", "color": "#3776AB"},
    {"name": "Pandas", "icon": "🐼", "color": "#150458"},
    {"name": "NumPy", "icon": "🔢", "color": "#013243"},
    {"name": "Scikit-learn", "icon": "🤖", "color": "#F7931E"},
    {"name": "Streamlit", "icon": "✨", "color": "#FF4B4B"}
]

for idx, tech in enumerate(tech_stack):
    with [tech_col1, tech_col2, tech_col3, tech_col4, tech_col5][idx]:
        st.markdown(f"""
            <div style="
                background: white;
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid #e2e8f0;
            ">
                <div style="font-size: 2.5rem;">{tech['icon']}</div>
                <div style="font-weight: 600; color: #1a202c; margin-top: 0.5rem;">{tech['name']}</div>
            </div>
        """, unsafe_allow_html=True)

# Appraisal Process Flow
st.markdown('<div class="sub-header">🔄 Appraisal Process Flow</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="
        background: #f7fafc;
        padding: 1.5rem;
        border-radius: 12px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    ">
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 2rem;">📊</div>
            <div style="font-weight: 600; font-size: 0.9rem;">Data Collection</div>
            <div style="font-size: 0.8rem; color: #718096;">Performance metrics</div>
        </div>
        <div style="font-size: 1.5rem; color: #2b6cb0;">→</div>
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 2rem;">🤖</div>
            <div style="font-weight: 600; font-size: 0.9rem;">ML Analysis</div>
            <div style="font-size: 0.8rem; color: #718096;">Random Forest</div>
        </div>
        <div style="font-size: 1.5rem; color: #2b6cb0;">→</div>
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 2rem;">🧠</div>
            <div style="font-weight: 600; font-size: 0.9rem;">Pattern Detection</div>
            <div style="font-size: 0.8rem; color: #718096;">Behavioral insights</div>
        </div>
        <div style="font-size: 1.5rem; color: #2b6cb0;">→</div>
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 2rem;">⚖️</div>
            <div style="font-weight: 600; font-size: 0.9rem;">Fairness Check</div>
            <div style="font-size: 0.8rem; color: #718096;">Bias mitigation</div>
        </div>
        <div style="font-size: 1.5rem; color: #2b6cb0;">→</div>
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 2rem;">📋</div>
            <div style="font-weight: 600; font-size: 0.9rem;">Final Evaluation</div>
            <div style="font-size: 0.8rem; color: #718096;">Decision output</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Research Context
st.markdown('<div class="sub-header">📚 Research Context</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2b6cb0;
    ">
        <h4 style="color: #1a365d; margin-top: 0;">Sri Lankan IT Sector Context</h4>
        <p style="color: #2d3748; line-height: 1.6;">
            This conceptual framework is specifically designed to address the unique challenges 
            of employee performance appraisal in the Sri Lankan IT sector, including:
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div style="background: #f7fafc; padding: 0.75rem; border-radius: 8px;">
                <strong>🔹 High Power Distance</strong>
                <div style="font-size: 0.85rem; color: #718096;">Addressing hierarchical influence in evaluations</div>
            </div>
            <div style="background: #f7fafc; padding: 0.75rem; border-radius: 8px;">
                <strong>🔹 Cultural Diversity</strong>
                <div style="font-size: 0.85rem; color: #718096;">Ensuring culturally fair assessments</div>
            </div>
            <div style="background: #f7fafc; padding: 0.75rem; border-radius: 8px;">
                <strong>🔹 Skill-Based Industry</strong>
                <div style="font-size: 0.85rem; color: #718096;">Technical and soft skill evaluation</div>
            </div>
            <div style="background: #f7fafc; padding: 0.75rem; border-radius: 8px;">
                <strong>🔹 Fast-Paced Environment</strong>
                <div style="font-size: 0.85rem; color: #718096;">Dynamic performance tracking</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #718096; font-size: 0.85rem;">
        <strong>🏢 Conceptual AI-Based Employee Performance Appraisal System</strong><br>
        Designed for the Sri Lankan IT Sector | Research Framework | All data is simulated for demonstration
        <br><br>
        <span style="font-size: 0.75rem; color: #a0aec0;">
            © 2024 Research Project | Version 1.0.0
        </span>
    </div>
""", unsafe_allow_html=True)