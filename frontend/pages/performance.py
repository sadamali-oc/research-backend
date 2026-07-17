import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

# API Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Performance Prediction Module",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
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
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #2b6cb0;
    }
    
    .band-high {
        background: #48bb78;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .band-medium {
        background: #ed8936;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .band-low {
        background: #fc8181;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .metric-box {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎯 Module 1: Performance Prediction</div>', unsafe_allow_html=True)

# Check API connection
try:
    response = requests.get(f"{API_URL}/health")
    api_connected = response.status_code == 200
except:
    api_connected = False

if not api_connected:
    st.error("❌ Cannot connect to backend API. Please start the backend server first.")
    st.code("python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
    st.stop()

# Sidebar - Training Controls
with st.sidebar:
    st.markdown("### 🧠 Model Training")

    if st.button("🔄 Train Model"):
        with st.spinner("Training model on employee data..."):
            try:
                response = requests.post(f"{API_URL}/api/predictions/train")
                result = response.json()

                if result.get('success'):
                    st.success(f"✅ {result.get('message')}")
                    if result.get('accuracy'):
                        st.metric("Accuracy", f"{result['accuracy']:.2%}")
                    if result.get('f1_score'):
                        st.metric("F1 Score", f"{result['f1_score']:.2%}")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('message')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.markdown("---")
    st.markdown("### 📊 Model Status")

    try:
        stats = requests.get(f"{API_URL}/api/predictions/stats").json()
        if stats.get('total_predictions', 0) > 0:
            st.success("✅ Model is ready")
            st.metric("Total Predictions", stats.get('total_predictions', 0))
        else:
            st.warning("⚠️ No predictions yet")
    except:
        st.warning("⚠️ Model not trained")

    st.markdown("---")
    st.markdown("### 🔍 Search Employee")
    st.caption("Search by Employee ID")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    # Employee Search
    st.markdown("### 🔍 Employee Search")

    search_term = st.text_input("Enter Employee ID", placeholder="e.g., 1, EMP001")

    if search_term:
        try:
            response = requests.get(f"{API_URL}/api/employees/search?q={search_term}")
            employees = response.json()

            if employees:
                selected_employee = st.selectbox(
                    "Select Employee",
                    employees,
                    format_func=lambda x: f"{x['employee_id']} - {x['job_role']}"
                )
            else:
                st.warning("No matching employees found")
                selected_employee = None
        except:
            st.error("Error searching employees")
            selected_employee = None
    else:
        # Get all employees
        try:
            response = requests.get(f"{API_URL}/api/employees?limit=100")
            data = response.json()
            employees = data.get('employees', [])

            if employees:
                selected_employee = st.selectbox(
                    "Select Employee",
                    employees,
                    format_func=lambda x: f"{x['employee_id']} - {x['job_role']}"
                )
            else:
                st.warning("No employees found")
                selected_employee = None
        except:
            st.error("Error loading employees")
            selected_employee = None

    # Predict button
    if selected_employee:
        if st.button("🎯 Predict Performance", use_container_width=True):
            with st.spinner(f"Predicting performance..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/predictions/predict",
                        json={"employee_id": selected_employee['employee_id']}
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['prediction_result'] = result
                        st.success("✅ Prediction complete!")
                    else:
                        st.error(f"❌ Prediction failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    # Display prediction result
    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']

        # Prediction card
        band = result['performance_band']
        band_class = f"band-{band.lower()}"

        st.markdown(f"""
            <div class="prediction-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="margin: 0;">Employee: {result['employee_id']}</h3>
                    </div>
                    <div class="{band_class}">
                        {band} Performer
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                    <div class="metric-box">
                        <div style="font-size: 0.85rem; color: #718096;">Performance Score</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #1a202c;">
                            {result['performance_score']:.1f}%
                        </div>
                    </div>
                    <div class="metric-box">
                        <div style="font-size: 0.85rem; color: #718096;">Confidence</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #1a202c;">
                            {result['confidence']:.1%}
                        </div>
                    </div>
                    <div class="metric-box">
                        <div style="font-size: 0.85rem; color: #718096;">Predicted At</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #1a202c;">
                            {result['predicted_at'][:10]}
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Feature importance
        if result.get('top_features'):
            st.markdown("### 📊 Top Factors Influencing Performance")

            features_df = pd.DataFrame(
                result['top_features'],
                columns=['Feature', 'Importance']
            )

            fig = go.Figure(data=[
                go.Bar(
                    x=features_df['Importance'],
                    y=features_df['Feature'],
                    orientation='h',
                    marker_color='#2b6cb0',
                    text=features_df['Importance'].apply(lambda x: f'{x:.3f}'),
                    textposition='outside'
                )
            ])

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title='Feature Importance',
                yaxis_title=''
            )

            st.plotly_chart(fig, use_container_width=True)

# Performance Distribution Histogram
st.markdown("---")
st.markdown("### 📊 Performance Distribution")

try:
    # Get all predictions
    response = requests.get(f"{API_URL}/api/predictions/all")
    if response.status_code == 200:
        predictions = response.json()

        if predictions:
            df_predictions = pd.DataFrame(predictions)

            # Create histogram
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Performance Score Distribution', 'Performance Band Distribution'),
                specs=[[{'type': 'histogram'}, {'type': 'pie'}]]
            )

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=df_predictions['performance_score'],
                    nbinsx=20,
                    marker_color='#2b6cb0',
                    name='Scores'
                ),
                row=1, col=1
            )

            # Pie chart for bands
            band_counts = df_predictions['performance_band'].value_counts()
            colors = {'High': '#48bb78', 'Medium': '#ed8936', 'Low': '#fc8181'}

            fig.add_trace(
                go.Pie(
                    labels=band_counts.index,
                    values=band_counts.values,
                    marker_colors=[colors.get(b, '#4299e1') for b in band_counts.index],
                    name='Bands'
                ),
                row=1, col=2
            )

            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show predictions table
            st.markdown("### 📋 All Predictions")

            display_df = df_predictions[['employee_id', 'performance_band', 'performance_score', 'confidence', 'predicted_at']]
            display_df.columns = ['Employee ID', 'Performance Band', 'Score', 'Confidence', 'Predicted At']
            display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.1f}%")
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
            display_df['Predicted At'] = pd.to_datetime(display_df['Predicted At']).dt.strftime('%Y-%m-%d %H:%M')

            st.dataframe(display_df, use_container_width=True)

            # Stats
            stats = requests.get(f"{API_URL}/api/predictions/stats").json()
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Predictions", stats.get('total_predictions', 0))
                with col2:
                    st.metric("High Performers", stats.get('high_performers', 0))
                with col3:
                    st.metric("Medium Performers", stats.get('medium_performers', 0))
                with col4:
                    st.metric("Low Performers", stats.get('low_performers', 0))
        else:
            st.info("No predictions available yet. Train the model and make predictions for employees.")
    else:
        st.error("Failed to fetch predictions")

except Exception as e:
    st.error(f"Error loading predictions: {str(e)}")

# Footer
st.markdown("---")
st.caption("🎯 Performance Prediction Module - Powered by Random Forest Classifier")