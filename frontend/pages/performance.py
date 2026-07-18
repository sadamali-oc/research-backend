import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from datetime import datetime
import time

# API Configuration
API_URL = "http://localhost:8000"

# Cache the health check
@st.cache_data(ttl=30)
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=60)
def get_prediction_stats():
    try:
        response = requests.get(f"{API_URL}/api/predictions/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

@st.cache_data(ttl=60)
def get_all_predictions():
    try:
        response = requests.get(f"{API_URL}/api/predictions/all", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def main():
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
        .sub-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #1a365d;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        .band-high { background: #48bb78; color: white; padding: 0.3rem 1rem; border-radius: 20px; font-weight: 600; display: inline-block; }
        .band-medium { background: #ed8936; color: white; padding: 0.3rem 1rem; border-radius: 20px; font-weight: 600; display: inline-block; }
        .band-low { background: #fc8181; color: white; padding: 0.3rem 1rem; border-radius: 20px; font-weight: 600; display: inline-block; }
        .training-step {
            background: #f7fafc;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #2b6cb0;
        }
        .training-step-success { border-left-color: #48bb78; background: #f0fff4; }
        .training-step-running { border-left-color: #ed8936; background: #fffaf0; animation: pulse 1.5s ease-in-out infinite; }
        .training-step-error { border-left-color: #fc8181; background: #fff5f5; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
        .step-icon { font-size: 1.5rem; min-width: 2rem; }
        .step-content { flex: 1; }
        .step-title { font-weight: 600; color: #1a202c; }
        .step-detail { font-size: 0.9rem; color: #718096; }
        .stat-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            border-top: 3px solid #2b6cb0;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1a202c;
        }
        .stat-label {
            font-size: 0.85rem;
            color: #718096;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🎯 Performance Prediction Module</div>', unsafe_allow_html=True)

    # Check API connection with caching
    api_connected = check_api_health()

    if not api_connected:
        st.error("❌ Cannot connect to backend API.")
        st.info("Please start the backend server: `python -m uvicorn backend.main:app --reload --port 8000`")
        return

    # Initialize session state
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'training_steps' not in st.session_state:
        st.session_state.training_steps = []
    if 'selected_employee' not in st.session_state:
        st.session_state.selected_employee = None
    if 'start_training' not in st.session_state:
        st.session_state.start_training = False
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    if 'prediction_stats' not in st.session_state:
        st.session_state.prediction_stats = None

    # Sidebar Navigation
    st.sidebar.markdown("### 🧭 Module Navigation")
    page_option = st.sidebar.radio(
        "Select Section",
        ["📚 Model Training", "👤 Employee View", "📊 Overall View"],
        index=0 if not st.session_state.training_complete else 1
    )

    # Reset button in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Training State", use_container_width=True):
        st.session_state.training_complete = False
        st.session_state.training_steps = []
        st.session_state.start_training = False
        st.session_state.training_in_progress = False
        st.session_state.prediction_stats = None
        st.rerun()

    # ==========================================
    # SECTION 1: MODEL TRAINING
    # ==========================================
    if page_option == "📚 Model Training":
        st.markdown('<div class="sub-header">📚 Model Training & Configuration</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            algorithm = st.selectbox(
                "Select Algorithm",
                ["Hybrid (Random Forest + Gradient Boosting)", "Random Forest", "Gradient Boosting"]
            )
            st.caption("💡 Hybrid combines both algorithms for better accuracy")

        with col2:
            if st.button("🚀 Train Model", use_container_width=True, type="primary", disabled=st.session_state.training_in_progress):
                st.session_state.training_steps = []
                st.session_state.training_complete = False
                st.session_state.start_training = True
                st.session_state.training_in_progress = True
                st.rerun()

        # Show training progress
        if st.session_state.training_steps:
            st.markdown("### 📊 Training Progress")

            all_complete = True
            for step in st.session_state.training_steps:
                icon = step.get('icon', '📌')
                title = step.get('title', '')
                detail = step.get('detail', '')
                status = step.get('status', '')

                if status == 'success':
                    st.markdown(f"""
                        <div class="training-step training-step-success">
                            <div class="step-icon">{icon}</div>
                            <div class="step-content">
                                <div class="step-title">✅ {title}</div>
                                <div class="step-detail">{detail}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                elif status == 'running':
                    all_complete = False
                    st.markdown(f"""
                        <div class="training-step training-step-running">
                            <div class="step-icon">{icon}</div>
                            <div class="step-content">
                                <div class="step-title">⏳ {title}</div>
                                <div class="step-detail">{detail}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                elif status == 'error':
                    all_complete = False
                    st.markdown(f"""
                        <div class="training-step training-step-error">
                            <div class="step-icon">{icon}</div>
                            <div class="step-content">
                                <div class="step-title">❌ {title}</div>
                                <div class="step-detail">{detail}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    all_complete = False
                    st.markdown(f"""
                        <div class="training-step">
                            <div class="step-icon">{icon}</div>
                            <div class="step-content">
                                <div class="step-title">{title}</div>
                                <div class="step-detail">{detail}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            # If training is complete, show the summary
            if st.session_state.training_complete:
                st.success("✅ Training and prediction completed successfully!")

                stats = st.session_state.prediction_stats
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

                st.info("🎯 All employees have been predicted. Go to 'Employee View' or 'Overall View' to see results.")

        # Handle training process
        if st.session_state.start_training and not st.session_state.training_complete:
            try:
                # Step 1: Fetch data
                if not st.session_state.training_steps:
                    st.session_state.training_steps.append({
                        'icon': '📊',
                        'title': 'Step 1: Fetching Historical Data',
                        'detail': 'Retrieving quarterly performance data from database...',
                        'status': 'running'
                    })
                    st.rerun()

                # Step 2: Start training
                if len(st.session_state.training_steps) == 1:
                    st.session_state.training_steps[-1] = {
                        'icon': '📊',
                        'title': 'Step 1: Data Retrieved',
                        'detail': '✅ Historical data loaded successfully',
                        'status': 'success'
                    }
                    st.session_state.training_steps.append({
                        'icon': '🧠',
                        'title': 'Step 2: Training Model',
                        'detail': f'Training {algorithm} model on historical data... This may take a few minutes.',
                        'status': 'running'
                    })
                    st.rerun()

                # Step 3: Call API
                if len(st.session_state.training_steps) == 2:
                    with st.spinner("Training in progress... This may take a few minutes."):
                        try:
                            train_response = requests.post(
                                f"{API_URL}/api/predictions/train",
                                json={"algorithm": algorithm},
                                timeout=600
                            )

                            if train_response.status_code == 200:
                                result = train_response.json()

                                if result.get('success'):
                                    # Update Step 2
                                    st.session_state.training_steps[-1] = {
                                        'icon': '🧠',
                                        'title': 'Step 2: Model Trained',
                                        'detail': f"✅ Accuracy: {result.get('accuracy', 0):.2%} | F1: {result.get('f1_score', 0):.2%}",
                                        'status': 'success'
                                    }

                                    # Add Step 3 - Predictions
                                    predictions_count = result.get('predictions_count', 0)
                                    st.session_state.training_steps.append({
                                        'icon': '🎯',
                                        'title': 'Step 3: Predictions Complete',
                                        'detail': f"✅ Predicted for {predictions_count} employees",
                                        'status': 'success'
                                    })

                                    # Mark training as complete
                                    st.session_state.training_complete = True
                                    st.session_state.start_training = False
                                    st.session_state.training_in_progress = False

                                    # Store stats in session
                                    stats = result.get('stats', {})
                                    st.session_state.prediction_stats = stats

                                    st.success(f"✅ Training completed! Predictions made for {predictions_count} employees.")
                                    st.rerun()
                                else:
                                    st.session_state.training_steps[-1] = {
                                        'icon': '❌',
                                        'title': 'Training Failed',
                                        'detail': f"Error: {result.get('message', 'Unknown error')}",
                                        'status': 'error'
                                    }
                                    st.session_state.start_training = False
                                    st.session_state.training_in_progress = False
                                    st.rerun()
                            else:
                                st.session_state.training_steps[-1] = {
                                    'icon': '❌',
                                    'title': 'Training Failed',
                                    'detail': f"Server error: {train_response.status_code}",
                                    'status': 'error'
                                }
                                st.session_state.start_training = False
                                st.session_state.training_in_progress = False
                                st.rerun()
                        except requests.Timeout:
                            st.session_state.training_steps[-1] = {
                                'icon': '⏰',
                                'title': 'Training Timeout',
                                'detail': 'Training took too long. Try again with less data.',
                                'status': 'error'
                            }
                            st.session_state.start_training = False
                            st.session_state.training_in_progress = False
                            st.rerun()

            except Exception as e:
                st.session_state.training_steps.append({
                    'icon': '❌',
                    'title': 'Training Failed',
                    'detail': f"Error: {str(e)}",
                    'status': 'error'
                })
                st.session_state.start_training = False
                st.session_state.training_in_progress = False
                st.rerun()

    # ==========================================
    # SECTION 2: EMPLOYEE VIEW
    # ==========================================
    elif page_option == "👤 Employee View":
        st.markdown('<div class="sub-header">👤 Employee Performance View</div>', unsafe_allow_html=True)

        if not st.session_state.training_complete:
            st.warning("⚠️ Please train the model first in the 'Model Training' section.")
            return

        # Get employees
        try:
            emp_response = requests.get(f"{API_URL}/api/employees/?limit=1000", timeout=10)
            if emp_response.status_code == 200:
                employees = emp_response.json().get('employees', [])

                if not employees:
                    st.warning("No employees found in the system.")
                    return

                emp_options = {}
                for emp in employees:
                    emp_id = emp.get('employee_id')
                    if emp_id:
                        emp_options[emp_id] = f"{emp_id} - {emp.get('job_role', 'N/A')} ({emp.get('department', 'N/A')})"

                # Search
                col1, col2 = st.columns([2, 1])

                with col1:
                    search_term = st.text_input("🔍 Search Employee", placeholder="Enter Employee ID...")

                    if search_term:
                        filtered = {k: v for k, v in emp_options.items() if search_term.lower() in k.lower()}
                        if filtered:
                            selected_employee = st.selectbox("Select Employee", list(filtered.keys()), format_func=lambda x: filtered[x])
                        else:
                            st.warning("No employees found")
                            selected_employee = None
                    else:
                        selected_employee = st.selectbox("Select Employee", list(emp_options.keys()), format_func=lambda x: emp_options[x])

                with col2:
                    if selected_employee:
                        if st.button("📊 View Performance", use_container_width=True):
                            st.session_state.selected_employee = selected_employee
                            st.rerun()

                # Display employee data
                if st.session_state.selected_employee:
                    employee_id = st.session_state.selected_employee

                    with st.spinner("Loading employee data..."):
                        # Get history and predictions
                        history_response = requests.get(f"{API_URL}/api/employees/history/{employee_id}", timeout=10)
                        pred_response = requests.get(f"{API_URL}/api/predictions/employee/{employee_id}", timeout=10)

                        if history_response.status_code == 200:
                            history = history_response.json()

                            if history:
                                df_history = pd.DataFrame(history)

                                # Performance trend
                                st.markdown("### 📈 Performance Trend")

                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=df_history['period'],
                                    y=df_history['score'],
                                    mode='lines+markers',
                                    name='Historical Score',
                                    line=dict(color='#2b6cb0', width=3),
                                    marker=dict(size=10)
                                ))

                                # Check if we have predictions
                                predictions = []
                                if pred_response.status_code == 200:
                                    predictions = pred_response.json()
                                    if predictions:
                                        pred = predictions[0]
                                        fig.add_trace(go.Scatter(
                                            x=[pred['period']],
                                            y=[pred['predicted_score']],
                                            mode='markers',
                                            name='Predicted Score',
                                            marker=dict(color='#48bb78', size=15, symbol='star')
                                        ))

                                fig.update_layout(
                                    height=350,
                                    margin=dict(l=20, r=20, t=20, b=20),
                                    xaxis_title="Period",
                                    yaxis_title="Score"
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Metrics summary
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    latest_score = df_history.iloc[-1]['score'] if len(df_history) > 0 else 0
                                    st.metric("Current Score", f"{latest_score:.1f}%")

                                    if predictions:
                                        pred = predictions[0]
                                        st.metric("Predicted Next Quarter", f"{pred['predicted_score']:.1f}%")
                                        st.caption(f"Confidence: {pred['confidence']:.1%}")
                                    else:
                                        st.info("No prediction available.")

                                with col2:
                                    band = 'High' if latest_score >= 70 else 'Medium' if latest_score >= 40 else 'Low'
                                    band_class = f"band-{band.lower()}"
                                    st.markdown(f"""
                                        <div style="text-align: center; padding: 1rem;">
                                            <div class="{band_class}" style="font-size: 1.5rem; padding: 0.5rem 2rem;">
                                                {band}
                                            </div>
                                            <p style="color: #718096; margin-top: 0.5rem;">Performance Level</p>
                                        </div>
                                    """, unsafe_allow_html=True)

                                with col3:
                                    if len(df_history) > 1:
                                        trend = df_history['score'].iloc[-1] - df_history['score'].iloc[-2]
                                        st.metric(
                                            "Quarterly Change",
                                            f"{trend:+.1f}%",
                                            delta_color="normal" if trend > 0 else "inverse"
                                        )
                                    st.metric("Quarters Tracked", len(df_history))

                                # Show prediction details if available
                                if predictions:
                                    pred = predictions[0]
                                    st.markdown("### 📊 Prediction Details")

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Predicted Band", pred['predicted_band'])
                                    with col2:
                                        st.metric("Predicted Score", f"{pred['predicted_score']:.1f}%")
                                    with col3:
                                        st.metric("Algorithm", pred.get('algorithm', 'Hybrid'))

                                    # Show RF and GB individual predictions
                                    if 'rf_prediction' in pred and 'gb_prediction' in pred:
                                        st.markdown("#### Ensemble Breakdown")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.info(f"**Random Forest**: {pred.get('rf_prediction', 'N/A')} (Confidence: {pred.get('rf_confidence', 0):.1%})")
                                        with col2:
                                            st.info(f"**Gradient Boosting**: {pred.get('gb_prediction', 'N/A')} (Confidence: {pred.get('gb_confidence', 0):.1%})")

                                # Feature importance
                                if predictions and predictions[0].get('feature_importance'):
                                    st.markdown("### 🔑 Key Influencing Factors")

                                    features = predictions[0]['feature_importance']
                                    if isinstance(features, dict):
                                        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:10]

                                        df_features = pd.DataFrame(top_features, columns=['Feature', 'Importance'])

                                        fig = go.Figure(data=[
                                            go.Bar(
                                                x=df_features['Importance'],
                                                y=df_features['Feature'],
                                                orientation='h',
                                                marker_color='#2b6cb0'
                                            )
                                        ])

                                        fig.update_layout(
                                            height=300,
                                            margin=dict(l=20, r=20, t=20, b=20),
                                            xaxis_title='Importance',
                                            yaxis_title=''
                                        )

                                        st.plotly_chart(fig, use_container_width=True)

                                # Historical data table
                                with st.expander("📋 View Historical Data"):
                                    display_df = df_history[['period', 'score', 'band', 'deadline_adherence', 'punctuality', 'problem_solving', 'leadership', 'collaboration', 'communication']]
                                    display_df.columns = ['Period', 'Score', 'Band', 'Deadline Adherence', 'Punctuality', 'Problem Solving', 'Leadership', 'Collaboration', 'Communication']
                                    display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.1f}%")
                                    st.dataframe(display_df, use_container_width=True)

                            else:
                                st.warning("No historical data found for this employee")
                        else:
                            st.error("Failed to fetch employee data")

                        # Add a predict button if no prediction exists
                        if pred_response.status_code != 200 or not pred_response.json():
                            if st.button("🎯 Get Prediction for This Employee", use_container_width=True):
                                with st.spinner("Making prediction..."):
                                    pred_request = requests.post(
                                        f"{API_URL}/api/predictions/predict",
                                        json={"employee_id": employee_id}
                                    )
                                    if pred_request.status_code == 200:
                                        st.success("Prediction made successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to make prediction")

        except Exception as e:
            st.error(f"Error loading employee data: {str(e)}")

    # ==========================================
    # SECTION 3: OVERALL VIEW
    # ==========================================
    else:
        st.markdown('<div class="sub-header">📊 Overall Performance View</div>', unsafe_allow_html=True)

        if not st.session_state.training_complete:
            st.warning("⚠️ Please train the model first in the 'Model Training' section.")
            return

        # Get stats
        stats = get_prediction_stats()

        if stats and stats.get('total_predictions', 0) > 0:
            # Key metrics
            st.markdown("### 📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats.get('total_predictions', 0)}</div>
                        <div class="stat-label">Total Predictions</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class="stat-card" style="border-top-color: #48bb78;">
                        <div class="stat-value">{stats.get('high_performers', 0)}</div>
                        <div class="stat-label">High Performers</div>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div class="stat-card" style="border-top-color: #ed8936;">
                        <div class="stat-value">{stats.get('medium_performers', 0)}</div>
                        <div class="stat-label">Medium Performers</div>
                    </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                    <div class="stat-card" style="border-top-color: #fc8181;">
                        <div class="stat-value">{stats.get('low_performers', 0)}</div>
                        <div class="stat-label">Low Performers</div>
                    </div>
                """, unsafe_allow_html=True)

            # Performance Distribution Chart
            st.markdown("### 📊 Performance Distribution")

            band_dist = stats.get('band_distribution', {})
            if band_dist:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(band_dist.keys()),
                        values=list(band_dist.values()),
                        marker_colors=['#48bb78', '#ed8936', '#fc8181'],
                        hole=0.4,
                        textinfo='label+percent'
                    )
                ])
                fig.update_layout(
                    height=400,
                    title="Employee Performance Band Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Score Distribution Histogram
            st.markdown("### 📊 Score Distribution")

            predictions = get_all_predictions()
            if predictions:
                df_pred = pd.DataFrame(predictions)

                fig = go.Figure(data=[
                    go.Histogram(
                        x=df_pred['predicted_score'],
                        nbinsx=20,
                        marker_color='#2b6cb0',
                        name='Predicted Scores'
                    )
                ])
                fig.update_layout(
                    height=400,
                    title="Distribution of Predicted Performance Scores",
                    xaxis_title="Score",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig, use_container_width=True)

            # All Predictions Table
            st.markdown("### 📋 All Predictions")

            if predictions:
                df_display = pd.DataFrame(predictions)
                df_display = df_display[['employee_id', 'period', 'predicted_band', 'predicted_score', 'confidence', 'algorithm']]
                df_display.columns = ['Employee ID', 'Period', 'Band', 'Score', 'Confidence', 'Algorithm']
                df_display['Score'] = df_display['Score'].apply(lambda x: f"{x:.1f}%")
                df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x:.1%}")

                st.dataframe(df_display, use_container_width=True)

                # Download button
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No predictions available yet. Please train the model and make predictions.")

if __name__ == "__main__":
    main()