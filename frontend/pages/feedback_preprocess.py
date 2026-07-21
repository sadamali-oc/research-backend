# frontend/pages/feedback_preprocess.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"
    st.warning("⚠️ Using default API_URL (http://localhost:8000). To configure, create .streamlit/secrets.toml")

def main():
    st.title("📊 360-Degree Feedback Preprocessing")
    st.markdown("Aggregate and analyze 360-degree feedback data for the Dialogue Diplomat system.")

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "⚙️ Preprocess",
        "📈 Aggregated Data",
        "🔍 Divergence Analysis"
    ])

    # ---------- Tab 1: Data Overview (Data already in DB) ----------
    with tab1:
        st.subheader("📊 Database Overview")
        st.markdown("View statistics about the feedback data already loaded in the database.")

        if st.button("🔄 Refresh Statistics", type="primary", key="refresh_stats"):
            with st.spinner("Loading statistics..."):
                try:
                    response = requests.get(f"{API_URL}/api/feedback/stats", timeout=10)

                    if response.status_code == 200:
                        stats = response.json()

                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📝 Total Evaluations", stats.get('total_evaluations', 0))
                        col2.metric("👥 Total Employees", stats.get('total_employees', 0))
                        col3.metric("🏢 Total Institutions", stats.get('total_institutions', 0))
                        col4.metric("📅 Periods Available", len(stats.get('periods_available', [])))

                        # Periods
                        st.subheader("📅 Available Periods")
                        periods_df = pd.DataFrame(stats.get('periods_available', []))
                        if not periods_df.empty:
                            st.dataframe(periods_df)
                        else:
                            st.info("No periods found in the database.")

                        # Evaluation Type Distribution
                        st.subheader("📊 Evaluation Type Distribution")
                        type_dist = stats.get('evaluation_type_distribution', {})
                        if type_dist:
                            type_df = pd.DataFrame({
                                'Evaluation Type': list(type_dist.keys()),
                                'Count': list(type_dist.values())
                            })
                            st.bar_chart(type_df.set_index('Evaluation Type'))
                        else:
                            st.info("No evaluation type data found.")

                        # Competency Stats
                        st.subheader("📈 Competency Statistics")
                        comp_stats = stats.get('competency_stats', {})
                        if comp_stats:
                            comp_df = pd.DataFrame(comp_stats).T
                            st.dataframe(comp_df)
                        else:
                            st.info("No competency data found.")

                        # Sample data preview
                        st.subheader("🔍 Sample Feedback Data")
                        sample_response = requests.get(
                            f"{API_URL}/api/feedback/raw",
                            params={"limit": 5},
                            timeout=10
                        )
                        if sample_response.status_code == 200:
                            sample_data = sample_response.json()
                            if sample_data:
                                sample_df = pd.DataFrame(sample_data)
                                display_cols = ['evaluation_id', 'evaluatee_id', 'evaluation_type',
                                                'period_year', 'period_quarter', 'overall_rating']
                                sample_df = sample_df[[c for c in display_cols if c in sample_df.columns]]
                                st.dataframe(sample_df)
                            else:
                                st.info("No sample data found.")
                        else:
                            st.warning("Could not fetch sample data.")
                    else:
                        st.error(f"❌ Failed to fetch statistics: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the backend is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ---------- Tab 2: Preprocess ----------
    with tab2:
        st.subheader("⚙️ Aggregate & Preprocess Data")
        st.markdown("""
        Aggregate 360-degree feedback into per-employee, per-quarter metrics.
        
        **What this does:**
        - Calculates average scores for Self, Manager, Peer, and Subordinate evaluations
        - Computes `performance_score` (overall performance)
        - Computes `relational_score` (collaboration + communication)
        - Computes `divergence_score` (standard deviation across evaluation types)
        - Calculates PDI proxy (Power Distance Index proxy for each institution)
        """)

        # Show current data status
        try:
            stats_response = requests.get(f"{API_URL}/api/feedback/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                st.info(f"📊 Database contains {stats.get('total_evaluations', 0)} evaluations for {stats.get('total_employees', 0)} employees")
            else:
                st.warning("⚠️ Could not fetch current database stats.")
        except:
            st.warning("⚠️ Could not connect to API. Make sure backend is running.")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            inst_filter = st.text_input("Institution ID (optional)", placeholder="I01", key="preprocess_inst")
        with col2:
            year_filter = st.number_input("Year (optional)", min_value=2020, max_value=2030, step=1, value=2026, key="preprocess_year")
        with col3:
            quarter_filter = st.selectbox("Quarter (optional)", [None, 1, 2, 3, 4], index=0, key="preprocess_quarter")

        if st.button("⚙️ Run Preprocessing", type="primary", key="run_preprocess"):
            with st.spinner("Preprocessing data... This may take a few moments..."):
                payload = {}
                if inst_filter:
                    payload["institution_id"] = inst_filter
                if year_filter:
                    payload["period_year"] = year_filter
                if quarter_filter:
                    payload["period_quarter"] = quarter_filter

                try:
                    response = requests.post(
                        f"{API_URL}/api/feedback/preprocess",
                        json=payload,
                        timeout=300  # 5 minute timeout
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Preprocessing complete!")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📝 Records Processed", result.get('records_processed', 0))
                        col2.metric("👥 Employees", result.get('employees_processed', 0))
                        col3.metric("📅 Quarters", result.get('quarters_processed', 0))
                        col4.metric("🏢 Institutions", result.get('institutions_processed', 0))

                        st.json(result)
                    else:
                        st.error(f"❌ Preprocessing failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the backend is running.")
                except requests.exceptions.Timeout:
                    st.error("❌ Preprocessing timed out. The dataset might be too large.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ---------- Tab 3: Aggregated Data ----------
    with tab3:
        st.subheader("📈 Aggregated Performance Data")
        st.markdown("View aggregated performance metrics per employee per quarter.")

        # Check if aggregated data exists
        try:
            check_response = requests.get(f"{API_URL}/api/feedback/aggregated", params={"limit": 1}, timeout=5)
            has_data = check_response.status_code == 200 and len(check_response.json()) > 0
            if not has_data:
                st.warning("⚠️ No aggregated data found. Please run preprocessing first (Tab 2).")
        except:
            st.warning("⚠️ Could not connect to API.")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            emp_filter = st.text_input("Employee ID (optional)", placeholder="E001", key="agg_emp")
        with col2:
            inst_filter2 = st.text_input("Institution ID (optional)", placeholder="I01", key="agg_inst")
        with col3:
            quarter_filter2 = st.selectbox("Quarter (optional)", [None, 1, 2, 3, 4], index=0, key="agg_quarter")

        if st.button("🔍 Fetch Aggregated Data", type="primary", key="fetch_agg"):
            with st.spinner("Loading..."):
                params = {}
                if emp_filter:
                    params["employee_id"] = emp_filter
                if inst_filter2:
                    params["institution_id"] = inst_filter2
                if quarter_filter2:
                    params["period_quarter"] = quarter_filter2

                try:
                    response = requests.get(
                        f"{API_URL}/api/feedback/aggregated",
                        params=params,
                        timeout=30
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            df = pd.DataFrame(data)
                            st.write(f"**{len(df)} records found**")

                            # Select columns for display
                            display_cols = ['employee_id', 'period_year', 'period_quarter',
                                            'performance_score', 'relational_score', 'divergence_score',
                                            'self_count', 'manager_count', 'peer_count', 'sub_count']
                            display_df = df[[c for c in display_cols if c in df.columns]]
                            st.dataframe(display_df)

                            # Summary stats
                            st.subheader("📊 Summary Statistics")
                            if 'performance_score' in df.columns:
                                perf_stats = df['performance_score'].describe()
                                st.dataframe(pd.DataFrame(perf_stats).T)

                            # Download
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download as CSV",
                                csv,
                                f"aggregated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                        else:
                            st.info("No data found. Please run preprocessing first (Tab 2).")
                    else:
                        st.error(f"❌ Failed to fetch data: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the backend is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ---------- Tab 4: Divergence Analysis ----------
    with tab4:
        st.subheader("🔍 Divergence Analysis (Power Distance Proxy)")
        st.markdown("""
        Analyze divergence between evaluation types to detect power distance influence.
        
        **Interpretation:**
        - **High Divergence (> 0.5):** Manager ratings significantly differ from others → High Power Distance Influence
        - **Medium Divergence (0.25 - 0.5):** Moderate Power Distance Influence
        - **Low Divergence (< 0.25):** Consensus across evaluators → Low Power Distance Influence
        """)

        col1, col2 = st.columns(2)
        with col1:
            emp_filter3 = st.text_input("Employee ID (optional)", placeholder="E001", key="div_emp")
        with col2:
            inst_filter3 = st.text_input("Institution ID (optional)", placeholder="I01", key="div_inst")

        if st.button("🔍 Analyze Divergence", type="primary", key="analyze_div"):
            with st.spinner("Analyzing..."):
                params = {}
                if emp_filter3:
                    params["employee_id"] = emp_filter3
                if inst_filter3:
                    params["institution_id"] = inst_filter3

                try:
                    response = requests.get(
                        f"{API_URL}/api/feedback/divergence",
                        params=params,
                        timeout=30
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            df = pd.DataFrame(data)
                            st.write(f"**{len(df)} records found**")

                            # Display
                            st.dataframe(df)

                            # Interpretation summary
                            st.subheader("📊 Interpretation Summary")
                            interpretation_counts = df['interpretation'].value_counts()
                            st.bar_chart(interpretation_counts)

                            # Divergence distribution
                            st.subheader("📈 Divergence Score Distribution")
                            st.bar_chart(df['divergence_score'].value_counts().sort_index())

                            # High divergence employees
                            high_div = df[df['divergence_score'] > 0.5]
                            if not high_div.empty:
                                st.warning(f"⚠️ {len(high_div)} employees show High Power Distance Influence")
                                st.dataframe(high_div[['employee_id', 'period_year', 'period_quarter',
                                                       'divergence_score', 'interpretation']])
                            else:
                                st.success("✅ No employees with high power distance influence detected.")

                            # Low divergence employees (consensus)
                            low_div = df[df['divergence_score'] < 0.25]
                            if not low_div.empty:
                                st.info(f"✅ {len(low_div)} employees show consensus (Low Power Distance Influence)")
                                st.dataframe(low_div[['employee_id', 'period_year', 'period_quarter',
                                                      'divergence_score', 'interpretation']])

                            # Export
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Divergence Analysis",
                                csv,
                                f"divergence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                        else:
                            st.info("No data found. Please run preprocessing first (Tab 2).")
                    else:
                        st.error(f"❌ Failed to fetch data: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the backend is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()