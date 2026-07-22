# frontend/pages/culture_clustering.py
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"

def main():
    st.title("🧩 Sub-Culture Identification (Step 2 — CAPAF Module)")
    st.markdown("""
    K-Means clustering on demographic features, cross-tabulated against Proposition 2 alignment.
    
    **Key Concepts:**
    - **Ability Rank**: Performance score percentile (higher = better performer)
    - **Relational Rank**: Collaboration + Communication score percentile (higher = more relational)
    - **Aligned**: Ability rank and Relational rank are in the same direction (both high or both low)
    - **Anti-Aligned**: Ability rank and Relational rank are in opposite directions
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Run Clustering",
        "📊 Cluster Summary",
        "🔄 Opinion Dynamics",
        "🔍 Employee-Level View"
    ])

    # ---------- Tab 1: Run ----------
    with tab1:
        st.subheader("⚙️ Run K-Means Clustering")

        col1, col2 = st.columns(2)
        with col1:
            inst_filter = st.text_input("Institution ID (optional — leave blank for all)", placeholder="I01", key="cluster_inst")
        with col2:
            n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=4, step=1, key="n_clusters")

        st.info("Clustering runs on: years_of_experience (numeric) + gender, age_group, department, ethnicity, language_proficiency (categorical, one-hot encoded).")

        if st.button("🚀 Run Clustering", type="primary", key="run_cluster"):
            with st.spinner("Fitting K-Means and computing alignment..."):
                payload = {
                    "n_clusters": int(n_clusters),
                    "include_employees": True
                }
                if inst_filter:
                    payload["institution_id"] = inst_filter

                try:
                    response = requests.post(f"{API_URL}/api/culture/cluster", json=payload, timeout=60)

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["culture_result"] = result
                        st.success(f"✅ Clustered {result.get('n_employees', 0)} employees into {result.get('n_clusters', 0)} clusters")

                        col1, col2 = st.columns(2)
                        col1.metric("👥 Employees Clustered", result.get('n_employees', 0))
                        col2.metric("🧩 Clusters", result.get('n_clusters', 0))

                        # Show opinion dynamics significance if available
                        sig = result.get("opinion_dynamics_significance", {})
                        if sig:
                            st.info(f"📊 Kruskal-Wallis p-value: {sig.get('p_value', 0):.4f}")
                            if sig.get('significant', False):
                                st.success("✅ Divergence significantly varies by cluster (p < 0.05)")
                            else:
                                st.info("ℹ️ Divergence does not significantly vary by cluster (p ≥ 0.05)")
                    elif response.status_code == 404:
                        st.error(f"❌ {response.json().get('detail', 'Insufficient data')}")
                    else:
                        st.error(f"❌ Failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the backend is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ---------- Tab 2: Cluster Summary ----------
    with tab2:
        st.subheader("📊 Cluster Alignment Summary")

        result = st.session_state.get("culture_result")
        if not result:
            st.info("Run clustering in Tab 1 first.")
        else:
            summary = result.get("alignment_summary", [])
            if summary:
                summary_df = pd.DataFrame(summary)
                st.dataframe(summary_df, use_container_width=True)

                # Bar chart: % aligned by cluster
                fig1 = px.bar(
                    summary_df, x='cluster_id', y='pct_aligned',
                    title="% of Prop-2-Aligned Employees per Cluster",
                    labels={'cluster_id': 'Cluster', 'pct_aligned': '% Aligned'},
                    text_auto='.0%'
                )
                fig1.update_yaxes(range=[0, 1], tickformat='.0%')
                st.plotly_chart(fig1, use_container_width=True)

                # Bar chart: cluster size
                fig2 = px.bar(
                    summary_df, x='cluster_id', y='n',
                    title="Cluster Sizes",
                    labels={'cluster_id': 'Cluster', 'n': 'Employee Count'}
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Performance vs relational mean by cluster (updated column names)
                fig3 = px.scatter(
                    summary_df, x='mean_ability', y='mean_relational', size='n',
                    color='cluster_id', text='cluster_id',
                    title="Cluster Centers: Mean Ability vs Mean Relational Score",
                    labels={'mean_ability': 'Mean Ability Composite', 'mean_relational': 'Mean Relational Score'}
                )
                fig3.update_traces(textposition='top center')
                st.plotly_chart(fig3, use_container_width=True)

                # Alignment significance
                alignment_sig = result.get("alignment_significance", {})
                if alignment_sig:
                    st.subheader("📊 Alignment Significance")
                    st.metric("Chi-square p-value", round(alignment_sig.get('p_value', 0), 4))
                    if alignment_sig.get('significant', False):
                        st.success("✅ Alignment is significantly associated with cluster membership (p < 0.05)")
                    else:
                        st.info("ℹ️ Alignment is NOT significantly associated with cluster membership (p ≥ 0.05)")

                # Sanity check flag
                if summary_df['pct_aligned'].nunique() == 1:
                    st.warning("⚠️ All clusters show identical alignment % — check if clustering features are actually separating the population, or if alignment computation is off.")
            else:
                st.info("No summary data returned.")

    # ---------- Tab 3: Opinion Dynamics ----------
    with tab3:
        st.subheader("🔄 Opinion Dynamics Profile")
        st.markdown("""
        **Opinion Dynamics** measures how divergence (disagreement between evaluators) varies across sub-cultures.
        
        - **Higher divergence** = more disagreement → potential power distance influence
        - **Lower divergence** = more consensus → lower power distance
        """)

        result = st.session_state.get("culture_result")
        if not result:
            st.info("Run clustering in Tab 1 first.")
        else:
            # Opinion dynamics profile
            opinion_profile = result.get("opinion_dynamics_profile", {})
            opinion_sig = result.get("opinion_dynamics_significance", {})

            if opinion_profile:
                op_df = pd.DataFrame(opinion_profile)
                op_df['cluster_id'] = op_df['cluster_id'].astype(str)

                st.subheader("📊 Divergence by Cluster")
                st.dataframe(op_df, use_container_width=True)

                # Bar chart: divergence by cluster
                fig = px.bar(
                    op_df, x='cluster_id', y='mean_divergence',
                    title="Mean Divergence by Cluster",
                    labels={'cluster_id': 'Cluster', 'mean_divergence': 'Mean Divergence Score'},
                    color='mean_divergence',
                    color_continuous_scale='RdBu_r',
                    text_auto='.3f'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Bar chart: divergence spread
                fig2 = px.bar(
                    op_df, x='cluster_id', y='std_divergence',
                    title="Divergence Standard Deviation by Cluster",
                    labels={'cluster_id': 'Cluster', 'std_divergence': 'Std Dev of Divergence'},
                    color='std_divergence',
                    color_continuous_scale='Blues',
                    text_auto='.3f'
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Scatter: cluster size vs divergence
                fig3 = px.scatter(
                    op_df, x='n', y='mean_divergence',
                    text='cluster_id', size='n',
                    title="Cluster Size vs Mean Divergence",
                    labels={'n': 'Cluster Size', 'mean_divergence': 'Mean Divergence'},
                    color='mean_divergence',
                    color_continuous_scale='RdBu_r'
                )
                fig3.update_traces(textposition='top center')
                st.plotly_chart(fig3, use_container_width=True)

            else:
                st.info("No opinion dynamics profile available. Make sure 'include_opinion_dynamics' was included in the clustering request.")

            # Significance
            if opinion_sig:
                st.subheader("📊 Statistical Significance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kruskal-Wallis p-value", round(opinion_sig.get('p_value', 0), 4))
                with col2:
                    if opinion_sig.get('significant', False):
                        st.success("✅ Divergence significantly varies by cluster (p < 0.05)")
                        st.write("**Interpretation:** Different sub-cultures have significantly different levels of evaluator disagreement.")
                    else:
                        st.info("ℹ️ Divergence does NOT significantly vary by cluster (p ≥ 0.05)")
                        st.write("**Interpretation:** Sub-cultures have similar levels of evaluator disagreement.")

    # ---------- Tab 4: Employee-Level View ----------
    with tab4:
        st.subheader("🔍 Employee-Level Cluster Assignment")

        result = st.session_state.get("culture_result")
        if not result:
            st.info("Run clustering in Tab 1 first.")
        else:
            employees = result.get("employees", [])
            if employees:
                df = pd.DataFrame(employees)
                st.write(f"**{len(df)} employees**")

                cluster_filter = st.multiselect(
                    "Filter by cluster", sorted(df['cluster_id'].unique().tolist()),
                    default=sorted(df['cluster_id'].unique().tolist()), key="emp_cluster_filter"
                )
                filtered = df[df['cluster_id'].isin(cluster_filter)]

                # Display employee table with key fields
                display_cols = ['employee_id', 'cluster_id', 'ability_rank', 'relational_rank',
                                'ability_composite', 'relational_score', 'aligned']
                display_df = filtered[[c for c in display_cols if c in filtered.columns]]
                st.dataframe(display_df, use_container_width=True)

                # Scatter: ability vs relational rank, colored by cluster
                fig = px.scatter(
                    filtered, x='ability_rank', y='relational_rank', color=filtered['cluster_id'].astype(str),
                    hover_data=['employee_id', 'department', 'gender', 'age_group', 'ability_composite'],  # was performance_score
                    title="Ability Rank vs Relational Rank (Proposition 2 alignment view)",
                    labels={'ability_rank': 'Ability Rank (percentile)', 'relational_rank': 'Relational Rank (percentile)', 'color': 'Cluster'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Relational Mean")
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Ability Mean")

                # Annotate quadrants
                fig.add_annotation(x=0.75, y=0.75, text="🟢 Aligned<br>(High/High)", showarrow=False, font_size=12)
                fig.add_annotation(x=0.25, y=0.25, text="🟢 Aligned<br>(Low/Low)", showarrow=False, font_size=12)
                fig.add_annotation(x=0.75, y=0.25, text="🔴 Anti-Aligned<br>(High/Low)", showarrow=False, font_size=12)
                fig.add_annotation(x=0.25, y=0.75, text="🔴 Anti-Aligned<br>(Low/High)", showarrow=False, font_size=12)

                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points in the top-right or bottom-left quadrants are Proposition-2 aligned. Top-left / bottom-right are anti-aligned.")

                # Demographic distribution per cluster
                st.subheader("📈 Demographic Composition by Cluster")
                demo_col = st.selectbox("Demographic to inspect", ['gender', 'age_group', 'department', 'ethnicity'], key="demo_select")
                if demo_col in filtered.columns:
                    comp = pd.crosstab(filtered['cluster_id'], filtered[demo_col], normalize='index')
                    st.bar_chart(comp)

                    # Also show absolute counts
                    comp_abs = pd.crosstab(filtered['cluster_id'], filtered[demo_col])
                    st.caption("Absolute counts")
                    st.dataframe(comp_abs, use_container_width=True)

                # Download
                csv = filtered.to_csv(index=False)
                st.download_button(
                    "📥 Download Cluster Assignments",
                    csv,
                    f"cluster_assignments_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            else:
                st.info("No employee-level data returned. Make sure 'include_employees' was checked when running.")

if __name__ == "__main__":
    main()