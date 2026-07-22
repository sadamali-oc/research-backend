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
    - **Ability composite**: task competencies + hard KPI + predicted future performance (excludes collaboration/communication)
    - **Relational score**: collaboration + communication only
    - **Aligned**: ability rank and relational rank point the same direction (both high or both low)
    - **Halo gap**: holistic performance_score minus ability_composite — positive means a cluster is rated better than pure task skill would predict
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚙️ Run Clustering",
        "📊 Cluster Summary",
        "🔄 Opinion Dynamics",
        "🔍 Employee-Level View",
        "✨ Halo Gap"
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
                payload = {"n_clusters": int(n_clusters), "include_employees": True}
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

                fig1 = px.bar(
                    summary_df, x='cluster_id', y='pct_aligned',
                    title="% of Prop-2-Aligned Employees per Cluster",
                    labels={'cluster_id': 'Cluster', 'pct_aligned': '% Aligned'},
                    text_auto='.0%'
                )
                fig1.update_yaxes(range=[0, 1], tickformat='.0%')
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = px.bar(
                    summary_df, x='cluster_id', y='n',
                    title="Cluster Sizes",
                    labels={'cluster_id': 'Cluster', 'n': 'Employee Count'}
                )
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.scatter(
                    summary_df, x='mean_ability', y='mean_relational', size='n',
                    color='cluster_id', text='cluster_id',
                    title="Cluster Centers: Mean Ability vs Mean Relational Score",
                    labels={'mean_ability': 'Mean Ability Composite', 'mean_relational': 'Mean Relational Score'}
                )
                fig3.update_traces(textposition='top center')
                st.plotly_chart(fig3, use_container_width=True)

                alignment_sig = result.get("alignment_significance", {})
                if alignment_sig:
                    st.subheader("📊 Alignment Significance")
                    st.metric("Chi-square p-value", round(alignment_sig.get('p_value', 0), 4))
                    if alignment_sig.get('significant', False):
                        st.success("✅ Alignment is significantly associated with cluster membership (p < 0.05)")
                    else:
                        st.info("ℹ️ Alignment is NOT significantly associated with cluster membership (p ≥ 0.05)")

                if summary_df['pct_aligned'].nunique() == 1:
                    st.warning("⚠️ All clusters show identical alignment % — check if clustering features are actually separating the population.")
            else:
                st.info("No summary data returned.")

    # ---------- Tab 3: Opinion Dynamics ----------
    with tab3:
        st.subheader("🔄 Opinion Dynamics Profile")
        st.markdown("Measures how disagreement between evaluators varies across sub-cultures.")

        result = st.session_state.get("culture_result")
        if not result:
            st.info("Run clustering in Tab 1 first.")
        else:
            opinion_profile = result.get("opinion_dynamics_profile", {})
            opinion_sig = result.get("opinion_dynamics_significance", {})

            if opinion_profile:
                op_df = pd.DataFrame(opinion_profile)
                op_df['cluster_id'] = op_df['cluster_id'].astype(str)

                st.subheader("📊 Divergence by Cluster")
                st.dataframe(op_df, use_container_width=True)

                fig = px.bar(
                    op_df, x='cluster_id', y='mean_divergence',
                    title="Mean Divergence by Cluster",
                    labels={'cluster_id': 'Cluster', 'mean_divergence': 'Mean Divergence Score'},
                    color='mean_divergence', color_continuous_scale='RdBu_r', text_auto='.3f'
                )
                st.plotly_chart(fig, use_container_width=True)

                if 'std_divergence' in op_df.columns:
                    fig2 = px.bar(
                        op_df, x='cluster_id', y='std_divergence',
                        title="Divergence Standard Deviation by Cluster",
                        labels={'cluster_id': 'Cluster', 'std_divergence': 'Std Dev of Divergence'},
                        color='std_divergence', color_continuous_scale='Blues', text_auto='.3f'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No opinion dynamics profile available.")

            if opinion_sig:
                st.subheader("📊 Statistical Significance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kruskal-Wallis p-value", round(opinion_sig.get('p_value', 0), 4))
                with col2:
                    if opinion_sig.get('significant', False):
                        st.success("✅ Significant (p < 0.05)")
                    else:
                        st.info("ℹ️ Not significant (p ≥ 0.05)")

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

                display_cols = ['employee_id', 'cluster_id', 'ability_rank', 'relational_rank',
                                'ability_composite', 'relational_score', 'aligned']
                display_df = filtered[[c for c in display_cols if c in filtered.columns]]
                st.dataframe(display_df, use_container_width=True)

                fig = px.scatter(
                    filtered, x='ability_rank', y='relational_rank', color=filtered['cluster_id'].astype(str),
                    hover_data=['employee_id', 'department', 'gender', 'age_group', 'ability_composite'],
                    title="Ability Rank vs Relational Rank (Proposition 2 alignment view)",
                    labels={'ability_rank': 'Ability Rank (percentile)', 'relational_rank': 'Relational Rank (percentile)', 'color': 'Cluster'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
                fig.add_annotation(x=0.75, y=0.75, text="Aligned (High/High)", showarrow=False, font_size=11)
                fig.add_annotation(x=0.25, y=0.25, text="Aligned (Low/Low)", showarrow=False, font_size=11)
                fig.add_annotation(x=0.75, y=0.25, text="Anti-aligned", showarrow=False, font_size=11)
                fig.add_annotation(x=0.25, y=0.75, text="Anti-aligned", showarrow=False, font_size=11)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📈 Demographic Composition by Cluster")
                demo_col = st.selectbox("Demographic to inspect", ['gender', 'age_group', 'department', 'ethnicity'], key="demo_select")
                if demo_col in filtered.columns:
                    comp = pd.crosstab(filtered['cluster_id'], filtered[demo_col], normalize='index')
                    st.bar_chart(comp)
                    comp_abs = pd.crosstab(filtered['cluster_id'], filtered[demo_col])
                    st.caption("Absolute counts")
                    st.dataframe(comp_abs, use_container_width=True)

                csv = filtered.to_csv(index=False)
                st.download_button("📥 Download Cluster Assignments", csv,
                                   f"cluster_assignments_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            else:
                st.info("No employee-level data returned. Make sure 'include_employees' was checked when running.")

    # ---------- Tab 5: Halo Gap ----------
    with tab5:
        st.subheader("✨ Halo Gap — Does relational skill inflate holistic ratings?")
        st.markdown("""
        `ability_composite` measures pure task skill (punctuality, problem-solving, leadership, KPI, predicted performance).
        `performance_score` is the holistic average, which *does* include collaboration/communication.

        **Halo gap = performance_score − ability_composite.** Positive means a group's holistic rating exceeds
        what its task skill alone would predict — a relational "halo" boost. Negative means the opposite:
        under-rated relative to demonstrated skill.
        """)

        result = st.session_state.get("culture_result")
        if not result:
            st.info("Run clustering in Tab 1 first.")
        else:
            halo_profile = result.get("halo_profile", [])
            halo_sig = result.get("halo_significance", {})

            if halo_profile:
                halo_df = pd.DataFrame(halo_profile)
                halo_df['cluster_id'] = halo_df['cluster_id'].astype(str)
                st.dataframe(halo_df, use_container_width=True)

                fig = px.bar(
                    halo_df, x='cluster_id', y='mean_halo_gap',
                    title="Mean Halo Gap by Cluster",
                    labels={'cluster_id': 'Cluster', 'mean_halo_gap': 'Halo Gap (performance − ability)'},
                    color='mean_halo_gap', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                    text_auto='.3f'
                )
                fig.add_hline(y=0, line_color="gray", line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.bar(
                    halo_df.melt(id_vars='cluster_id', value_vars=['mean_ability_composite', 'mean_performance_score_norm'],
                                 var_name='metric', value_name='score'),
                    x='cluster_id', y='score', color='metric', barmode='group',
                    title="Ability Composite vs Holistic Performance Score, by Cluster",
                    labels={'score': 'Normalized score (0-1)', 'cluster_id': 'Cluster'}
                )
                st.plotly_chart(fig2, use_container_width=True)

                if halo_sig:
                    st.subheader("📊 Statistical Significance")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Kruskal-Wallis p-value", round(halo_sig.get('p_value', 0), 4))
                    with col2:
                        if halo_sig.get('significant', False):
                            st.success("✅ Halo effect significantly varies by cluster (p < 0.05)")
                        else:
                            st.info("ℹ️ Halo effect does NOT significantly vary by cluster (p ≥ 0.05)")

                employees = result.get("employees", [])
                if employees:
                    emp_df = pd.DataFrame(employees)
                    if 'halo_gap' in emp_df.columns:
                        st.subheader("🔍 Individual Halo Gap Distribution")
                        fig3 = px.box(
                            emp_df, x='cluster_id', y='halo_gap', color=emp_df['cluster_id'].astype(str),
                            title="Halo Gap Distribution per Employee, by Cluster",
                            labels={'halo_gap': 'Halo Gap', 'cluster_id': 'Cluster'}
                        )
                        fig3.add_hline(y=0, line_color="gray", line_dash="dash")
                        st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No halo profile returned — re-run clustering in Tab 1 to refresh with the latest backend.")


if __name__ == "__main__":
    main()