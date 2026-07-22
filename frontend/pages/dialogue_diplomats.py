import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"


def main():
    st.title("🤝 Dialogue Diplomats (Step 3 — CAPAF Module)")
    st.markdown("""
    Three formal-utility agents negotiate team composition via weighted arbitration:
    - **Diplomat A (Performance)** — maximizes `ability_composite` (task + KPI + prediction)
    - **Diplomat B (Diversity)** — maximizes sub-culture spread, penalizes Proposition-2 anti-alignment
    - **Diplomat C (Conflict)** — minimizes `divergence_score` (Step 1's opinion-dynamics output)

    The arbitration engine weights the three diplomats based on **project complexity (β)**:
    higher complexity shifts weight away from raw performance and toward diversity + low-conflict composition.

    *Scope: restricted to the 432 employees with Feedback_360 coverage.*
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚙️ Form a Team",
        "📊 Diplomat Reasoning",
        "📈 Complexity Sensitivity",
        "🔍 Team Detail",
        "🤖 Formal vs LLM"
    ])

    # ---------- Tab 1: Run ----------
    with tab1:
        st.subheader("⚙️ Configure & Optimize")

        col1, col2, col3 = st.columns(3)
        with col1:
            team_size = st.number_input("Team size", min_value=2, max_value=20, value=5, step=1, key="team_size")
        with col2:
            project_complexity = st.slider(
                "Project complexity (β, raw scale)",
                min_value=1.0, max_value=10.0, value=5.0, step=0.1, key="project_complexity"
            )
        with col3:
            n_clusters = st.number_input("Sub-culture clusters", min_value=2, max_value=10, value=4, step=1, key="n_clusters_diplomat")

        inst_filter = st.text_input("Institution ID (optional)", placeholder="I01", key="diplomat_inst")

        st.caption("β is normalized internally against the observed project_complexity range (~1.0–9.99) before arbitration weighting.")

        if st.button("🚀 Form Optimal Team", type="primary", key="run_diplomat"):
            with st.spinner("Diplomats negotiating team composition..."):
                payload = {
                    "team_size": int(team_size),
                    "project_complexity": float(project_complexity),
                    "n_clusters": int(n_clusters),
                    "n_restarts": 20
                }
                if inst_filter:
                    payload["institution_id"] = inst_filter

                try:
                    response = requests.post(f"{API_URL}/api/diplomats/optimize-team", json=payload, timeout=60)

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["diplomat_result"] = result
                        st.success(f"✅ Team formed — utility = {result.get('team_utility', 0):.4f}")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Normalized β", result.get('project_complexity_normalized', 0))
                        col2.metric("Team Utility", round(result.get('team_utility', 0), 4))
                        col3.metric("Surplus Capacity", round(result.get('surplus_capacity', 0), 3))

                        weights = result.get('arbitration_weights', {})
                        st.write("**Arbitration weights:**", weights)
                    elif response.status_code == 400:
                        st.error(f"❌ {response.json().get('detail', 'Bad request')}")
                    elif response.status_code == 404:
                        st.error(f"❌ {response.json().get('detail', 'Insufficient data')}")
                    else:
                        st.error(f"❌ Failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the backend is running.")
                except requests.exceptions.Timeout:
                    st.error("❌ Timed out — try reducing n_restarts or team_size.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ---------- Tab 2: Diplomat Reasoning ----------
    with tab2:
        st.subheader("📊 Diplomat Reasoning (Explainability)")

        result = st.session_state.get("diplomat_result")
        if not result:
            st.info("Form a team in Tab 1 first.")
        else:
            reasoning = result.get("diplomat_reasoning", {})
            weights = result.get("arbitration_weights", {})

            if reasoning:
                cols = st.columns(3)
                diplomat_labels = {
                    'A_performance': ('🏆 Diplomat A — Performance', 'A'),
                    'B_diversity': ('🌈 Diplomat B — Diversity', 'B'),
                    'C_conflict': ('🕊️ Diplomat C — Conflict', 'C')
                }
                for col, (key, (label, wkey)) in zip(cols, diplomat_labels.items()):
                    with col:
                        detail = reasoning.get(key, {})
                        st.markdown(f"**{label}**")
                        st.metric("Weight", detail.get('weight', 0))
                        st.metric("Utility", round(detail.get('utility', 0), 4))
                        st.caption(detail.get('rationale', ''))

                # Weighted contribution chart
                contrib = pd.DataFrame([
                    {'diplomat': 'A (Performance)', 'weighted_contribution': reasoning['A_performance']['weight'] * reasoning['A_performance']['utility']},
                    {'diplomat': 'B (Diversity)', 'weighted_contribution': reasoning['B_diversity']['weight'] * reasoning['B_diversity']['utility']},
                    {'diplomat': 'C (Conflict)', 'weighted_contribution': reasoning['C_conflict']['weight'] * reasoning['C_conflict']['utility']}
                ])
                fig = px.bar(
                    contrib, x='diplomat', y='weighted_contribution', color='diplomat',
                    title="Each Diplomat's Weighted Contribution to Final Team Utility",
                    labels={'weighted_contribution': 'Weight × Utility'}
                )
                fig.add_hline(y=0, line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

                st.caption(f"Sum of weighted contributions ≈ team_utility = {result.get('team_utility', 0):.4f}")

            # Rejected alternative
            rejected = result.get("rejected_alternative")
            st.subheader("🥈 Rejected Alternative")
            if rejected:
                st.write(f"**Runner-up team utility:** {rejected['team_utility']:.4f} "
                         f"(gap of {rejected['utility_gap_vs_best']:.4f} vs. selected team)")
                st.dataframe(pd.DataFrame(rejected['team']), use_container_width=True)
            else:
                st.info("All optimizer restarts converged to the same team — no distinct runner-up found. "
                        "Consider increasing n_restarts if you expect more variation.")

    # ---------- Tab 3: Complexity Sensitivity ----------
    with tab3:
        st.subheader("📈 How Team Composition Shifts with Project Complexity")
        st.markdown("Runs the optimizer across a range of β values to show the arbitration weight transition and resulting team changes.")

        col1, col2 = st.columns(2)
        with col1:
            sens_team_size = st.number_input("Team size", min_value=2, max_value=20, value=5, step=1, key="sens_team_size")
        with col2:
            n_steps = st.slider("Number of β steps", min_value=3, max_value=10, value=5, key="n_steps")

        if st.button("▶️ Run Sensitivity Sweep", key="run_sensitivity"):
            beta_values = [1.0 + i * (9.99 - 1.0) / (n_steps - 1) for i in range(n_steps)]
            sweep_results = []

            progress = st.progress(0.0)
            for i, beta in enumerate(beta_values):
                try:
                    response = requests.post(
                        f"{API_URL}/api/diplomats/optimize-team",
                        json={"team_size": int(sens_team_size), "project_complexity": beta, "n_restarts": 10},
                        timeout=60
                    )
                    if response.status_code == 200:
                        r = response.json()
                        sweep_results.append({
                            'beta_raw': round(beta, 2),
                            'beta_normalized': r['project_complexity_normalized'],
                            'weight_A': r['arbitration_weights']['A'],
                            'weight_B': r['arbitration_weights']['B'],
                            'weight_C': r['arbitration_weights']['C'],
                            'team_utility': r['team_utility'],
                            'surplus_capacity': r['surplus_capacity'],
                            'mean_ability': r['diplomat_reasoning']['A_performance']['utility'],
                            'mean_divergence': -r['diplomat_reasoning']['C_conflict']['utility'],  # C is negated
                            'team_ids': ", ".join([m['employee_id'] for m in r['team']])
                        })
                except Exception as e:
                    st.warning(f"β={beta:.2f} failed: {e}")
                progress.progress((i + 1) / n_steps)

            if sweep_results:
                st.session_state["sensitivity_results"] = pd.DataFrame(sweep_results)

        sweep_df = st.session_state.get("sensitivity_results")
        if sweep_df is not None:
            st.dataframe(sweep_df, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sweep_df['beta_raw'], y=sweep_df['weight_A'], mode='lines+markers', name='Weight A (Performance)'))
            fig.add_trace(go.Scatter(x=sweep_df['beta_raw'], y=sweep_df['weight_B'], mode='lines+markers', name='Weight B (Diversity)'))
            fig.add_trace(go.Scatter(x=sweep_df['beta_raw'], y=sweep_df['weight_C'], mode='lines+markers', name='Weight C (Conflict)'))
            fig.update_layout(title="Arbitration Weights vs. Project Complexity",
                              xaxis_title="Project Complexity (raw β)", yaxis_title="Weight")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=sweep_df['beta_raw'], y=sweep_df['mean_ability'], mode='lines+markers', name='Team Mean Ability'))
            fig2.add_trace(go.Scatter(x=sweep_df['beta_raw'], y=sweep_df['mean_divergence'], mode='lines+markers', name='Team Mean Divergence'))
            fig2.update_layout(title="Resulting Team Composition vs. Project Complexity",
                               xaxis_title="Project Complexity (raw β)", yaxis_title="Score")
            st.plotly_chart(fig2, use_container_width=True)

            st.caption("If the design is working: mean ability should trend down (or stay flat) and diversity/low-divergence "
                       "should dominate as β increases, since weight shifts from A toward B+C.")

    # ---------- Tab 4: Team Detail ----------
    with tab4:
        st.subheader("🔍 Selected Team — Full Detail")

        result = st.session_state.get("diplomat_result")
        if not result:
            st.info("Form a team in Tab 1 first.")
        else:
            team = result.get("team", [])
            if team:
                team_df = pd.DataFrame(team)
                st.dataframe(team_df, use_container_width=True)

                fig = px.bar(
                    team_df, x='employee_id', y='ability_composite', color='cluster_id',
                    title="Team Members — Ability Composite by Sub-Culture Cluster",
                    labels={'ability_composite': 'Ability Composite', 'employee_id': 'Employee'}
                )
                st.plotly_chart(fig, use_container_width=True)

                growth = result.get("growth_headroom", {})
                if growth:
                    growth_df = pd.DataFrame(list(growth.items()), columns=['employee_id', 'growth_headroom'])
                    fig2 = px.bar(
                        growth_df, x='employee_id', y='growth_headroom',
                        title="Growth Headroom per Member (0 = at pool max ability)",
                        labels={'growth_headroom': 'Headroom'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption(f"Team surplus capacity: {result.get('surplus_capacity', 0):.3f} "
                               f"— average room for improvement before hitting the pool's current performance ceiling.")

                csv = team_df.to_csv(index=False)
                st.download_button("📥 Download Team", csv,
                                   f"diplomat_team_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            else:
                st.info("No team data found in the last result.")
    # ---------- Tab 5: Formal vs LLM ----------
    with tab5:
        st.subheader("🤖 Formal Optimizer vs LLM-Negotiated Diplomats")
        st.warning("⚠️ This makes real Anthropic API calls and costs money per run. Start small (1-2 LLM runs) before scaling up.")

        col1, col2, col3 = st.columns(3)
        with col1:
            llm_team_size = st.number_input("Team size", min_value=2, max_value=20, value=5, step=1, key="llm_team_size")
        with col2:
            llm_complexity = st.slider("Project complexity (β)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, key="llm_complexity")
        with col3:
            n_llm_runs = st.number_input("Number of LLM runs", min_value=1, max_value=10, value=2, step=1, key="n_llm_runs")

        col4, col5 = st.columns(2)
        with col4:
            candidate_pool_size = st.number_input("Candidate pool size (both methods search within this)", min_value=10, max_value=200, value=40, step=10, key="candidate_pool_size")
        with col5:
            model_choice = st.selectbox("LLM model", ["claude-haiku-4-5-20251001", "claude-sonnet-5"], key="llm_model")

        est_calls = n_llm_runs * 4
        st.caption(f"Estimated API calls this run: ~{est_calls} (3 personas + 1 arbitrator, × {n_llm_runs} runs)")

        if st.button("🚀 Run Comparison", type="primary", key="run_llm_compare"):
            with st.spinner(f"Running formal optimizer + {n_llm_runs} LLM negotiation(s)... this may take a minute"):
                payload = {
                    "team_size": int(llm_team_size),
                    "project_complexity": float(llm_complexity),
                    "n_llm_runs": int(n_llm_runs),
                    "candidate_pool_size": int(candidate_pool_size),
                    "model": model_choice
                }
                try:
                    response = requests.post(f"{API_URL}/api/llm-diplomats/compare", json=payload, timeout=180)

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["llm_compare_result"] = result
                        st.success("✅ Comparison complete")
                    elif response.status_code == 502:
                        st.error(f"❌ LLM call failed: {response.json().get('detail', 'Unknown error')}")
                    elif response.status_code == 404:
                        st.error(f"❌ {response.json().get('detail', 'Insufficient data')}")
                    else:
                        st.error(f"❌ Failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}.")
                except requests.exceptions.Timeout:
                    st.error("❌ Timed out — LLM calls can be slow, try fewer runs.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        result = st.session_state.get("llm_compare_result")
        if result:
            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("Sampled pool size", result.get('sampled_pool_size', 0))
            col2.metric("Formal team utility", round(result.get('formal_team_utility', 0), 4))

            st.write("**Formal optimizer team:**", result.get('formal_team', []))

            llm_runs = result.get('llm_runs', [])
            if llm_runs:
                runs_df = pd.DataFrame([{
                    'run': i + 1,
                    'llm_team': ", ".join(r['llm_team']),
                    'overlap_with_formal': r['jaccard_overlap_with_formal'],
                    'llm_utility_scored_by_formal_math': r['llm_team_scored_by_formal_utility']
                } for i, r in enumerate(llm_runs)])
                st.dataframe(runs_df, use_container_width=True)

                fig = px.bar(
                    runs_df, x='run', y=['overlap_with_formal', 'llm_utility_scored_by_formal_math'],
                    barmode='group', title="LLM Runs: Overlap with Formal Team & Utility (scored by formal math)",
                    labels={'value': 'Score (0-1)', 'run': 'LLM Run'}
                )
                fig.add_hline(y=result.get('formal_team_utility', 0), line_dash="dash", line_color="green",
                              annotation_text="Formal optimizer's own utility")
                st.plotly_chart(fig, use_container_width=True)

                stability = result.get('llm_run_to_run_stability_mean_jaccard')
                if stability is not None:
                    st.metric("Run-to-run stability (Jaccard)", stability)
                    st.caption("1.0 = LLM picks the identical team every time given identical inputs. Lower = more variation between runs.")

                st.subheader("💬 LLM Reasoning per Run")
                for i, r in enumerate(llm_runs):
                    with st.expander(f"Run {i+1} reasoning"):
                        st.write(r['reasoning'])

if __name__ == "__main__":
    main()