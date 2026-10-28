"""
NBA Accreditation Improvement Planning: Multi-Objective Decision Support System.
Streamlit Frontend Application for Academic & Institutional Optimization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.algorithms.moead import MOEADOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer, OptimizationResult
from src.algorithms.spea2 import SPEA2Optimizer
from src.baseline.greedy import GreedyBaseline
from src.baseline.milp_solver import HiGHSMILPSolver
from src.baseline.random_search import RandomSearchBaseline
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.metrics import (
    compute_coverage,
    compute_hypervolume,
    compute_igd,
    compute_spacing,
    compute_spread,
)
from src.problem import Action, Criterion, EvaluationResult, NBAProblem

# Configure Streamlit page
st.set_page_config(
    page_title="NBA Accreditation DSS | Multi-Objective Optimization",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }
    .metric-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 0.85rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #6B7280;
        font-weight: 500;
    }
    .metric-delta {
        font-size: 0.78rem;
        font-weight: 600;
    }
    .delta-good { color: #10B981; }
    .delta-bad { color: #EF4444; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_problem(config_file: str) -> NBAProblem:
    """Load and cache the problem instance."""
    return NBAProblem(config_file)


# ==============================================================================
# SIDEBAR CONFIGURATION & PROFILE SELECTION
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/diploma.png", width=64)
st.sidebar.title("NBA Decision System")
st.sidebar.caption("UG Engineering Tier-I SAR Optimization")

# Config selection
available_configs = [
    "experiments/config.yaml",
    "experiments/config_small.yaml",
    "experiments/config_small_f4.yaml",
]
selected_config = st.sidebar.selectbox("Configuration Profile", available_configs, index=0)

try:
    problem = load_problem(selected_config)
except Exception as e:
    st.error(f"Error loading configuration '{selected_config}': {e}")
    st.stop()

# Detect configuration changes and safely reinitialize state
if st.session_state.get("active_config") != selected_config:
    st.session_state["active_config"] = selected_config
    st.session_state["whatif_vector"] = np.zeros(problem.num_actions, dtype=int)
    st.session_state["whatif_nonce"] = st.session_state.get("whatif_nonce", 0) + 1
    st.session_state["selected_plan_override"] = None

if "results_cache" not in st.session_state:
    st.session_state["results_cache"] = {}

if selected_config not in st.session_state["results_cache"]:
    st.session_state["results_cache"][selected_config] = {}

if "whatif_nonce" not in st.session_state:
    st.session_state["whatif_nonce"] = 0

if "whatif_vector" not in st.session_state or len(st.session_state["whatif_vector"]) != problem.num_actions:
    st.session_state["whatif_vector"] = np.zeros(problem.num_actions, dtype=int)

st.sidebar.markdown("---")
st.sidebar.subheader("Institutional Constraints")
st.sidebar.markdown(f"• **Budget Limit (B):** ₹`{problem.budget:.1f}` Lakh")
st.sidebar.markdown(f"• **Faculty Capacity (K_m):** `{problem.monthly_capacity:.0f}` hrs/mo")
st.sidebar.markdown(f"• **Planning Horizon (T):** `{problem.planning_horizon}` months")
st.sidebar.markdown(f"• **Tier-I Target Score (R):** `{problem.min_accreditation_score:.0f}` / 1000")

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Summary")
st.sidebar.markdown(f"• **Criteria:** {problem.num_criteria}")
evidence_count = sum(1 for a in problem.actions if a.is_evidence)
st.sidebar.markdown(f"• **Actions:** {problem.num_actions} ({evidence_count} Evidence, {problem.num_actions - evidence_count} Interventions)")
baseline_score = float(np.sum(problem.s0))
st.sidebar.markdown(f"• **Baseline Score:** `{baseline_score:.0f}` / 1000")
st.sidebar.markdown(f"• **Score Deficit:** `+{max(0.0, problem.min_accreditation_score - baseline_score):.0f}` Marks")

st.sidebar.markdown("""
<div style="font-size: 0.75rem; color: #6B7280; margin-top: 2rem; border-top: 1px solid #E5E7EB; padding-top: 0.5rem;">
    <strong>Disclaimer:</strong> Academic benchmark framework for engineering accreditation operations research.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('<div class="main-title">NBA Accreditation Strategic Improvement Planning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A Constrained Discrete Multi-Objective Optimization Framework '
    'for Educational Resource Allocation (Tier-I / Washington Accord)</div>',
    unsafe_allow_html=True,
)

# Badges
st.markdown("""
<div>
    <span class="badge-pill badge-blue">Official NBA SAR Format</span>
    <span class="badge-pill badge-green">Washington Accord Tier-I</span>
    <span class="badge-pill badge-amber">4 Conflicting Objectives</span>
    <span class="badge-pill badge-red">6 Hard Institutional Constraints</span>
</div>
<br>
""", unsafe_allow_html=True)


# ==============================================================================
# MAIN NAVIGATION TABS
# ==============================================================================
tab_overview, tab_catalog, tab_opt, tab_pareto, tab_inspector, tab_whatif = st.tabs([
    "📊 Executive Overview",
    "📋 Action Catalog & Schedule",
    "⚡ Optimization Studio",
    "📈 Pareto Frontiers & Trade-offs",
    "🎯 Implementation Roadmap",
    "🛠️ What-If Scenario Simulator",
])


# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("1. Institutional Accreditation Status & Gap Analysis")
    st.markdown("""
    Under the National Board of Accreditation (NBA) Tier-I criteria, an undergraduate engineering department 
    is evaluated across **10 criteria** totaling **1,000 marks**. To achieve full accreditation 
    (5-6 years), institutions must secure at least **750 marks ($R = 750$)** while adhering to minimum quality floors ($L_c$) 
    across all criteria and maintaining strict faculty workload and financial caps.
    """)

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    deficit = max(0.0, problem.min_accreditation_score - baseline_score)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BASELINE NBA SCORE</div>
            <div class="metric-value">{baseline_score:.0f} <span style="font-size: 0.9rem; color: #6B7280;">/ 1000</span></div>
            <div class="metric-delta delta-bad">Current: {(baseline_score/10):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ACCREDITATION TARGET (R)</div>
            <div class="metric-value">{problem.min_accreditation_score:.0f} <span style="font-size: 0.9rem; color: #6B7280;">/ 1000</span></div>
            <div class="metric-delta delta-good">Target: 75.0%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SCORE DEFICIT GAP</div>
            <div class="metric-value">+{deficit:.0f} <span style="font-size: 0.9rem; color: #6B7280;">Marks</span></div>
            <div class="metric-delta delta-bad">Required Improvement</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BUDGET CEILING (B)</div>
            <div class="metric-value">₹{problem.budget:.1f} <span style="font-size: 0.9rem; color: #6B7280;">Lakh</span></div>
            <div class="metric-delta delta-good">Max Capital & OpEx</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        viol_floors = sum(1 for c in problem.criteria if c.baseline_score < c.min_threshold)
        floor_color = "delta-bad" if viol_floors > 0 else "delta-good"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">QUALITY FLOOR BREACHES</div>
            <div class="metric-value">{viol_floors} <span style="font-size: 0.9rem; color: #6B7280;">Criteria</span></div>
            <div class="metric-delta {floor_color}">{'Action Required' if viol_floors > 0 else 'All Floors Met'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Criteria Progress Table & Chart
    crit_rows = []
    for c in problem.criteria:
        gap_to_floor = max(0.0, c.min_threshold - c.baseline_score)
        gap_to_max = c.max_score - c.baseline_score
        status = "⚠️ Floor Breached" if c.baseline_score < c.min_threshold else "✅ Floor Satisfied"
        crit_rows.append({
            "Criterion": f"C{c.id}: {c.code}",
            "Focus Area": c.name,
            "Max Marks (M_c)": c.max_score,
            "Baseline Score (s0_c)": c.baseline_score,
            "Min Floor (L_c)": c.min_threshold,
            "Floor %": f"{(c.min_threshold / c.max_score * 100):.1f}%",
            "Current %": f"{(c.baseline_score / c.max_score * 100):.1f}%",
            "Gap to Max": gap_to_max,
            "Status": status,
        })
    df_crit = pd.DataFrame(crit_rows)

    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        st.markdown("##### 10 Criteria Performance & Compliance Matrix")
        try:
            st.dataframe(
                df_crit.style.apply(
                    lambda row: ["background-color: #FEE2E2" if "Breached" in str(row["Status"]) else "" for _ in row],
                    axis=1
                ),
                use_container_width=True,
                height=380,
            )
        except Exception:
            st.dataframe(df_crit, use_container_width=True, height=380)

    with c_right:
        st.markdown("##### Baseline vs. Minimum Floor vs. Maximum Marks")
        fig_crit = go.Figure()
        fig_crit.add_trace(go.Bar(
            x=[f"C{c.id}" for c in problem.criteria],
            y=problem.s0,
            name="Baseline Score",
            marker_color="#3B82F6",
        ))
        fig_crit.add_trace(go.Scatter(
            x=[f"C{c.id}" for c in problem.criteria],
            y=problem.L,
            name="Minimum Floor (L_c)",
            mode="lines+markers",
            line=dict(color="#EF4444", width=2, dash="dash"),
            marker=dict(size=8, color="#EF4444"),
        ))
        fig_crit.add_trace(go.Scatter(
            x=[f"C{c.id}" for c in problem.criteria],
            y=problem.M,
            name="Criterion Maximum (M_c)",
            mode="lines+markers",
            line=dict(color="#10B981", width=2, dash="dot"),
            marker=dict(size=8, color="#10B981"),
        ))
        fig_crit.update_layout(
            barmode="group",
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="NBA Criterion",
            yaxis_title="Marks",
        )
        st.plotly_chart(fig_crit, use_container_width=True)

    # Sub-criteria details explorer (safely handles configs with or without subcriteria_file)
    with st.expander("🔍 Explore NBA SAR Sub-Criteria Traceability"):
        sub_rel_file = problem.config.get("data", {}).get("subcriteria_file", "data/nba_subcriteria.csv")
        sub_path = problem.root_dir / sub_rel_file
        if sub_path.exists():
            df_sub = pd.read_csv(sub_path)
            st.dataframe(df_sub, use_container_width=True)
        else:
            st.info(f"Sub-criteria traceability file '{sub_rel_file}' not found.")


# ==============================================================================
# TAB 2: ACTION CATALOG & SCHEDULE DAG
# ==============================================================================
with tab_catalog:
    st.subheader("2. Action Catalog & Critical Path Schedule")
    st.markdown(r"""
    The active profile specifies **discrete improvement actions** across the 10 criteria.
    - **Evidence Packages:** Core prerequisites ($x_E \in \{0, 1\}$) required before subsequent actions can begin.
    - **Technical Actions:** Targeted academic and infrastructure interventions ($x_a \in \{0, \dots, u_a\}$).
    - Action durations $d_a$ and prerequisite dependencies define a DAG schedule with Earliest Start ($S_a$) and Earliest Finish ($F_a$).
    """)

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        sel_crit = st.multiselect("Filter by Criterion", [f"C{i}" for i in range(1, 11)], default=[])
    with f_col2:
        sel_type = st.radio("Action Type", ["All", "Evidence Packages Only", "Interventions Only"], horizontal=True)
    with f_col3:
        search_kw = st.text_input("Search Action Name or ID", "")

    # Build action table
    action_table_data = []
    for a in problem.actions:
        crit_label = f"C{a.crit_id}"
        if sel_crit and crit_label not in sel_crit:
            continue
        if sel_type == "Evidence Packages Only" and not a.is_evidence:
            continue
        if sel_type == "Interventions Only" and a.is_evidence:
            continue
        if search_kw and (search_kw.lower() not in a.name.lower() and search_kw.lower() not in a.action_id.lower()):
            continue

        prereq_str = ", ".join(a.prerequisites) if a.prerequisites else "None (Independent)"
        action_table_data.append({
            "Action ID": a.action_id,
            "Criterion": crit_label,
            "Action Name": a.name,
            "Type": "Evidence Package" if a.is_evidence else "Intervention",
            "Max Units": a.max_units,
            "Marks / Unit": a.score_gain,
            "Max Gain": a.score_gain * a.max_units,
            "Cost (Lakh)": a.cost_lakh,
            "Faculty Hours": a.faculty_hours,
            "Duration (Mo)": a.duration_months,
            "Earliest Window": f"Month {a.earliest_start} → {a.earliest_finish}",
            "Prerequisites": prereq_str,
            "Schedulable (<=12 Mo)": "✅ Yes" if a.is_schedulable else "❌ Exceeds Horizon",
        })
    df_actions = pd.DataFrame(action_table_data)

    st.markdown(f"**Displaying {len(df_actions)} of {problem.num_actions} Actions**")
    if not df_actions.empty:
        st.dataframe(df_actions, use_container_width=True, height=350)
    else:
        st.info("No actions match the active filters.")

    # Gantt Chart of Actions Schedule
    st.markdown("##### Earliest-Start Schedule Timeline")
    gantt_df = []
    for a in problem.actions:
        if sel_crit and f"C{a.crit_id}" not in sel_crit:
            continue
        gantt_df.append({
            "Action": f"{a.action_id}: {a.name[:28]}...",
            "Criterion": f"C{a.crit_id}",
            "Start": a.earliest_start,
            "Finish": a.earliest_finish,
            "Duration": max(1, a.duration_months),
            "Cost": a.cost_lakh,
            "Hours": a.faculty_hours,
        })
    df_gantt = pd.DataFrame(gantt_df)

    if not df_gantt.empty:
        fig_gantt = px.bar(
            df_gantt,
            x="Duration",
            y="Action",
            base="Start",
            color="Criterion",
            orientation="h",
            hover_data=["Start", "Finish", "Duration", "Cost", "Hours"],
            title="Earliest-Start Execution Timeline",
            labels={"Duration": "Timeline (Months 0 to 12)"},
        )
        fig_gantt.add_vline(x=problem.planning_horizon, line_dash="dash", line_color="red", annotation_text=f"Horizon T={problem.planning_horizon}")
        fig_gantt.update_layout(
            height=max(360, len(df_gantt) * 18),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gantt, use_container_width=True)

    # Interactive Prerequisite Dependency Explorer
    with st.expander("🔗 Action Prerequisite Dependency Inspector"):
        action_names = [f"{a.action_id} — {a.name}" for a in problem.actions]
        sel_inspect = st.selectbox("Select an Action to Inspect Dependencies", range(len(action_names)), format_func=lambda i: action_names[i])
        target_a = problem.actions[sel_inspect]

        dep_col1, dep_col2 = st.columns(2)
        with dep_col1:
            st.markdown(f"**Action:** `{target_a.action_id}` — {target_a.name}")
            st.markdown(f"**Criterion:** C{target_a.crit_id} | **Duration:** {target_a.duration_months} mo")
            immediate_prereqs = target_a.prerequisites
            st.markdown(f"**Immediate Prerequisites:** {', '.join(immediate_prereqs) if immediate_prereqs else 'None'}")
            transitive_prereqs = [problem.actions[i].action_id for i in np.flatnonzero(problem.dep[:, target_a.index])]
            st.markdown(f"**All Transitive Prerequisites (Ancestors):** {', '.join(transitive_prereqs) if transitive_prereqs else 'None'}")
        with dep_col2:
            dependents = [problem.actions[i].action_id for i in np.flatnonzero(problem.dep[target_a.index, :])]
            st.markdown(f"**Actions That Depend on {target_a.action_id} (Descendants):** {', '.join(dependents) if dependents else 'None'}")


# ==============================================================================
# TAB 3: OPTIMIZATION STUDIO
# ==============================================================================
with tab_opt:
    st.subheader("3. Multi-Objective Optimization Studio")
    st.markdown("""
    Execute metaheuristic evolutionary algorithms or exact mathematical programming to compute the Pareto-optimal frontier.
    """)

    opt_col1, opt_col2 = st.columns([1, 2])

    with opt_col1:
        st.markdown("##### ⚙️ Optimizer Settings")
        chosen_alg = st.selectbox(
            "Algorithm Selection",
            [
                "Hybrid-Memetic (Recommended)",
                "NSGA-II (Deb et al., 2002)",
                "SPEA2 (Zitzler et al., 2001)",
                "MOEA/D (Zhang & Li, 2007)",
                "HiGHS MILP (Exact Mathematical Programming)",
                "Greedy-Baseline (Efficiency Heuristic)",
                "Random-Baseline (Monte Carlo)",
            ],
            index=0,
        )

        pop_size = st.slider("Population Size", min_value=10, max_value=200, value=40, step=10)
        max_gens = st.slider("Max Generations", min_value=10, max_value=200, value=30, step=10)
        seed = st.number_input("Random Seed", value=42, step=1)

        crossover_p = 0.9
        mutation_p = 0.15
        if any(k in chosen_alg for k in ["NSGA", "SPEA", "MOEA", "Hybrid"]):
            with st.expander("Advanced Genetic Operators"):
                crossover_p = st.slider("Crossover Probability", 0.5, 1.0, 0.9, 0.05)
                mutation_p = st.slider("Mutation Probability", 0.01, 0.5, 0.15, 0.02)

        run_btn = st.button("🚀 Run Optimizer Live", type="primary", use_container_width=True)

    with opt_col2:
        st.markdown("##### 📊 Live Run Telemetry & Benchmark History")

        run_container = st.container()

        if run_btn:
            with run_container:
                st.info(f"Executing **{chosen_alg}** on {selected_config} (Seed={seed}, Pop={pop_size}, Gens={max_gens})...")
                start_t = time.time()

                with st.spinner("Optimizing Pareto frontier..."):
                    if "Hybrid" in chosen_alg:
                        opt = MemeticNSGA2Optimizer(
                            problem=problem,
                            pop_size=pop_size,
                            max_generations=max_gens,
                            crossover_prob=crossover_p,
                            mutation_prob=mutation_p,
                            seed=seed,
                        )
                        res = opt.run()
                    elif "NSGA-II" in chosen_alg:
                        opt = NSGA2Optimizer(
                            problem=problem,
                            pop_size=pop_size,
                            max_generations=max_gens,
                            crossover_prob=crossover_p,
                            mutation_prob=mutation_p,
                            seed=seed,
                        )
                        res = opt.run()
                    elif "SPEA2" in chosen_alg:
                        opt = SPEA2Optimizer(
                            problem=problem,
                            pop_size=pop_size,
                            archive_size=pop_size,
                            max_generations=max_gens,
                            crossover_prob=crossover_p,
                            mutation_prob=mutation_p,
                            seed=seed,
                        )
                        res = opt.run()
                    elif "MOEA/D" in chosen_alg:
                        opt = MOEADOptimizer(
                            problem=problem,
                            pop_size=pop_size,
                            n_neighbors=min(15, pop_size),
                            max_generations=max_gens,
                            crossover_prob=crossover_p,
                            mutation_prob=mutation_p,
                            seed=seed,
                        )
                        res = opt.run()
                    elif "HiGHS" in chosen_alg:
                        solver = HiGHSMILPSolver(problem)
                        sols = solver.solve_epsilon_constraint_front(n_budget_steps=8, n_hour_steps=6)
                        res = OptimizationResult(
                            algorithm_name="HiGHS-MILP",
                            seed=seed,
                            elapsed_time=time.time() - start_t,
                            final_population=sols,
                            pareto_solutions=sols,
                            history=[],
                            total_evaluations=len(sols),
                        )
                    elif "Greedy" in chosen_alg:
                        opt_g = GreedyBaseline(problem=problem, seed=seed)
                        res = opt_g.run()
                    else:
                        opt_r = RandomSearchBaseline(problem=problem, seed=seed)
                        res = opt_r.run(n_samples=300)

                elapsed = time.time() - start_t
                st.success(f"Optimization completed in {elapsed:.2f} seconds! Found {len(res.pareto_solutions)} Pareto solutions.")

                # Store in session state for current config
                alg_key = chosen_alg.split(" ")[0].replace("-Baseline", "").replace("(", "")
                st.session_state["results_cache"][selected_config][alg_key] = res
                st.session_state["latest_res"] = res

        # Display latest or cached results summary
        display_res = st.session_state.get("latest_res", None)

        if display_res and display_res.pareto_solutions:
            st.markdown(f"###### Results for **{display_res.algorithm_name}**")
            sols = display_res.pareto_solutions

            obj_matrix_min = np.array([s.objectives_min for s in sols], dtype=np.float64)
            obj_matrix_nat = np.array([s.objectives for s in sols], dtype=np.float64)

            hv = compute_hypervolume(obj_matrix_min, normalize=True) if len(sols) > 0 else 0.0
            sp = compute_spacing(obj_matrix_nat) if len(sols) > 0 else 0.0
            spr = compute_spread(obj_matrix_nat) if len(sols) > 0 else 0.0

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Pareto Solutions", len(sols))
            m_col2.metric("Hypervolume (HV)", f"{hv:.4f}")
            m_col3.metric("Spacing (S)", f"{sp:.2f}")
            m_col4.metric("Spread (Δ)", f"{spr:.3f}")

            # Convergence plot if history exists
            if display_res.history:
                hist_df = pd.DataFrame(display_res.history)
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=hist_df["generation"],
                    y=hist_df["f1_max"],
                    name="Max Total Score (f1)",
                    line=dict(color="#2563EB", width=2),
                ))
                fig_hist.add_hline(
                    y=problem.min_accreditation_score,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Target R={problem.min_accreditation_score:.0f}",
                )
                fig_hist.update_layout(
                    title="Convergence History: Best NBA Score (f1) over Generations",
                    xaxis_title="Generation",
                    yaxis_title="Total Marks",
                    margin=dict(l=20, r=20, t=35, b=20),
                    height=280,
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Select an algorithm and click 'Run Optimizer Live' or view benchmark comparison below.")

    # Benchmark summary table
    st.markdown("---")
    st.markdown("##### 🏆 Precomputed Cross-Algorithm Benchmark Comparison")
    summary_path = Path("results/benchmark_metrics_summary.csv")
    raw_path = Path("results/benchmark_metrics_raw.csv")

    if summary_path.exists() and summary_path.stat().st_size > 10:
        try:
            df_summary = pd.read_csv(summary_path)
            st.dataframe(df_summary, use_container_width=True)

            if raw_path.exists() and raw_path.stat().st_size > 10:
                df_raw = pd.read_csv(raw_path)
                fig_bar = px.bar(
                    df_raw,
                    x="algorithm",
                    y="hypervolume",
                    color="algorithm",
                    title="Hypervolume Comparison Across Optimization Algorithms",
                    labels={"hypervolume": "Hypervolume (HV) [Higher is Better]"},
                )
                fig_bar.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.caption(f"Benchmark summary loading note: {e}")
    else:
        st.info("Benchmark summary CSVs not yet present in `results/`. You can run optimizers live above or run `python main.py` for full multi-seed benchmark.")


# ==============================================================================
# TAB 4: PARETO FRONTIER & TRADE-OFF ANALYSIS
# ==============================================================================
with tab_pareto:
    st.subheader("4. Pareto Frontier & Multi-Criteria Trade-off Analysis")
    st.markdown("""
    Explore the trade-offs between **Score ($f_1$)**, **Cost ($f_2$)**, **Faculty Effort ($f_3$)**, and **Criterion Balance ($f_4$)**.
    Solutions on the frontier represent Pareto-optimal plans where improving one objective requires sacrificing another.
    """)

    # Collect available pareto solutions compatible with current problem
    pareto_solutions_to_plot: Dict[str, List[EvaluationResult]] = {}

    # Check cached results for current config
    config_cache = st.session_state["results_cache"].get(selected_config, {})
    for alg_name, res in config_cache.items():
        if res.pareto_solutions:
            valid_sols = [s for s in res.pareto_solutions if len(s.x) == problem.num_actions]
            if valid_sols:
                pareto_solutions_to_plot[alg_name] = valid_sols

    # Check all_pareto_solutions.csv on disk if cache is empty
    pareto_csv = Path("results/all_pareto_solutions.csv")
    if not pareto_solutions_to_plot and pareto_csv.exists() and pareto_csv.stat().st_size > 10:
        try:
            df_all_p = pd.read_csv(pareto_csv)
            for alg, grp in df_all_p.groupby("algorithm"):
                sols_loaded: List[EvaluationResult] = []
                for _, row in grp.iterrows():
                    try:
                        x_arr = np.array(json.loads(row["decision_vector"]), dtype=int)
                        if len(x_arr) == problem.num_actions:
                            eval_s = problem.evaluate(x_arr)
                            sols_loaded.append(eval_s)
                    except Exception:
                        pass
                if sols_loaded:
                    pareto_solutions_to_plot[str(alg)] = sols_loaded
        except Exception:
            pass

    # If still empty for this configuration, compute HiGHS reference solutions instantly
    if not pareto_solutions_to_plot:
        with st.spinner("Generating initial exact HiGHS Pareto frontier..."):
            solver = HiGHSMILPSolver(problem)
            ref_sols = solver.solve_epsilon_constraint_front(n_budget_steps=8, n_hour_steps=6)
            opt_g = GreedyBaseline(problem=problem, seed=42)
            res_g = opt_g.run()

            pareto_solutions_to_plot["HiGHS-MILP"] = ref_sols
            pareto_solutions_to_plot["Greedy-Baseline"] = res_g.pareto_solutions

            config_cache["HiGHS-MILP"] = OptimizationResult(
                algorithm_name="HiGHS-MILP",
                seed=42,
                elapsed_time=0.25,
                final_population=ref_sols,
                pareto_solutions=ref_sols,
                history=[],
                total_evaluations=len(ref_sols),
            )
            config_cache["Greedy-Baseline"] = res_g

    # Flatten all pareto solutions into a plotting dataframe
    plot_rows = []
    for alg_name, sols in pareto_solutions_to_plot.items():
        for s_idx, s in enumerate(sols):
            plot_rows.append({
                "Algorithm": alg_name,
                "Solution ID": f"{alg_name} #{s_idx+1}",
                "Score (f1)": s.f1_total_score,
                "Cost (f2, Lakh)": s.f2_cost_lakh,
                "Hours (f3)": s.f3_faculty_hours,
                "Balance (f4)": s.f4_worst_criterion_ratio,
                "Active Actions": int(np.count_nonzero(s.x)),
                "Feasible": s.is_feasible,
                "eval_result": s,
            })
    df_plot = pd.DataFrame(plot_rows)

    if not df_plot.empty:
        # 3D Pareto Frontier
        st.markdown("##### 🌐 Interactive 3D Pareto Frontier (Cost vs. Hours vs. Score)")
        fig_3d = px.scatter_3d(
            df_plot,
            x="Cost (f2, Lakh)",
            y="Hours (f3)",
            z="Score (f1)",
            color="Balance (f4)",
            symbol="Algorithm",
            hover_name="Solution ID",
            hover_data=["Score (f1)", "Cost (f2, Lakh)", "Hours (f3)", "Balance (f4)", "Active Actions"],
            color_continuous_scale="Viridis",
            title="3D Pareto Surface: Implementation Cost vs. Faculty Effort vs. Total NBA Marks",
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Cost (Lakh INR)",
                yaxis_title="Faculty Hours",
                zaxis_title="Total NBA Score",
            ),
            margin=dict(l=10, r=10, t=35, b=10),
            height=500,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # 2D Projections (2x2 Grid)
        st.markdown("##### 📐 2D Trade-off Projections")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            # Score vs Cost
            fig_2d_1 = px.scatter(
                df_plot,
                x="Cost (f2, Lakh)",
                y="Score (f1)",
                color="Algorithm",
                hover_name="Solution ID",
                hover_data=["Hours (f3)", "Balance (f4)"],
                title="Total Score (f1) vs. Implementation Cost (f2)",
            )
            fig_2d_1.add_vline(x=problem.budget, line_dash="dash", line_color="red", annotation_text=f"Budget B={problem.budget}L")
            fig_2d_1.add_hline(y=problem.min_accreditation_score, line_dash="dash", line_color="green", annotation_text=f"Target R={problem.min_accreditation_score}")
            fig_2d_1.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320)
            st.plotly_chart(fig_2d_1, use_container_width=True)

            # Balance vs Cost
            fig_2d_3 = px.scatter(
                df_plot,
                x="Cost (f2, Lakh)",
                y="Balance (f4)",
                color="Algorithm",
                hover_name="Solution ID",
                title="Criterion Balance (f4) vs. Implementation Cost (f2)",
            )
            fig_2d_3.add_vline(x=problem.budget, line_dash="dash", line_color="red")
            fig_2d_3.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320)
            st.plotly_chart(fig_2d_3, use_container_width=True)

        with p_col2:
            # Score vs Hours
            fig_2d_2 = px.scatter(
                df_plot,
                x="Hours (f3)",
                y="Score (f1)",
                color="Algorithm",
                hover_name="Solution ID",
                title="Total Score (f1) vs. Faculty Hours (f3)",
            )
            fig_2d_2.add_hline(y=problem.min_accreditation_score, line_dash="dash", line_color="green")
            fig_2d_2.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320)
            st.plotly_chart(fig_2d_2, use_container_width=True)

            # Balance vs Hours
            fig_2d_4 = px.scatter(
                df_plot,
                x="Hours (f3)",
                y="Balance (f4)",
                color="Algorithm",
                hover_name="Solution ID",
                title="Criterion Balance (f4) vs. Faculty Hours (f3)",
            )
            fig_2d_4.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320)
            st.plotly_chart(fig_2d_4, use_container_width=True)

        # Interactive Table with selection
        st.markdown("##### 📋 Pareto Solutions Explorer")
        st.dataframe(
            df_plot[["Solution ID", "Algorithm", "Score (f1)", "Cost (f2, Lakh)", "Hours (f3)", "Balance (f4)", "Active Actions", "Feasible"]],
            use_container_width=True,
            height=260,
        )
    else:
        st.info("No Pareto solutions found. Please execute an algorithm in Tab 3.")


# ==============================================================================
# TAB 5: IMPLEMENTATION ROADMAP & PLAN INSPECTOR
# ==============================================================================
with tab_inspector:
    st.subheader("5. Strategic Implementation Roadmap & Plan Inspector")
    st.markdown("""
    Inspect any Pareto plan to generate its complete institutional deployment blueprint: 
    10-Criteria radar profile, 12-month earliest-start Gantt timeline, monthly faculty workload capacity check, 
    and detailed action checklist.
    """)

    # Archetype selector
    plan_archetype = st.radio(
        "Select Strategic Plan Archetype:",
        [
            "🥇 Maximum Score Plan (Maximize Accreditation Marks)",
            "💰 Minimum Cost Plan (Achieve Target with Minimal Expenditure)",
            "⚖️ Balanced Plan (Maximize Worst-Off Criterion Balance)",
            "🔍 Custom Selection from Frontier",
        ],
        horizontal=True,
    )

    # Find solutions across available fronts compatible with current problem
    all_sols_flat: List[Tuple[str, EvaluationResult]] = []
    for alg, sols in pareto_solutions_to_plot.items():
        for s in sols:
            if len(s.x) == problem.num_actions:
                all_sols_flat.append((alg, s))

    if not all_sols_flat:
        st.warning("No compatible Pareto solutions available. Please run an optimizer in Tab 3.")
        st.stop()

    active_eval: Optional[EvaluationResult] = None
    plan_title = ""

    if "Maximum Score" in plan_archetype:
        sorted_f1 = sorted(all_sols_flat, key=lambda pair: pair[1].f1_total_score, reverse=True)
        active_eval = sorted_f1[0][1]
        plan_title = f"Maximum Score Plan ({sorted_f1[0][0]})"
    elif "Minimum Cost" in plan_archetype:
        qual_sols = [pair for pair in all_sols_flat if pair[1].f1_total_score >= problem.min_accreditation_score - 1e-4]
        if qual_sols:
            sorted_f2 = sorted(qual_sols, key=lambda pair: pair[1].f2_cost_lakh)
            active_eval = sorted_f2[0][1]
            plan_title = f"Minimum Cost Plan ({sorted_f2[0][0]})"
        else:
            sorted_f2 = sorted(all_sols_flat, key=lambda pair: pair[1].f2_cost_lakh)
            active_eval = sorted_f2[0][1]
            plan_title = f"Minimum Cost Plan ({sorted_f2[0][0]})"
    elif "Balanced" in plan_archetype:
        sorted_f4 = sorted(all_sols_flat, key=lambda pair: pair[1].f4_worst_criterion_ratio, reverse=True)
        active_eval = sorted_f4[0][1]
        plan_title = f"Balanced Equity Plan ({sorted_f4[0][0]})"
    else:
        sol_options = [f"#{idx+1}: {alg} | Score={s.f1_total_score:.1f} | Cost=₹{s.f2_cost_lakh:.1f}L | Hours={s.f3_faculty_hours:.0f}h" for idx, (alg, s) in enumerate(all_sols_flat)]
        chosen_idx = st.selectbox("Select Solution", range(len(sol_options)), format_func=lambda i: sol_options[i])
        active_eval = all_sols_flat[chosen_idx][1]
        plan_title = f"Custom Selected Plan ({all_sols_flat[chosen_idx][0]})"

    # Plan Summary KPIs
    st.markdown(f"#### 📑 {plan_title}")
    pk1, pk2, pk3, pk4, pk5 = st.columns(5)

    accred_status = "5-6 Years (Full)" if active_eval.f1_total_score >= problem.min_accreditation_score else "Not Accredited"
    accred_color = "delta-good" if active_eval.f1_total_score >= problem.min_accreditation_score else "delta-bad"

    with pk1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL NBA SCORE</div>
            <div class="metric-value">{active_eval.f1_total_score:.1f} <span style="font-size: 0.9rem; color: #6B7280;">/ 1000</span></div>
            <div class="metric-delta {accred_color}">{accred_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with pk2:
        rem_budget = problem.budget - active_eval.f2_cost_lakh
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">IMPLEMENTATION COST</div>
            <div class="metric-value">₹{active_eval.f2_cost_lakh:.2f} <span style="font-size: 0.9rem; color: #6B7280;">Lakh</span></div>
            <div class="metric-delta delta-good">Remaining: ₹{rem_budget:.2f}L</div>
        </div>
        """, unsafe_allow_html=True)

    with pk3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">FACULTY EFFORT</div>
            <div class="metric-value">{active_eval.f3_faculty_hours:.0f} <span style="font-size: 0.9rem; color: #6B7280;">Hours</span></div>
            <div class="metric-delta delta-good">Avg: {(active_eval.f3_faculty_hours/problem.planning_horizon):.0f} hrs/mo</div>
        </div>
        """, unsafe_allow_html=True)

    with pk4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">WORST-OFF RATIO (f4)</div>
            <div class="metric-value">{(active_eval.f4_worst_criterion_ratio * 100):.1f}%</div>
            <div class="metric-delta delta-good">Criterion Fairness</div>
        </div>
        """, unsafe_allow_html=True)

    with pk5:
        active_count = int(np.count_nonzero(active_eval.x))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ACTIONS ACTIVATED</div>
            <div class="metric-value">{active_count} <span style="font-size: 0.9rem; color: #6B7280;">/ {problem.num_actions}</span></div>
            <div class="metric-delta delta-good">{(active_count/problem.num_actions*100):.1f}% Action Density</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar Chart & Monthly Workload
    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        st.markdown("##### 🕸️ 10-Criteria Achievement Radar Profile")
        categories = [f"C{c.id}: {c.code}" for c in problem.criteria]
        categories.append(categories[0])  # Close polygon

        plan_scores = list(active_eval.criteria_scores)
        plan_scores.append(plan_scores[0])

        baseline_scores = list(problem.s0)
        baseline_scores.append(baseline_scores[0])

        floor_scores = list(problem.L)
        floor_scores.append(floor_scores[0])

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=plan_scores,
            theta=categories,
            fill='toself',
            name="Selected Plan Score",
            line_color='#2563EB',
            fillcolor='rgba(37, 99, 235, 0.25)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_scores,
            theta=categories,
            name="Baseline Score (s0)",
            line=dict(color='#9CA3AF', dash='dash'),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=floor_scores,
            theta=categories,
            name="Minimum Floor (L_c)",
            line=dict(color='#DC2626', dash='dot'),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, float(np.max(problem.M)) * 1.05])),
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with r_col2:
        st.markdown("##### ⏱️ Monthly Faculty Workload")
        months = [f"M{m}" for m in range(1, problem.planning_horizon + 1)]
        monthly_load = active_eval.monthly_hours

        fig_workload = go.Figure()
        fig_workload.add_trace(go.Bar(
            x=months,
            y=monthly_load,
            name="Assigned Faculty Hours",
            marker_color=["#EF4444" if h > problem.monthly_capacity else "#3B82F6" for h in monthly_load],
            text=[f"{h:.0f}h" for h in monthly_load],
            textposition="auto",
        ))
        fig_workload.add_hline(
            y=problem.monthly_capacity,
            line_dash="dash",
            line_color="#DC2626",
            line_width=2,
            annotation_text=f"Capacity Limit K_m = {problem.monthly_capacity:.0f} hrs",
        )
        fig_workload.update_layout(
            yaxis_title="Faculty Person-Hours",
            xaxis_title=f"Implementation Month (1 to {problem.planning_horizon})",
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
        )
        st.plotly_chart(fig_workload, use_container_width=True)

    # Earliest-Start Gantt for Chosen Plan
    st.markdown("##### 📅 Selected Plan Action Deployment Timeline")
    active_action_rows = []
    for a in problem.actions:
        units = int(active_eval.x[a.index])
        if units > 0:
            active_action_rows.append({
                "Action ID": a.action_id,
                "Action Name": f"{a.action_id}: {a.name}",
                "Criterion": f"C{a.crit_id}",
                "Units": units,
                "Marks Gain": a.score_gain * units,
                "Expenditure (Lakh)": a.cost_lakh * units,
                "Total Hours": a.faculty_hours * units,
                "Start Month": a.earliest_start,
                "Finish Month": a.earliest_finish,
                "Duration": max(1, a.duration_months),
                "Prerequisites": ", ".join(a.prerequisites) if a.prerequisites else "None",
            })
    df_active_actions = pd.DataFrame(active_action_rows)

    if not df_active_actions.empty:
        fig_active_gantt = px.bar(
            df_active_actions,
            x="Duration",
            y="Action Name",
            base="Start Month",
            color="Criterion",
            orientation="h",
            hover_data=["Units", "Expenditure (Lakh)", "Total Hours", "Marks Gain", "Start Month", "Finish Month"],
            title="Earliest-Start Execution Timeline for Selected Actions",
        )
        fig_active_gantt.add_vline(x=problem.planning_horizon, line_dash="dash", line_color="red")
        fig_active_gantt.update_layout(
            height=max(320, len(df_active_actions) * 22),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_active_gantt, use_container_width=True)

        # Action Checklist & BOQ Table
        st.markdown("##### 📋 Implementation Bill of Quantities (BOQ)")
        st.dataframe(df_active_actions, use_container_width=True)

        # Export Buttons
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = df_active_actions.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Implementation Roadmap (CSV)",
                data=csv_data,
                file_name="NBA_Implementation_Roadmap.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col2:
            json_dict = {
                "plan_title": plan_title,
                "total_score": active_eval.f1_total_score,
                "cost_lakh": active_eval.f2_cost_lakh,
                "faculty_hours": active_eval.f3_faculty_hours,
                "worst_ratio": active_eval.f4_worst_criterion_ratio,
                "is_feasible": active_eval.is_feasible,
                "actions": df_active_actions.to_dict(orient="records"),
            }
            st.download_button(
                label="📥 Download Blueprint (JSON)",
                data=json.dumps(json_dict, indent=2).encode("utf-8"),
                file_name="NBA_Implementation_Blueprint.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info("No improvement actions are deployed in this plan (baseline profile only).")


# ==============================================================================
# TAB 6: INTERACTIVE WHAT-IF SCENARIO SIMULATOR
# ==============================================================================
with tab_whatif:
    st.subheader("6. Interactive 'What-If' Planning Simulator")
    st.markdown("""
    Test hypothetical institutional investment strategies. Adjust individual action units manually, 
    observe real-time score gains and constraint violations, or click **✨ Auto-Repair Plan** 
    to automatically enforce mathematical feasibility!
    """)

    # Ensure what-if vector matches current problem dimension
    if "whatif_vector" not in st.session_state or len(st.session_state["whatif_vector"]) != problem.num_actions:
        st.session_state["whatif_vector"] = np.zeros(problem.num_actions, dtype=int)
        st.session_state["whatif_nonce"] = st.session_state.get("whatif_nonce", 0) + 1

    sim_c1, sim_c2, sim_c3 = st.columns([1, 1, 2])
    with sim_c1:
        if st.button("✨ Auto-Repair Plan (Fix Violations)", type="primary", use_container_width=True):
            repaired_x = problem.repair(st.session_state["whatif_vector"])
            st.session_state["whatif_vector"] = repaired_x.copy()
            st.session_state["whatif_nonce"] += 1
            st.toast("Solution successfully repaired to satisfy all constraints!", icon="✅")
            st.rerun()
    with sim_c2:
        if st.button("🔄 Reset Plan to Baseline", use_container_width=True):
            st.session_state["whatif_vector"] = np.zeros(problem.num_actions, dtype=int)
            st.session_state["whatif_nonce"] += 1
            st.rerun()

    # Evaluate current what-if vector
    current_eval = problem.evaluate(st.session_state["whatif_vector"])

    # Simulator KPI Dashboard
    sim_k1, sim_k2, sim_k3, sim_k4 = st.columns(4)

    feas_badge = "✅ FULLY FEASIBLE" if current_eval.is_feasible else f"❌ INFEASIBLE (Viol: {current_eval.total_violation:.1f})"
    feas_color = "delta-good" if current_eval.is_feasible else "delta-bad"

    with sim_k1:
        score_diff = current_eval.f1_total_score - problem.min_accreditation_score
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PROJECTED SCORE (f1)</div>
            <div class="metric-value">{current_eval.f1_total_score:.1f} <span style="font-size: 0.9rem; color: #6B7280;">/ 1000</span></div>
            <div class="metric-delta {'delta-good' if score_diff >= 0 else 'delta-bad'}">
                {'+' if score_diff >= 0 else ''}{score_diff:.1f} vs Target ({problem.min_accreditation_score:.0f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sim_k2:
        cost_ok = current_eval.f2_cost_lakh <= problem.budget
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PROJECTED COST (f2)</div>
            <div class="metric-value">₹{current_eval.f2_cost_lakh:.2f} <span style="font-size: 0.9rem; color: #6B7280;">Lakh</span></div>
            <div class="metric-delta {'delta-good' if cost_ok else 'delta-bad'}">
                Ceiling: ₹{problem.budget:.1f}L ({'Within Budget' if cost_ok else 'OVER BUDGET'})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sim_k3:
        max_monthly = float(np.max(current_eval.monthly_hours))
        workload_ok = max_monthly <= problem.monthly_capacity
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PEAK FACULTY LOAD (H_m)</div>
            <div class="metric-value">{max_monthly:.0f} <span style="font-size: 0.9rem; color: #6B7280;">hrs/mo</span></div>
            <div class="metric-delta {'delta-good' if workload_ok else 'delta-bad'}">
                Limit: {problem.monthly_capacity:.0f}h ({'Compliant' if workload_ok else 'EXCEEDED'})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sim_k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">FEASIBILITY STATUS</div>
            <div class="metric-value" style="font-size: 1.25rem; font-weight: 800;">{feas_badge}</div>
            <div class="metric-delta {feas_color}">
                {'Ready for Deployment' if current_eval.is_feasible else 'Click Auto-Repair to fix'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Violations Details Callout if infeasible
    if not current_eval.is_feasible:
        with st.expander("⚠️ View Detailed Constraint Violations Breakdown", expanded=True):
            v_cols = st.columns(3)
            with v_cols[0]:
                st.markdown(f"• **Budget Overrun:** ₹{current_eval.violations['budget']:.2f} Lakh")
                st.markdown(f"• **Monthly Hours Overload:** {current_eval.violations['monthly_hours']:.1f} hrs")
            with v_cols[1]:
                st.markdown(f"• **Prerequisites Unfulfilled:** {current_eval.violations['prerequisites']:.0f} units")
                st.markdown(f"• **Deadline/Horizon Violations:** {current_eval.violations['deadline']:.0f} units")
            with v_cols[2]:
                st.markdown(f"• **Criterion Below Quality Floor:** {current_eval.violations['criteria_min']:.1f} marks")
                st.markdown(f"• **Target Score Shortfall:** {current_eval.violations['target_score']:.1f} marks")

    # Action Unit Adjusters Grouped by Criterion
    st.markdown("##### 🎛️ Adjust Improvement Action Units by Criterion")
    nonce = st.session_state["whatif_nonce"]

    for c in problem.criteria:
        c_actions = [a for a in problem.actions if a.crit_id == c.id]
        c_score = current_eval.criteria_scores[c.id - 1]
        status_symbol = "⚠️" if c_score < c.min_threshold else "✅"

        with st.expander(f"{status_symbol} Criterion {c.id}: {c.code} — {c.name} (Score: {c_score:.1f} / {c.max_score:.0f})"):
            c_cols = st.columns(3)
            for idx, a in enumerate(c_actions):
                col_target = c_cols[idx % 3]
                with col_target:
                    curr_val = int(st.session_state["whatif_vector"][a.index])
                    widget_key = f"whatif_{selected_config}_{nonce}_{a.index}"
                    new_val = st.number_input(
                        f"**{a.action_id}**: {a.name[:30]}",
                        min_value=0,
                        max_value=a.max_units,
                        value=min(curr_val, a.max_units),
                        step=1,
                        key=widget_key,
                        help=f"Cost: ₹{a.cost_lakh}L | Effort: {a.faculty_hours}h | Gain: +{a.score_gain} marks | Prereqs: {','.join(a.prerequisites) if a.prerequisites else 'None'}",
                    )
                    if new_val != curr_val:
                        st.session_state["whatif_vector"][a.index] = new_val
                        st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.8rem;">
    National Board of Accreditation (NBA) Improvement Planning Optimization DSS | Python 3.11+ • Streamlit • SciPy HiGHS • Pymoo
</div>
""", unsafe_allow_html=True)
