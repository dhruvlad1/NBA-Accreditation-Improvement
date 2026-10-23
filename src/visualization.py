"""
Publication-quality visualization module for NBA Accreditation Improvement Planning.
Includes 2D/3D Pareto front projections, parallel coordinates, criteria radar charts,
Gantt schedule timeline with monthly faculty workload, and convergence comparisons.

All generated plots are explicitly marked SYNTHETIC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.nsga2 import OptimizationResult
from src.problem import EvaluationResult, NBAProblem

# Consistent high-aesthetic palette
PALETTE = {
    "NSGA-II": "#1f77b4",          # Deep Blue
    "SPEA2": "#ff7f0e",            # Amber / Warm Orange
    "MOEA/D": "#2ca02c",           # Emerald Green
    "Hybrid-Memetic": "#d62728",   # Crimson / Ruby Red
    "HiGHS-MILP": "#9467bd",       # Royal Violet
    "Greedy-Baseline": "#8c564b",  # Sienna
    "Random-Baseline": "#7f7f7f",  # Neutral Slate
    "Baseline-s0": "#e377c2",      # Pink
    "Threshold": "#d62728",        # Red
}

STYLE_CONFIG = {
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
}


def set_plot_style() -> None:
    """Apply clean, modern publication styling."""
    plt.rcParams.update(STYLE_CONFIG)


def plot_pareto_2d_projections(
    results_map: Dict[str, OptimizationResult],
    problem: NBAProblem,
    save_path: str | Path | None = None
) -> plt.Figure:
    """
    Plot 2x2 grid of 2D Pareto front projections:
    (a) Total Score f1 vs Cost f2
    (b) Total Score f1 vs Faculty Hours f3
    (c) Cost f2 vs Worst Criterion Ratio f4
    (d) Faculty Hours f3 vs Worst Criterion Ratio f4
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # Subplot pairs: (x_key, y_key, xlabel, ylabel, title)
    pairs = [
        (
            lambda s: s.f2_cost_lakh,
            lambda s: s.f1_total_score,
            "Total Cost f2 (Lakh INR) [Min]",
            "Total NBA Score f1 (Marks) [Max]",
            "Total Score vs. Cost Trade-Off",
        ),
        (
            lambda s: s.f3_faculty_hours,
            lambda s: s.f1_total_score,
            "Total Faculty Hours f3 (Hours) [Min]",
            "Total NBA Score f1 (Marks) [Max]",
            "Total Score vs. Faculty Hours Trade-Off",
        ),
        (
            lambda s: s.f2_cost_lakh,
            lambda s: s.f4_worst_criterion_ratio,
            "Total Cost f2 (Lakh INR) [Min]",
            "Worst Criterion Ratio f4 (Score/Max) [Max]",
            "Fairness (f4) vs. Cost Trade-Off",
        ),
        (
            lambda s: s.f3_faculty_hours,
            lambda s: s.f4_worst_criterion_ratio,
            "Total Faculty Hours f3 (Hours) [Min]",
            "Worst Criterion Ratio f4 (Score/Max) [Max]",
            "Fairness (f4) vs. Faculty Hours Trade-Off",
        ),
    ]

    for ax_idx, (get_x, get_y, xlabel, ylabel, title) in enumerate(pairs):
        ax = axes[ax_idx // 2, ax_idx % 2]
        ax.grid(True)

        for name, res in results_map.items():
            sols = res.pareto_solutions
            if not sols:
                continue
            xs = [get_x(s) for s in sols]
            ys = [get_y(s) for s in sols]
            color = PALETTE.get(name, "#333333")
            marker = "o" if "Hybrid" in name else ("s" if "NSGA" in name else ("^" if "SPEA" in name else "D"))
            ax.scatter(xs, ys, label=name, color=color, marker=marker, s=45, alpha=0.85, edgecolors="none" if marker != "D" else None)

        # Add problem thresholds
        if ax_idx == 0:  # f2 vs f1
            ax.axvline(problem.budget, color="#b22222", linestyle=":", linewidth=1.5, label=f"Budget B={problem.budget}L")
            ax.axhline(problem.min_accreditation_score, color="#2e8b57", linestyle=":", linewidth=1.5, label=f"Target R={problem.min_accreditation_score}")
        elif ax_idx == 1:  # f3 vs f1
            ax.axhline(problem.min_accreditation_score, color="#2e8b57", linestyle=":", linewidth=1.5)

        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        if ax_idx == 0:
            ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)

    fig.suptitle(
        "[SYNTHETIC] NBA Accreditation Improvement Planning: Multi-Objective Pareto Projections",
        fontsize=15,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved Pareto 2D plot to {save_path}")

    return fig


def plot_pareto_3d(
    pareto_solutions: List[EvaluationResult],
    title: str = "3D Pareto Surface",
    save_path: str | Path | None = None
) -> plt.Figure:
    """
    Plot 3D scatter: Cost (f2) vs Faculty Hours (f3) vs Total Score (f1),
    with color mapped to Worst Criterion Ratio (f4).
    """
    set_plot_style()
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    f2 = np.array([s.f2_cost_lakh for s in pareto_solutions])
    f3 = np.array([s.f3_faculty_hours for s in pareto_solutions])
    f1 = np.array([s.f1_total_score for s in pareto_solutions])
    f4 = np.array([s.f4_worst_criterion_ratio for s in pareto_solutions])

    sc = ax.scatter(f2, f3, f1, c=f4, cmap="viridis", s=60, alpha=0.9, edgecolors="k", linewidths=0.5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Worst-Off Ratio f4 (min_c Score_c / M_c)", fontweight="bold")

    ax.set_xlabel("Cost f2 (Lakh INR)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Faculty Hours f3", fontweight="bold", labelpad=10)
    ax.set_zlabel("Total Score f1 (Marks)", fontweight="bold", labelpad=10)
    ax.set_title(f"[SYNTHETIC] {title}", fontweight="bold", fontsize=13, pad=15)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved 3D Pareto plot to {save_path}")

    return fig


def plot_criteria_radar(
    problem: NBAProblem,
    solutions_to_compare: Dict[str, EvaluationResult],
    save_path: str | Path | None = None
) -> plt.Figure:
    """
    Radar (spider) chart of the 10 NBA criteria satisfaction ratios (Score_c / M_c).
    Compares baseline s0, minimum threshold L, and representative Pareto solutions.
    """
    set_plot_style()
    num_crit = problem.num_criteria
    angles = np.linspace(0, 2 * np.pi, num_crit, endpoint=False).tolist()
    angles += angles[:1]  # Complete loop

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), dpi=300)

    # 1. Plot baseline s0
    s0_ratios = (problem.s0 / problem.M).tolist()
    s0_ratios += s0_ratios[:1]
    ax.plot(angles, s0_ratios, color="#7f7f7f", linestyle="--", linewidth=1.5, label="Baseline (s0)")
    ax.fill(angles, s0_ratios, color="#7f7f7f", alpha=0.1)

    # 2. Plot minimum threshold floor L
    l_ratios = (problem.L / problem.M).tolist()
    l_ratios += l_ratios[:1]
    ax.plot(angles, l_ratios, color="#d62728", linestyle=":", linewidth=2, label="Minimum Threshold (L_c)")

    # 3. Plot candidate solutions
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for idx, (label, sol) in enumerate(solutions_to_compare.items()):
        ratios = (sol.criteria_scores / problem.M).tolist()
        ratios += ratios[:1]
        c = colors[idx % len(colors)]
        ax.plot(angles, ratios, color=c, linewidth=2.2, label=f"{label} (f1={sol.f1_total_score:.0f}, Cost={sol.f2_cost_lakh:.1f}L)")
        ax.fill(angles, ratios, color=c, alpha=0.15)

    labels = [f"C{c.id}\n({c.code})" for c in problem.criteria]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="#555555")

    ax.set_title(
        "[SYNTHETIC] 10 NBA Criteria Attainment Radar (Score_c / M_c)",
        fontsize=13,
        fontweight="bold",
        pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved radar chart to {save_path}")

    return fig


def plot_schedule_gantt_and_workload(
    problem: NBAProblem,
    solution: EvaluationResult,
    save_path: str | Path | None = None
) -> plt.Figure:
    """
    Two-panel figure:
    Top: Earliest-start Gantt chart for active actions (x_a > 0).
    Bottom: Monthly faculty workload (H_m) vs capacity K_m = 300.
    """
    set_plot_style()
    fig, (ax_gantt, ax_hours) = plt.subplots(2, 1, figsize=(13, 10), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

    active_indices = [a.index for a in problem.actions if solution.x[a.index] > 0]
    active_actions = [problem.actions[idx] for idx in active_indices]

    # Sort by start time and criterion
    active_actions.sort(key=lambda a: (a.earliest_start, a.crit_id))

    y_ticks: List[int] = []
    y_labels: List[str] = []

    for i, a in enumerate(active_actions):
        units = solution.x[a.index]
        color = "#ff7f0e" if a.is_evidence else "#1f77b4"
        start = a.earliest_start
        dur = a.duration_months
        ax_gantt.barh(
            i, dur, left=start, height=0.65,
            color=color, edgecolor="black", linewidth=0.5, alpha=0.85
        )
        ax_gantt.text(
            start + dur / 2.0, i,
            f"x={units} ({dur}m, {a.faculty_hours * units:.0f}h)",
            ha="center", va="center", color="white", fontsize=7.5, fontweight="bold"
        )
        y_ticks.append(i)
        y_labels.append(f"C{a.crit_id} {a.action_id}")

    ax_gantt.set_yticks(y_ticks)
    ax_gantt.set_yticklabels(y_labels, fontsize=7.5)
    ax_gantt.invert_yaxis()
    ax_gantt.grid(True, axis="x")
    ax_gantt.set_ylabel("Selected Actions", fontweight="bold")
    ax_gantt.set_title(
        f"[SYNTHETIC] Earliest-Start Schedule (f1={solution.f1_total_score:.1f}, Cost={solution.f2_cost_lakh:.1f}L, Hours={solution.f3_faculty_hours:.0f}h)",
        fontsize=12, fontweight="bold"
    )

    # Bottom panel: Monthly Faculty Workload
    months = np.arange(1, problem.planning_horizon + 1)
    monthly_h = solution.monthly_hours

    bar_colors = ["#2ca02c" if h <= problem.monthly_capacity else "#d62728" for h in monthly_h]
    ax_hours.bar(months, monthly_h, width=0.6, color=bar_colors, edgecolor="black", alpha=0.85, label="Monthly Workload H_m")
    ax_hours.axhline(problem.monthly_capacity, color="#d62728", linestyle="--", linewidth=2.0, label=f"Max Capacity K_m={problem.monthly_capacity}h")

    for m_idx, h in enumerate(monthly_h):
        ax_hours.text(m_idx + 1, h + 8, f"{h:.0f}h", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax_hours.set_xlabel("Planning Horizon (Months)", fontweight="bold", fontsize=11)
    ax_hours.set_ylabel("Faculty Hours/Mo", fontweight="bold", fontsize=11)
    ax_hours.set_xticks(months)
    ax_hours.set_xticklabels([f"Month {m}" for m in months])
    ax_hours.set_ylim(0, max(np.max(monthly_h) * 1.25, problem.monthly_capacity * 1.2))
    ax_hours.grid(True, axis="y")
    ax_hours.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved schedule Gantt chart to {save_path}")

    return fig


def plot_convergence_history(
    results_map: Dict[str, OptimizationResult],
    save_path: str | Path | None = None
) -> plt.Figure:
    """
    Plot convergence history across generations (Max f1 score and archive size).
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for name, res in results_map.items():
        if not res.history:
            continue
        gens = [entry["generation"] for entry in res.history]
        f1_vals = [entry["f1_max"] for entry in res.history]
        arch_sizes = [entry["archive_size"] for entry in res.history]
        color = PALETTE.get(name, "#333333")

        ax1.plot(gens, f1_vals, label=name, color=color, linewidth=2.0)
        ax2.plot(gens, arch_sizes, label=name, color=color, linewidth=2.0)

    ax1.set_xlabel("Generation", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Maximum Total NBA Score f1", fontweight="bold", fontsize=11)
    ax1.set_title("Total Score (f1) Convergence", fontweight="bold", fontsize=12)
    ax1.grid(True)
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Generation", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Pareto Archive Cardinality", fontweight="bold", fontsize=11)
    ax2.set_title("Non-Dominated Archive Growth", fontweight="bold", fontsize=12)
    ax2.grid(True)
    ax2.legend(loc="lower right")

    fig.suptitle(
        "[SYNTHETIC] Evolutionary Algorithm Convergence Over Generations",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved convergence plot to {save_path}")

    return fig


def plot_algorithm_comparison_bars(
    metrics_summary_df: "pd.DataFrame",
    save_path: str | Path | None = None
) -> plt.Figure:
    """
    Bar plots comparing Hypervolume, Spacing, Spread, and Execution Time across algorithms.
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    cols = [
        ("hypervolume", "Hypervolume (HV) [Higher is Better]", True),
        ("spacing", "Spacing Metric (S) [Lower is Better]", False),
        ("spread", "Spread (Delta) [Lower is Better]", False),
        ("time_sec", "Runtime (Seconds) [Lower is Better]", False),
    ]

    df = metrics_summary_df
    algs = df["algorithm"].tolist()
    colors = [PALETTE.get(a, "#333333") for a in algs]

    for ax_idx, (col_name, title, higher_better) in enumerate(cols):
        ax = axes[ax_idx // 2, ax_idx % 2]
        if col_name not in df.columns:
            continue
        vals = df[col_name].values
        bars = ax.bar(algs, vals, color=colors, edgecolor="black", alpha=0.85, width=0.55)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height * 1.01,
                f"{height:.3f}" if height < 10 else f"{height:.1f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(algs, rotation=20, ha="right", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y")

    fig.suptitle(
        "[SYNTHETIC] Multi-Objective Performance Benchmark Comparison",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved algorithm comparison bar plot to {save_path}")

    return fig
