"""Create figures for the final medium-instance benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "final_runs.csv"
DENSE = ROOT / "results" / "milp_dense_front.json"
FIGURES = ROOT / "figures"
METHODS = [
    "Random-Baseline",
    "Greedy-Baseline",
    "GreedyRepaired",
    "NSGA-II",
    "SPEA2",
    "MOEA/D",
    "Hybrid-Memetic (no MILP seed)",
    "NSGA-II+Corner",
]
LABELS = {
    "Random-Baseline": "Random",
    "Greedy-Baseline": "Greedy",
    "GreedyRepaired": "GreedyRepaired",
    "NSGA-II": "NSGA-II",
    "SPEA2": "SPEA2",
    "MOEA/D": "MOEA/D",
    "Hybrid-Memetic (no MILP seed)": "Hybrid",
    "NSGA-II+Corner": "NSGA-II+Corner",
}
COLORS = dict(zip(METHODS, plt.get_cmap("tab10").colors))


def _load() -> tuple[pd.DataFrame, dict]:
    runs = pd.read_csv(RUNS)
    dense = json.loads(DENSE.read_text(encoding="utf-8"))
    return runs[runs["method"] != "Dense MILP front"].copy(), dense


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def _boxplot(
    runs: pd.DataFrame,
    column: str,
    output: Path,
    title: str,
    ylabel: str,
    note: str | None = None,
) -> None:
    values = [runs.loc[runs["method"] == method, column].dropna() for method in METHODS]
    fig, axis = plt.subplots(figsize=(13, 6))
    box = axis.boxplot(values, patch_artist=True, tick_labels=[LABELS[m] for m in METHODS])
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.75)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    if note:
        axis.text(0.01, -0.22, note, transform=axis.transAxes, fontsize=9)
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs, dense = _load()
    dense_plans = pd.DataFrame(dense["plans"])
    dense_actual = np.asarray([plan["objectives"] for plan in dense["plans"]], dtype=float)
    dense_hv = float(dense.get("hypervolume", np.nan))
    if not np.isfinite(dense_hv):
        dense_hv = float(
            pd.read_csv(ROOT / "results" / "final_summary.csv")
            .loc[lambda frame: frame["method"] == "Dense MILP front", "hypervolume_mean"]
            .iloc[0]
        )

    fig, axis = plt.subplots(figsize=(13, 6))
    values = [runs.loc[runs["method"] == method, "hypervolume"].dropna() for method in METHODS]
    box = axis.boxplot(values, patch_artist=True, tick_labels=[LABELS[m] for m in METHODS])
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.75)
    axis.axhline(dense_hv, color="black", linestyle="--", label="Dense MILP front")
    axis.set_title("Hypervolume by method")
    axis.set_ylabel("Hypervolume")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_hypervolume_box.png", dpi=160)
    plt.close(fig)

    finite_runs = runs[np.isfinite(runs["igd"])]
    _boxplot(
        finite_runs,
        "igd",
        FIGURES / "fig_igd_box.png",
        "IGD by method",
        "IGD to dense MILP front",
        "Note: runs with infinite IGD are excluded because no feasible front was available.",
    )

    corner_columns = [("min_cost", "Minimum cost", "Lakh INR", "min"),
                      ("min_hours", "Minimum faculty hours", "Hours", "min"),
                      ("max_f4", "Maximum balance ratio", "f4", "max")]
    dense_best = [dense_actual[:, 1].min(), dense_actual[:, 2].min(), dense_actual[:, 3].max()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(METHODS))
    for axis, (index, (column, title, ylabel, _)), best in zip(range(3), enumerate(corner_columns), dense_best):
        values = runs.groupby("method")[column].agg(["mean", "std"]).reindex(METHODS)
        axes[index].errorbar(
            x, values["mean"], yerr=values["std"], fmt="o", capsize=4,
            color="#1f4e79",
        )
        axes[index].axhline(best, color="black", linestyle="--", label="Dense front")
        axes[index].set_title(title)
        axes[index].set_ylabel(ylabel)
        axes[index].set_xticks(x, [LABELS[m] for m in METHODS], rotation=70, ha="right")
        _style_axis(axes[index])
    axes[-1].legend()
    fig.suptitle("Corner coverage across stochastic methods")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_corners.png", dpi=160)
    plt.close(fig)

    means = runs.groupby("method")["matched_pct"].mean().reindex(METHODS)
    fig, axis = plt.subplots(figsize=(13, 6))
    axis.bar([LABELS[m] for m in METHODS], means, color=[COLORS[m] for m in METHODS])
    axis.set_title("Mean percentage of dense-front plans matched or dominated")
    axis.set_ylabel("Matched or dominated reference plans (%)")
    axis.tick_params(axis="x", rotation=25)
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_matched.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        dense_actual[:, 1], dense_actual[:, 0], c=dense_actual[:, 2],
        cmap="viridis", s=45, edgecolors="black", linewidths=0.3,
    )
    axis.set_title("Dense MILP reference front")
    axis.set_xlabel("Total cost (Lakh INR)")
    axis.set_ylabel("Total score (marks)")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Faculty hours")
    _style_axis(axis)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_dense_front.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
