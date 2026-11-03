"""Ablate heuristic corner seeding and local search on the medium instance."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from benchmark_v2 import _metrics
from src.algorithms.nsga2 import NSGA2Optimizer
from src.baseline.greedy import GreedyRepaired
from src.hybrid.corner_nsga2 import CornerNSGA2
from src.problem import NBAProblem


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "config.yaml"
REFERENCE = ROOT / "results" / "milp_dense_front.json"
OUTPUT = ROOT / "results" / "ablation_runs.csv"
VARIANTS = ("V0", "V1", "V2", "V3")


def _reference_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = __import__("json").loads(REFERENCE.read_text(encoding="utf-8"))
    matrix = np.asarray([plan["objectives_min"] for plan in payload["plans"]], dtype=float)
    return matrix, np.min(matrix, axis=0), np.max(matrix, axis=0)


def _run_one(
    variant: str,
    seed: int,
    reference: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> dict[str, Any]:
    problem = NBAProblem(CONFIG)
    params = problem.config["algorithm_parameters"]
    common = {
        "pop_size": int(params["population_size"]),
        "max_generations": int(params["max_generations"]),
        "crossover_prob": float(params["crossover_probability"]),
        "mutation_prob": float(params["mutation_probability"]),
        "seed": seed,
    }
    if variant == "V0":
        optimizer = NSGA2Optimizer(problem, **common)
    else:
        optimizer = CornerNSGA2(
            problem,
            **common,
            extreme_seeding=variant in ("V1", "V3"),
            corner_local_search=variant in ("V2", "V3"),
        )
    result = optimizer.run()
    metrics = _metrics(result, reference, ideal, nadir)
    feasible = [solution for solution in result.pareto_solutions if solution.is_feasible]
    objectives = np.asarray([solution.objectives for solution in feasible], dtype=float)
    return {
        "variant": variant,
        "seed": seed,
        "hypervolume": metrics["hypervolume"],
        "igd": metrics["igd"],
        "matched_or_dominated_pct": metrics["milp_front_match_or_dominate_pct"],
        "min_cost": float(np.min(objectives[:, 1])),
        "min_hours": float(np.min(objectives[:, 2])),
        "max_f4": float(np.max(objectives[:, 3])),
        "runtime_sec": metrics["runtime_sec"],
        "evaluations": result.total_evaluations,
    }


def _run_one_worker(argument: tuple[str, int, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
    return _run_one(*argument)


def _write_rows(rows: list[dict[str, Any]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


def _greedy_report() -> None:
    result = GreedyRepaired(NBAProblem(CONFIG), seed=0).run()
    feasible = [solution for solution in result.final_population if solution.is_feasible]
    print("\nGreedyRepaired:")
    print(f"  feasible plans: {len(feasible)}/{len(result.final_population)}")
    for solution in feasible:
        print(f"  objectives={solution.objectives.tolist()}; feasible=True")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.jobs < 1:
        parser.error("--seeds and --jobs must be at least 1")

    reference, ideal, nadir = _reference_data()
    rows: list[dict[str, Any]] = []
    if args.resume and OUTPUT.exists():
        rows = pd.read_csv(OUTPUT).to_dict("records")
    completed = {(str(row["variant"]), int(row["seed"])) for row in rows}
    jobs = [
        (variant, seed)
        for seed in range(args.seeds)
        for variant in VARIANTS
        if (variant, seed) not in completed
    ]
    arguments = [
        (variant, seed, reference, ideal, nadir) for variant, seed in jobs
    ]
    if args.jobs == 1:
        results = [_run_one(*argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(_run_one_worker, arguments))
    for row in results:
        rows.append(row)
        _write_rows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT}")

    frame = pd.DataFrame(rows).sort_values(["variant", "seed"])
    print("\nAblation summary (mean +/- std):")
    summary = frame.groupby("variant").agg(
        hypervolume_mean=("hypervolume", "mean"),
        hypervolume_std=("hypervolume", "std"),
        igd_mean=("igd", "mean"),
        igd_std=("igd", "std"),
        matched_mean=("matched_or_dominated_pct", "mean"),
        matched_std=("matched_or_dominated_pct", "std"),
        min_cost_mean=("min_cost", "mean"),
        min_hours_mean=("min_hours", "mean"),
        max_f4_mean=("max_f4", "mean"),
        runtime_mean=("runtime_sec", "mean"),
    )
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))

    v0 = frame[frame["variant"] == "V0"]
    print("\nMann-Whitney U p-values vs V0:")
    for variant in VARIANTS[1:]:
        current = frame[frame["variant"] == variant]
        hv_p = mannwhitneyu(v0["hypervolume"], current["hypervolume"], alternative="two-sided").pvalue
        match_p = mannwhitneyu(
            v0["matched_or_dominated_pct"],
            current["matched_or_dominated_pct"],
            alternative="two-sided",
        ).pvalue
        print(f"  {variant}: hypervolume={hv_p:.6g}; matched={match_p:.6g}")
    _greedy_report()


if __name__ == "__main__":
    main()
