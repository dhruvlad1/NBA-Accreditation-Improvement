"""Configured baseline benchmark with corrected accounting and feasible metrics.

Compared with ``benchmark.py``, this version uses the configured evolutionary
run, counts actual objective evaluations, evaluates only feasible plans for
quality metrics, normalizes against a fixed MILP reference front, and supports
selectable instances and seed counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.algorithms.moead import MOEADOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer, OptimizationResult
from src.algorithms.spea2 import SPEA2Optimizer
from src.baseline.greedy import GreedyBaseline
from src.baseline.milp_solver import HiGHSMILPSolver
from src.baseline.random_search import RandomSearchBaseline
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.metrics import compute_hypervolume, compute_igd
from src.problem import EvaluationResult, NBAProblem


ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = 10
INSTANCE_CONFIGS = {
    "small": ROOT / "experiments" / "config_small.yaml",
    "medium": ROOT / "experiments" / "config.yaml",
}


def _counted_run(
    problem: NBAProblem,
    factory: Callable[[], OptimizationResult],
) -> Tuple[OptimizationResult, int]:
    """Run an algorithm while counting every call to NBAProblem.evaluate."""
    original_evaluate = problem.evaluate
    calls = 0

    def counted_evaluate(x: np.ndarray) -> EvaluationResult:
        nonlocal calls
        calls += 1
        return original_evaluate(x)

    problem.evaluate = counted_evaluate  # type: ignore[method-assign]
    try:
        result = factory()
    finally:
        problem.evaluate = original_evaluate  # type: ignore[method-assign]
    return result, calls


def _reference_front(
    problem: NBAProblem,
) -> Tuple[List[EvaluationResult], float, int]:
    """Build the exact MILP reference front and return runtime/evaluation count."""
    solver = HiGHSMILPSolver(problem)
    original_evaluate = problem.evaluate
    calls = 0

    def counted_evaluate(x: np.ndarray) -> EvaluationResult:
        nonlocal calls
        calls += 1
        return original_evaluate(x)

    problem.evaluate = counted_evaluate  # type: ignore[method-assign]
    start = time.perf_counter()
    try:
        front = solver.solve_epsilon_constraint_front()
    finally:
        problem.evaluate = original_evaluate  # type: ignore[method-assign]
    return front, time.perf_counter() - start, calls


def _objective_matrix(solutions: List[EvaluationResult]) -> np.ndarray:
    if not solutions:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray([solution.objectives_min for solution in solutions])


def _normalization_points(reference_front: List[EvaluationResult]) -> Tuple[np.ndarray, np.ndarray]:
    reference = _objective_matrix(reference_front)
    if len(reference) == 0:
        raise RuntimeError("MILP reference front is empty; cannot normalize metrics")
    return np.min(reference, axis=0), np.max(reference, axis=0)


def _matches_or_dominates(
    method_front: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """Return the percentage of reference plans weakly dominated by the method."""
    if len(reference_front) == 0:
        return 0.0
    if len(method_front) == 0:
        return 0.0
    matched = sum(
        any(np.all(method_point <= reference_point + 1e-9)
            for method_point in method_front)
        for reference_point in reference_front
    )
    return 100.0 * matched / len(reference_front)


def _metrics(
    result: OptimizationResult,
    reference_front: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> Dict[str, float]:
    feasible_population = [
        solution for solution in result.final_population if solution.is_feasible
    ]
    feasible_front = [
        solution for solution in result.pareto_solutions if solution.is_feasible
    ]
    front_matrix = _objective_matrix(feasible_front)
    return {
        "runtime_sec": float(result.elapsed_time),
        "evaluations_reported": float(result.total_evaluations),
        "evaluations_used": float("nan"),
        "feasibility_rate_pct": (
            100.0 * len(feasible_population) / len(result.final_population)
            if result.final_population else 0.0
        ),
        "feasible_nondominated_plans": float(len(feasible_front)),
        "hypervolume": compute_hypervolume(
            front_matrix,
            ideal_point=ideal,
            nadir_point=nadir,
            ref_point=np.ones(4) * 1.1,
            normalize=True,
        ),
        "igd": compute_igd(front_matrix, reference_front) if len(front_matrix) else float("inf"),
        "milp_front_match_or_dominate_pct": _matches_or_dominates(
            front_matrix, reference_front
        ),
    }


def _factories(
    problem: NBAProblem,
    seed: int,
) -> List[Tuple[str, Callable[[], OptimizationResult]]]:
    params = problem.config.get("algorithm_parameters", {})
    population_size = int(params.get("population_size", 100))
    max_generations = int(params.get("max_generations", 120))
    common = {
        "crossover_prob": float(params.get("crossover_probability", 0.9)),
        "mutation_prob": float(params.get("mutation_probability", 0.15)),
    }
    evolutionary = [
        (
            "NSGA-II",
            lambda: NSGA2Optimizer(
                problem, pop_size=population_size, max_generations=max_generations,
                seed=seed, **common
            ).run(),
        ),
        (
            "SPEA2",
            lambda: SPEA2Optimizer(
                problem, pop_size=population_size,
                archive_size=int(params.get("spea2_archive_size", population_size)),
                max_generations=max_generations, seed=seed, **common
            ).run(),
        ),
        (
            "MOEA/D",
            lambda: MOEADOptimizer(
                problem, pop_size=population_size,
                n_neighbors=int(params.get("moead_neighbors", 15)),
                max_generations=max_generations, seed=seed, **common
            ).run(),
        ),
        (
            "Hybrid-Memetic",
            lambda: MemeticNSGA2Optimizer(
                problem, pop_size=population_size, max_generations=max_generations,
                warm_start_milp=True, seed=seed, **common
            ).run(),
        ),
        (
            "Hybrid-Memetic (no MILP seed)",
            lambda: MemeticNSGA2Optimizer(
                problem, pop_size=population_size, max_generations=max_generations,
                warm_start_milp=False, seed=seed, **common
            ).run(),
        ),
    ]
    configured_evaluations = population_size * (max_generations + 1)
    return [
        (
            "Random-Baseline",
            lambda: RandomSearchBaseline(problem, seed=seed).run(
                n_samples=configured_evaluations
            ),
        ),
        (
            "Greedy-Baseline",
            lambda: GreedyBaseline(problem, seed=seed).run(
                n_samples=configured_evaluations
            ),
        ),
    ] + evolutionary


def run_benchmark(
    instance_names: List[str],
    seeds: List[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for instance_name in instance_names:
        problem = NBAProblem(INSTANCE_CONFIGS[instance_name])
        reference_solutions, reference_runtime, reference_evaluations = _reference_front(problem)
        reference_matrix = _objective_matrix(reference_solutions)
        ideal, nadir = _normalization_points(reference_solutions)
        params = problem.config.get("algorithm_parameters", {})
        configured_budget = int(params.get("population_size", 100)) * (
            int(params.get("max_generations", 120)) + 1
        )
        print(
            f"\nInstance={instance_name}; configured evolutionary budget="
            f"{configured_budget}; MILP seed enabled: yes for Hybrid-Memetic, "
            "no for Hybrid-Memetic (no MILP seed)"
        )
        print(
            f"MILP reference: {len(reference_solutions)} plans, "
            f"{reference_evaluations} objective evaluations, "
            f"{reference_runtime:.3f}s; ideal={ideal.tolist()}; nadir={nadir.tolist()}"
        )

        for seed in seeds:
            for algorithm, factory in _factories(problem, seed):
                result, evaluations_used = _counted_run(problem, factory)
                metrics = _metrics(result, reference_matrix, ideal, nadir)
                metrics["evaluations_used"] = float(evaluations_used)
                rows.append({
                    "instance": instance_name,
                    "algorithm": algorithm,
                    "seed": seed,
                    "configured_budget": configured_budget,
                    **metrics,
                })

            rows.append({
                "instance": instance_name,
                "algorithm": "MILP-HiGHS (reference)",
                "seed": seed,
                "configured_budget": configured_budget,
                "runtime_sec": reference_runtime,
                "evaluations_reported": float(len(reference_solutions)),
                "evaluations_used": float(reference_evaluations),
                "feasibility_rate_pct": 100.0,
                "feasible_nondominated_plans": float(len(reference_solutions)),
                "hypervolume": compute_hypervolume(
                    reference_matrix, ideal_point=ideal, nadir_point=nadir,
                    ref_point=np.ones(4) * 1.1, normalize=True
                ),
                "igd": compute_igd(reference_matrix, reference_matrix),
                "milp_front_match_or_dominate_pct": 100.0,
            })

    results = pd.DataFrame(rows)
    output_path = ROOT / "results" / "benchmark_baseline_v2.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    summary = (
        results.groupby(["instance", "algorithm"], as_index=False)
        .agg(
            runtime_mean_sec=("runtime_sec", "mean"),
            runtime_std_sec=("runtime_sec", "std"),
            evaluations_used_mean=("evaluations_used", "mean"),
            evaluations_used_std=("evaluations_used", "std"),
            feasibility_rate_mean_pct=("feasibility_rate_pct", "mean"),
            feasible_nondominated_mean=("feasible_nondominated_plans", "mean"),
            hypervolume_mean=("hypervolume", "mean"),
            hypervolume_std=("hypervolume", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            milp_match_or_dominate_mean_pct=("milp_front_match_or_dominate_pct", "mean"),
        )
    )
    print("\nSUMMARY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nSaved results to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help="Run seeds 0 through N-1 (default: 10)"
    )
    parser.add_argument(
        "--instance", choices=("small", "medium", "both"), default="both",
        help="Instance to benchmark (default: both)"
    )
    args = parser.parse_args()
    instance_names = ["small", "medium"] if args.instance == "both" else [args.instance]
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    run_benchmark(instance_names, list(range(args.seeds)))


if __name__ == "__main__":
    main()
