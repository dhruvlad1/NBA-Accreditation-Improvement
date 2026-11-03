"""Configured baseline benchmark with corrected accounting and feasible metrics.

Compared with ``benchmark.py``, this version uses the configured evolutionary
run, counts actual objective evaluations, evaluates only feasible plans for
quality metrics, normalizes against a fixed MILP reference front, and supports
selectable instances and seed counts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

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
    algorithm: str,
    seed: int,
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
        result = _run_algorithm(problem, algorithm, seed)
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


def _run_algorithm(
    problem: NBAProblem,
    algorithm: str,
    seed: int,
) -> OptimizationResult:
    params = problem.config.get("algorithm_parameters", {})
    population_size = int(params.get("population_size", 100))
    max_generations = int(params.get("max_generations", 120))
    common = {
        "crossover_prob": float(params.get("crossover_probability", 0.9)),
        "mutation_prob": float(params.get("mutation_probability", 0.15)),
    }
    if algorithm == "Random-Baseline":
        return RandomSearchBaseline(problem, seed=seed).run(
            n_samples=population_size * (max_generations + 1)
        )
    if algorithm == "Greedy-Baseline":
        return GreedyBaseline(problem, seed=seed).run(
            n_samples=population_size * (max_generations + 1)
        )
    if algorithm == "NSGA-II":
        return NSGA2Optimizer(
            problem, pop_size=population_size, max_generations=max_generations,
            seed=seed, **common
        ).run()
    if algorithm == "SPEA2":
        return SPEA2Optimizer(
            problem, pop_size=population_size,
            archive_size=int(params.get("spea2_archive_size", population_size)),
            max_generations=max_generations, seed=seed, **common
        ).run()
    if algorithm == "MOEA/D":
        return MOEADOptimizer(
            problem, pop_size=population_size,
            n_neighbors=int(params.get("moead_neighbors", 15)),
            max_generations=max_generations, seed=seed, **common
        ).run()
    if algorithm == "Hybrid-Memetic":
        return MemeticNSGA2Optimizer(
            problem, pop_size=population_size, max_generations=max_generations,
            warm_start_milp=True, seed=seed, **common
        ).run()
    if algorithm == "Hybrid-Memetic (no MILP seed)":
        return MemeticNSGA2Optimizer(
            problem, pop_size=population_size, max_generations=max_generations,
            warm_start_milp=False, seed=seed, **common
        ).run()
    raise ValueError(f"Unknown algorithm: {algorithm}")


ALGORITHMS = (
    "Random-Baseline",
    "Greedy-Baseline",
    "NSGA-II",
    "SPEA2",
    "MOEA/D",
    "Hybrid-Memetic",
    "Hybrid-Memetic (no MILP seed)",
    "MILP-HiGHS (reference)",
)


def _run_algorithm_job(
    instance_name: str,
    algorithm: str,
    seed: int,
    reference_matrix: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    configured_budget: int,
) -> Dict[str, Any]:
    """Picklable process worker for one algorithm/seed pair."""
    problem = NBAProblem(INSTANCE_CONFIGS[instance_name])
    if algorithm == "MILP-HiGHS (reference)":
        start = time.perf_counter()
        reference_solutions, _, reference_evaluations = _reference_front(problem)
        runtime = time.perf_counter() - start
        return {
            "instance": instance_name,
            "algorithm": algorithm,
            "seed": seed,
            "configured_budget": configured_budget,
            "runtime_sec": runtime,
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
        }

    start = time.perf_counter()
    result, evaluations_used = _counted_run(problem, algorithm, seed)
    metrics = _metrics(result, reference_matrix, ideal, nadir)
    metrics["runtime_sec"] = time.perf_counter() - start
    metrics["evaluations_used"] = float(evaluations_used)
    return {
        "instance": instance_name,
        "algorithm": algorithm,
        "seed": seed,
        "configured_budget": configured_budget,
        **metrics,
    }


def run_benchmark(
    instance_names: List[str],
    seeds: List[int],
    jobs: int = 1,
    methods: List[str] | None = None,
    output_path: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    selected_methods = methods or list(ALGORITHMS)
    output_path = output_path or ROOT / "results" / "benchmark_baseline_v2.csv"
    runs_path = output_path.with_name(f"benchmark_runs_{output_path.stem}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: List[Dict[str, Any]] = []
    if resume and runs_path.exists():
        existing_rows = pd.read_csv(runs_path).to_dict("records")
    completed = {
        (str(row["instance"]), str(row["algorithm"]), int(row["seed"]))
        for row in existing_rows
    }
    rows = list(existing_rows)
    total = sum(
        1 for instance in instance_names for seed in seeds
        for algorithm in selected_methods
        if (instance, algorithm, seed) not in completed
    )
    finished = 0

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

        jobs_to_run = [
            (instance_name, algorithm, seed)
            for seed in seeds
            for algorithm in selected_methods
            if (instance_name, algorithm, seed) not in completed
        ]
        worker_args = {
            (instance_name, algorithm, seed): (
                instance_name, algorithm, seed, reference_matrix, ideal, nadir,
                configured_budget,
            )
            for instance_name, algorithm, seed in jobs_to_run
        }
        if jobs == 1:
            results_iterator = (
                _run_algorithm_job(*worker_args[key]) for key in worker_args
            )
            for row in results_iterator:
                _record_row(row, rows, runs_path, output_path)
                finished += 1
                print(
                    f"[{finished}/{total}] {row['instance']} {row['algorithm']} "
                    f"{row['seed']} {row['runtime_sec']:.3f}s"
                )
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [
                    executor.submit(_run_algorithm_job, *arguments)
                    for arguments in worker_args.values()
                ]
                for future in as_completed(futures):
                    row = future.result()
                    _record_row(row, rows, runs_path, output_path)
                    finished += 1
                    print(
                        f"[{finished}/{total}] {row['instance']} {row['algorithm']} "
                        f"{row['seed']} {row['runtime_sec']:.3f}s"
                    )

    results = pd.DataFrame(rows)
    results.to_csv(runs_path, index=False)
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
    print(f"Saved per-run results to {runs_path}")
    return results


def _record_row(
    row: Dict[str, Any],
    rows: List[Dict[str, Any]],
    runs_path: Path,
    output_path: Path,
) -> None:
    """Append one completed run and persist both raw and summary CSVs."""
    rows.append(row)
    pd.DataFrame(rows).to_csv(runs_path, index=False)
    summary = (
        pd.DataFrame(rows)
        .groupby(["instance", "algorithm"], as_index=False)
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
    summary.to_csv(output_path, index=False)


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
    parser.add_argument("--jobs", type=int, default=1, help="Process workers (default: 1)")
    parser.add_argument(
        "--methods",
        default=",".join(ALGORITHMS),
        help="Comma-separated algorithm names (default: all)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows in the per-run CSV")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "benchmark_baseline_v2.csv",
        help="Summary output CSV path",
    )
    args = parser.parse_args()
    instance_names = ["small", "medium"] if args.instance == "both" else [args.instance]
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(methods) - set(ALGORITHMS))
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}")
    run_benchmark(
        instance_names,
        list(range(args.seeds)),
        jobs=args.jobs,
        methods=methods,
        output_path=args.output,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
