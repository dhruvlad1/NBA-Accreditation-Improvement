"""Reproducible baseline benchmark for the NBA optimization algorithms.

This runner intentionally does not generate data or plots.  The stochastic
algorithms use the same 100-plan evaluation budget so that later algorithm
changes can be compared against a stable baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.algorithms.moead import MOEADOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer, OptimizationResult
from src.algorithms.spea2 import SPEA2Optimizer
from src.baseline.greedy import GreedyBaseline
from src.baseline.random_search import RandomSearchBaseline
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.metrics import compute_hypervolume
from src.problem import NBAProblem

try:
    from src.baseline.milp_solver import HiGHSMILPSolver
except ImportError:
    HiGHSMILPSolver = None  # type: ignore[assignment,misc]


ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = [42, 101, 2024, 777, 999]
EVALUATION_BUDGET = 100
INSTANCE_CONFIGS = {
    "small": ROOT / "experiments" / "config_small.yaml",
    "medium": ROOT / "experiments" / "config.yaml",
}


def _algorithm_factories(
    problem: NBAProblem,
    params: Dict[str, Any],
    seed: int,
) -> List[Tuple[str, Callable[[], OptimizationResult]]]:
    """Create algorithms configured to consume the common plan budget."""
    common = {
        "crossover_prob": params.get("crossover_probability", 0.9),
        "mutation_prob": params.get("mutation_probability", 0.15),
    }
    return [
        (
            "Random-Baseline",
            lambda: RandomSearchBaseline(problem, seed=seed).run(
                n_samples=EVALUATION_BUDGET
            ),
        ),
        (
            "Greedy-Baseline",
            lambda: GreedyBaseline(problem, seed=seed).run(
                n_samples=EVALUATION_BUDGET
            ),
        ),
        (
            "NSGA-II",
            lambda: NSGA2Optimizer(
                problem,
                pop_size=EVALUATION_BUDGET,
                max_generations=0,
                seed=seed,
                **common,
            ).run(),
        ),
        (
            "SPEA2",
            lambda: SPEA2Optimizer(
                problem,
                pop_size=EVALUATION_BUDGET // 2,
                archive_size=EVALUATION_BUDGET // 2,
                max_generations=0,
                seed=seed,
                **common,
            ).run(),
        ),
        (
            "MOEA/D",
            lambda: MOEADOptimizer(
                problem,
                pop_size=EVALUATION_BUDGET,
                max_generations=0,
                n_neighbors=min(
                    params.get("moead_neighbors", 15), EVALUATION_BUDGET
                ),
                seed=seed,
                **common,
            ).run(),
        ),
        (
            "Hybrid-Memetic",
            lambda: MemeticNSGA2Optimizer(
                problem,
                pop_size=EVALUATION_BUDGET,
                max_generations=0,
                seed=seed,
                # No generation means local search is not entered, while the
                # optimizer's MILP warm-start behavior remains represented.
                **common,
            ).run(),
        ),
    ]


def _metrics(result: OptimizationResult) -> Dict[str, float]:
    """Compute the requested metrics from a standard optimizer result."""
    pareto = result.pareto_solutions
    objectives = np.asarray([solution.objectives_min for solution in pareto])
    feasible_pareto = [solution for solution in pareto if solution.is_feasible]
    feasible_population = sum(
        solution.is_feasible for solution in result.final_population
    )
    population_size = len(result.final_population)
    return {
        "runtime_sec": float(result.elapsed_time),
        "hypervolume": compute_hypervolume(objectives, normalize=True),
        "feasible_nondominated_plans": float(len(feasible_pareto)),
        "feasibility_rate_pct": (
            100.0 * feasible_population / population_size
            if population_size
            else 0.0
        ),
        "evaluations": float(result.total_evaluations),
    }


def _run_milp(problem: NBAProblem, seed: int) -> OptimizationResult:
    """Run the optional exact reference front and adapt it to the result API."""
    if HiGHSMILPSolver is None:
        raise RuntimeError("SciPy HiGHS is unavailable")

    start = time.perf_counter()
    solutions = HiGHSMILPSolver(problem).solve_epsilon_constraint_front(
        n_budget_steps=3, n_hour_steps=3
    )
    elapsed = time.perf_counter() - start
    return OptimizationResult(
        algorithm_name="MILP-HiGHS",
        seed=seed,
        elapsed_time=elapsed,
        final_population=solutions,
        pareto_solutions=solutions,
        history=[],
        total_evaluations=len(solutions),
    )


def run_benchmark(
    configs: Dict[str, Path] = INSTANCE_CONFIGS,
    seeds: List[int] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    """Run all requested instances, algorithms, and seeds."""
    rows: List[Dict[str, Any]] = []
    for instance, config_path in configs.items():
        problem = NBAProblem(config_path)
        params = problem.config.get("algorithm_parameters", {})
        for seed in seeds:
            for algorithm, factory in _algorithm_factories(problem, params, seed):
                result = factory()
                row = {
                    "instance": instance,
                    "algorithm": algorithm,
                    "seed": seed,
                    "evaluation_budget": EVALUATION_BUDGET,
                    **_metrics(result),
                }
                rows.append(row)

            if HiGHSMILPSolver is not None:
                milp_result = _run_milp(problem, seed)
                rows.append(
                    {
                        "instance": instance,
                        "algorithm": "MILP-HiGHS",
                        "seed": seed,
                        "evaluation_budget": EVALUATION_BUDGET,
                        **_metrics(milp_result),
                    }
                )

    results = pd.DataFrame(rows)
    output_path = ROOT / "results" / "benchmark_baseline.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary = (
        results.groupby(["instance", "algorithm"], as_index=False)
        .agg(
            runtime_mean_sec=("runtime_sec", "mean"),
            runtime_std_sec=("runtime_sec", "std"),
            hypervolume_mean=("hypervolume", "mean"),
            hypervolume_std=("hypervolume", "std"),
            feasible_nondominated_mean=("feasible_nondominated_plans", "mean"),
            feasible_nondominated_std=("feasible_nondominated_plans", "std"),
            feasibility_rate_mean_pct=("feasibility_rate_pct", "mean"),
            feasibility_rate_std_pct=("feasibility_rate_pct", "std"),
        )
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved baseline results to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds (default: 42 101 2024 777 999)",
    )
    args = parser.parse_args()
    run_benchmark(seeds=args.seeds)


if __name__ == "__main__":
    main()
