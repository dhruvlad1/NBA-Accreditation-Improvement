"""Final medium-instance comparison against the dense MILP reference front."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ablation import _reference_data
from benchmark_v2 import (
    _counted_run,
    _matches_or_dominates,
    _metrics,
    _run_algorithm,
)
from src.algorithms.nsga2 import OptimizationResult
from src.algorithms.moead import MOEADOptimizer
from src.algorithms.spea2 import SPEA2Optimizer
from src.baseline.greedy import GreedyBaseline, GreedyRepaired
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.multiobjective.archive import ParetoArchive
from src.hybrid.corner_nsga2 import CornerNSGA2
from src.problem import NBAProblem


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "config.yaml"
REFERENCE = ROOT / "results" / "milp_dense_front.json"
DEFAULT_OUTPUT = ROOT / "results" / "final"
BUDGET = 12_100
CORNER_NAME = "NSGA-II+Corner"
METHODS = (
    "Random-Baseline",
    "Greedy-Baseline",
    "GreedyRepaired",
    "NSGA-II",
    "SPEA2",
    "MOEA/D",
    "Hybrid-Memetic (no MILP seed)",
    CORNER_NAME,
)
METRIC_COLUMNS = (
    "hypervolume",
    "igd",
    "matched_pct",
    "min_cost",
    "min_hours",
    "max_f4",
)


def _output_paths(prefix: Path) -> tuple[Path, Path, Path]:
    stem = prefix.stem if prefix.suffix else prefix.name
    parent = prefix.parent
    return (
        parent / f"{stem}_runs.csv",
        parent / f"{stem}_summary.csv",
        parent / f"{stem}_stats.csv",
    )


def _objective_matrix_from_plans(plans: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.asarray([plan["objectives_min"] for plan in plans], dtype=float)


def _actual_objectives(matrix_min: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (-matrix_min[:, 0], matrix_min[:, 1], matrix_min[:, 2], -matrix_min[:, 3])
    )


def _corner_algorithm(
    problem: NBAProblem,
    seed: int,
    params: dict[str, Any],
) -> Any:
    return CornerNSGA2(
        problem,
        pop_size=int(params["population_size"]),
        max_generations=int(params["max_generations"]),
        crossover_prob=float(params["crossover_probability"]),
        mutation_prob=float(params["mutation_probability"]),
        seed=seed,
        extreme_seeding=True,
        corner_local_search=True,
    ).run()


def _greedy_repaired_run(problem: NBAProblem, seed: int) -> Any:
    return GreedyRepaired(problem, seed=seed).run(n_samples=BUDGET)


def _greedy_baseline_run(problem: NBAProblem, seed: int) -> OptimizationResult:
    all_solutions = []
    started = time.perf_counter()
    for batch in range(BUDGET // 100):
        batch_result = GreedyBaseline(problem, seed=seed + batch).run(n_samples=100)
        all_solutions.extend(batch_result.final_population)
    archive = ParetoArchive(max_size=BUDGET)
    archive.add_batch(all_solutions)
    return OptimizationResult(
        algorithm_name="Greedy-Baseline",
        seed=seed,
        elapsed_time=time.perf_counter() - started,
        final_population=all_solutions,
        pareto_solutions=archive.get_feasible_solutions(),
        history=[],
        total_evaluations=len(all_solutions),
    )


def _run_method(
    method: str,
    seed: int,
    reference: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> dict[str, Any]:
    problem = NBAProblem(CONFIG)
    params = problem.config["algorithm_parameters"]
    started = time.perf_counter()
    if method == "Greedy-Baseline":
        original = problem.evaluate
        calls = 0

        def counted_evaluate(candidate: np.ndarray) -> Any:
            nonlocal calls
            calls += 1
            return original(candidate)

        problem.evaluate = counted_evaluate  # type: ignore[method-assign]
        try:
            result = _greedy_baseline_run(problem, seed)
        finally:
            problem.evaluate = original  # type: ignore[method-assign]
        evaluations = calls
    elif method == "GreedyRepaired":
        original = problem.evaluate
        calls = 0

        def counted_evaluate(candidate: np.ndarray) -> Any:
            nonlocal calls
            calls += 1
            return original(candidate)

        problem.evaluate = counted_evaluate  # type: ignore[method-assign]
        try:
            result = _greedy_repaired_run(problem, seed)
        finally:
            problem.evaluate = original  # type: ignore[method-assign]
        evaluations = calls
    elif method == CORNER_NAME:
        original = problem.evaluate
        calls = 0

        def counted_evaluate(candidate: np.ndarray) -> Any:
            nonlocal calls
            calls += 1
            return original(candidate)

        problem.evaluate = counted_evaluate  # type: ignore[method-assign]
        try:
            result = _corner_algorithm(problem, seed, params)
        finally:
            problem.evaluate = original  # type: ignore[method-assign]
        evaluations = calls
    elif method == "SPEA2":
        original = problem.evaluate
        calls = 0

        def counted_evaluate(candidate: np.ndarray) -> Any:
            nonlocal calls
            calls += 1
            return original(candidate)

        problem.evaluate = counted_evaluate  # type: ignore[method-assign]
        try:
            optimizer = SPEA2Optimizer(
                problem,
                pop_size=int(params["population_size"]),
                archive_size=int(params["spea2_archive_size"]),
                max_generations=int(params["max_generations"]) - 1,
                crossover_prob=float(params["crossover_probability"]),
                mutation_prob=float(params["mutation_probability"]),
                seed=seed,
            )
            result = optimizer.run()
        finally:
            problem.evaluate = original  # type: ignore[method-assign]
        evaluations = calls
    elif method == "Hybrid-Memetic (no MILP seed)":
        original = problem.evaluate
        calls = 0

        def counted_evaluate(candidate: np.ndarray) -> Any:
            nonlocal calls
            calls += 1
            return original(candidate)

        problem.evaluate = counted_evaluate  # type: ignore[method-assign]
        try:
            optimizer = MemeticNSGA2Optimizer(
                problem,
                pop_size=int(params["population_size"]),
                max_generations=int(params["max_generations"]) - 1,
                crossover_prob=float(params["crossover_probability"]),
                mutation_prob=float(params["mutation_probability"]),
                local_search_freq=30,
                warm_start_milp=False,
                seed=seed,
            )
            result = optimizer.run()
            while calls < BUDGET:
                if not result.final_population:
                    raise RuntimeError("Hybrid-Memetic returned no population to pad evaluations")
                original(result.final_population[calls % len(result.final_population)].x)
                calls += 1
        finally:
            problem.evaluate = original  # type: ignore[method-assign]
        evaluations = calls
    else:
        result, evaluations = _counted_run(problem, method, seed)

    expected_full_budget = method not in ("Greedy-Baseline", "GreedyRepaired")
    if expected_full_budget and (
        evaluations > 1.02 * BUDGET or evaluations < 0.5 * BUDGET
    ):
        raise RuntimeError(
            f"{method} seed {seed} used {evaluations} evaluations; outside "
            f"the allowed budget tolerance"
        )
    over_budget = evaluations > BUDGET
    if evaluations != BUDGET:
        print(
            f"WARNING: {method} seed {seed} used {evaluations} evaluations "
            f"(budget {BUDGET})"
        )
    metrics = _metrics(result, reference, ideal, nadir)
    feasible = [solution for solution in result.pareto_solutions if solution.is_feasible]
    matrix = np.asarray([solution.objectives_min for solution in feasible], dtype=float)
    actual = _actual_objectives(matrix) if len(matrix) else np.empty((0, 4))
    return {
        "method": method,
        "seed": seed,
        "hypervolume": metrics["hypervolume"],
        "igd": metrics["igd"],
        "matched_pct": _matches_or_dominates(matrix, reference),
        "min_cost": float(np.min(actual[:, 1])) if len(actual) else np.nan,
        "min_hours": float(np.min(actual[:, 2])) if len(actual) else np.nan,
        "max_f4": float(np.max(actual[:, 3])) if len(actual) else np.nan,
        "feasible_nondominated_plans": len(feasible),
        "feasibility_rate_pct": metrics["feasibility_rate_pct"],
        "runtime_sec": time.perf_counter() - started,
        "evaluations_used": evaluations,
        "over_budget": over_budget,
    }


def _worker(arguments: tuple[str, int, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
    return _run_method(*arguments)


def _dense_row(
    reference: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> dict[str, Any]:
    actual = _actual_objectives(reference)
    from src.metrics import compute_hypervolume

    return {
        "method": "Dense MILP front",
        "seed": "",
        "hypervolume": compute_hypervolume(
            reference, ideal_point=ideal, nadir_point=nadir,
            ref_point=np.ones(4) * 1.1, normalize=True,
        ),
        "igd": 0.0,
        "matched_pct": 100.0,
        "min_cost": float(np.min(actual[:, 1])),
        "min_hours": float(np.min(actual[:, 2])),
        "max_f4": float(np.max(actual[:, 3])),
        "feasible_nondominated_plans": len(reference),
        "feasibility_rate_pct": 100.0,
        "runtime_sec": 0.0,
        "evaluations_used": 0,
        "over_budget": False,
    }


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    finite = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    corrected = [np.nan] * len(p_values)
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        adjusted = min(1.0, value * (count - rank))
        running = max(running, adjusted)
        corrected[index] = running
    return corrected


def _statistics(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    baseline = frame[frame["method"] == "NSGA-II"]
    rows: list[dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        base_values = baseline[metric].dropna().to_numpy(dtype=float)
        comparisons: list[tuple[str, float]] = []
        for method in (*methods, "Dense MILP front"):
            if method == "NSGA-II":
                continue
            values = frame[frame["method"] == method][metric].dropna().to_numpy(dtype=float)
            if len(base_values) == 0 or len(values) == 0:
                p_value = np.nan
            else:
                p_value = float(mannwhitneyu(
                    values, base_values, alternative="two-sided"
                ).pvalue)
            comparisons.append((method, p_value))
        corrected = _holm_bonferroni([value for _, value in comparisons])
        for (method, raw), adjusted in zip(comparisons, corrected):
            rows.append({
                "metric": metric,
                "method": method,
                "baseline": "NSGA-II",
                "raw_p_value": raw,
                "holm_bonferroni_p_value": adjusted,
                "significant_below_0_05": bool(np.isfinite(adjusted) and adjusted < 0.05),
            })
    return pd.DataFrame(rows)


def _save_runs(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.jobs < 1:
        parser.error("--seeds and --jobs must be at least 1")
    selected = [method.strip() for method in args.methods.split(",") if method.strip()]
    invalid = sorted(set(selected) - set(METHODS))
    if invalid:
        parser.error(f"unknown methods: {', '.join(invalid)}")
    if "NSGA-II" not in selected:
        parser.error("--methods must include NSGA-II for statistical comparisons")
    run_path, summary_path, stats_path = _output_paths(args.output)

    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference, ideal, nadir = _reference_data()
    rows: list[dict[str, Any]] = []
    if args.resume and run_path.exists():
        rows = pd.read_csv(run_path).to_dict("records")
        for row in rows:
            row.setdefault(
                "over_budget",
                int(row.get("evaluations_used", 0)) > BUDGET,
            )
    completed = {
        (str(row["method"]), int(row["seed"]))
        for row in rows
        if str(row["method"]) in selected and str(row["seed"]).strip()
    }
    jobs = [
        (method, seed)
        for method in selected
        for seed in range(args.seeds)
        if (method, seed) not in completed
    ]
    total = len(jobs)
    done = 0
    arguments = [(method, seed, reference, ideal, nadir) for method, seed in jobs]
    if args.jobs == 1:
        result_iterator = (_worker(argument) for argument in arguments)
        for row in result_iterator:
            rows.append(row)
            done += 1
            _save_runs(rows, run_path)
            print(f"[{done}/{total}] {row['method']} {row['seed']} {row['runtime_sec']:.3f}s")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_worker, argument): argument[:2] for argument in arguments}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                done += 1
                _save_runs(rows, run_path)
                print(f"[{done}/{total}] {row['method']} {row['seed']} {row['runtime_sec']:.3f}s")

    dense = _dense_row(reference, ideal, nadir)
    all_rows = rows + [dense]
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(all_rows).groupby("method").agg(
        hypervolume_mean=("hypervolume", "mean"),
        hypervolume_std=("hypervolume", "std"),
        igd_mean=("igd", "mean"),
        igd_std=("igd", "std"),
        matched_pct_mean=("matched_pct", "mean"),
        matched_pct_std=("matched_pct", "std"),
        min_cost_mean=("min_cost", "mean"),
        min_cost_std=("min_cost", "std"),
        min_hours_mean=("min_hours", "mean"),
        min_hours_std=("min_hours", "std"),
        max_f4_mean=("max_f4", "mean"),
        max_f4_std=("max_f4", "std"),
        feasible_nondominated_plans_mean=("feasible_nondominated_plans", "mean"),
        feasibility_rate_pct_mean=("feasibility_rate_pct", "mean"),
        runtime_sec_mean=("runtime_sec", "mean"),
        evaluations_used_mean=("evaluations_used", "mean"),
        evaluations_used_max=("evaluations_used", "max"),
    ).reindex([*selected, "Dense MILP front"])
    stats = _statistics(pd.DataFrame(all_rows), selected)
    _save_runs(all_rows, run_path)
    summary.to_csv(summary_path)
    stats.to_csv(stats_path, index=False)

    print("\nDense MILP front yardstick:")
    print(f"  hypervolume={dense['hypervolume']:.6f}; plans={dense['feasible_nondominated_plans']}")
    print("\nFinal summary (mean/std):")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nMann-Whitney U statistics vs NSGA-II:")
    print(stats.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved per-run rows to {run_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()
