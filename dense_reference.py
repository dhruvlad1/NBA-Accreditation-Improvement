"""Build and compare a denser MILP reference front on the medium instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, milp

from src.algorithms.nsga2 import NSGA2Optimizer, OptimizationResult
from src.baseline.milp_solver import HiGHSMILPSolver
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.metrics import compute_hypervolume
from src.problem import EvaluationResult, NBAProblem


ROOT = Path(__file__).resolve().parent
MEDIUM_CONFIG = ROOT / "experiments" / "config.yaml"
OUTPUT_PATH = ROOT / "results" / "milp_dense_front.json"

OLD_IDEAL = np.array([-791.0, 12.1, 1370.0, -0.67])
OLD_NADIR = np.array([-750.0, 30.0, 2256.0, -0.5067])
REFERENCE_POINT = np.ones(4) * 1.1
OBJECTIVE_NAMES = ("f1 score", "f2 cost", "f3 hours", "f4 ratio")


def _objective_matrix(solutions: Iterable[EvaluationResult]) -> np.ndarray:
    return np.asarray([solution.objectives_min for solution in solutions], dtype=float)


def _strictly_dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left <= right + 1e-9) and np.any(left < right - 1e-9))


def _nondominated(solutions: list[EvaluationResult]) -> list[EvaluationResult]:
    result: list[EvaluationResult] = []
    for index, candidate in enumerate(solutions):
        candidate_obj = candidate.objectives_min
        if any(
            _strictly_dominates(other.objectives_min, candidate_obj)
            for other_index, other in enumerate(solutions)
            if other_index != index
        ):
            continue
        if any(
            np.allclose(candidate.x, other.x) or
            np.allclose(candidate.objectives_min, other.objectives_min, atol=1e-8)
            for other in result
        ):
            continue
        result.append(candidate)
    return result


def _solve_f1(
    solver: HiGHSMILPSolver,
    max_budget: float,
    max_hours: float,
    min_ratio: float,
    time_limit: float,
) -> EvaluationResult | None:
    """Solve max-f1 with all three epsilon constraints."""
    objective = np.zeros(solver.total_vars, dtype=float)
    objective[solver.N : solver.N + solver.C] = -1.0
    constraints = solver._assemble_constraints(  # type: ignore[attr-defined]
        max_budget=max_budget,
        max_total_hours=max_hours,
        min_worst_ratio=min_ratio,
    )
    result = milp(
        c=objective,
        integrality=solver.integrality,
        bounds=Bounds(solver.lb, solver.ub),
        constraints=constraints,
        options={"disp": False, "presolve": True, "time_limit": time_limit},
    )
    if not result.success or result.x is None:
        return None
    solution = solver.problem.evaluate(np.rint(result.x[: solver.N]).astype(np.int32))
    return solution if solution.is_feasible else None


def build_dense_front(
    problem: NBAProblem,
    grid_f2: int,
    grid_f3: int,
    grid_f4: int,
    max_solves: int | None,
    time_limit: float,
) -> tuple[list[EvaluationResult], int, int, float]:
    solver = HiGHSMILPSolver(problem)
    f2_grid = np.linspace(OLD_IDEAL[1], OLD_NADIR[1], grid_f2)
    f3_grid = np.linspace(OLD_IDEAL[2], OLD_NADIR[2], grid_f3)
    f4_grid = np.linspace(-OLD_NADIR[3], -OLD_IDEAL[3], grid_f4)
    total_requested = grid_f2 * grid_f3 * grid_f4
    solve_limit = total_requested if max_solves is None else min(max_solves, total_requested)

    solutions: list[EvaluationResult] = []
    solves = feasible = 0
    started = time.perf_counter()
    for f2 in f2_grid:
        for f3 in f3_grid:
            for f4 in f4_grid:
                if solves >= solve_limit:
                    break
                solves += 1
                solution = _solve_f1(solver, float(f2), float(f3), float(f4), time_limit)
                if solution is not None:
                    feasible += 1
                    solutions.append(solution)
            if solves >= solve_limit:
                break
        if solves >= solve_limit:
            break
    front = _nondominated(solutions)
    return front, solves, feasible, time.perf_counter() - started


def _run_algorithms(problem: NBAProblem) -> dict[str, OptimizationResult]:
    params = problem.config["algorithm_parameters"]
    kwargs: dict[str, Any] = {
        "pop_size": int(params["population_size"]),
        "max_generations": int(params["max_generations"]),
        "crossover_prob": float(params["crossover_probability"]),
        "mutation_prob": float(params["mutation_probability"]),
        "seed": 0,
    }
    return {
        "NSGA-II": NSGA2Optimizer(problem, **kwargs).run(),
        "Hybrid-Memetic (no MILP seed)": MemeticNSGA2Optimizer(
            problem, warm_start_milp=False, **kwargs
        ).run(),
    }


def _front_solutions(result: OptimizationResult) -> list[EvaluationResult]:
    return [solution for solution in result.pareto_solutions if solution.is_feasible]


def _print_ranges(name: str, solutions: list[EvaluationResult], dense: np.ndarray) -> None:
    values = np.asarray([solution.objectives for solution in solutions], dtype=float)
    print(f"\n{name} objective ranges (actual objective directions):")
    for index, label in enumerate(OBJECTIVE_NAMES):
        method_min, method_max = np.min(values[:, index]), np.max(values[:, index])
        dense_min, dense_max = np.min(dense[:, index]), np.max(dense[:, index])
        print(
            f"  {label}: method [{method_min:g}, {method_max:g}] "
            f"vs dense [{dense_min:g}, {dense_max:g}]"
        )


def _coverage_counts(
    method: list[EvaluationResult], dense: list[EvaluationResult]
) -> tuple[int, int]:
    method_matrix = _objective_matrix(method)
    dense_matrix = _objective_matrix(dense)
    method_dominates_dense = sum(
        any(_strictly_dominates(method_obj, dense_obj) for dense_obj in dense_matrix)
        for method_obj in method_matrix
    )
    dense_matched_or_dominated = sum(
        any(np.all(method_obj <= dense_obj + 1e-9) for method_obj in method_matrix)
        for dense_obj in dense_matrix
    )
    return method_dominates_dense, dense_matched_or_dominated


def _unreached_endpoints(
    method: list[EvaluationResult], dense: np.ndarray
) -> list[str]:
    method_values = np.asarray([solution.objectives for solution in method], dtype=float)
    dense_min = np.min(dense, axis=0)
    dense_max = np.max(dense, axis=0)
    missing: list[str] = []
    for index, label in enumerate(OBJECTIVE_NAMES):
        method_min = np.min(method_values[:, index])
        method_max = np.max(method_values[:, index])
        if method_min > dense_min[index] + 1e-9:
            missing.append(f"{label} lower endpoint ({dense_min[index]:g})")
        if method_max < dense_max[index] - 1e-9:
            missing.append(f"{label} upper endpoint ({dense_max[index]:g})")
    return missing


def _serialize(front: list[EvaluationResult], metadata: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "plans": [
            {
                "objectives": solution.objectives.tolist(),
                "objectives_min": solution.objectives_min.tolist(),
                "decision_vector": solution.x.tolist(),
            }
            for solution in front
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-f2", type=int, default=6)
    parser.add_argument("--grid-f3", type=int, default=5)
    parser.add_argument("--grid-f4", type=int, default=6)
    parser.add_argument("--max-solves", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=30.0)
    args = parser.parse_args()
    if min(args.grid_f2, args.grid_f3, args.grid_f4) < 1:
        parser.error("grid sizes must be at least 1")
    if args.max_solves is not None and args.max_solves < 1:
        parser.error("--max-solves must be at least 1")
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")

    problem = NBAProblem(MEDIUM_CONFIG)
    dense_front, solves, feasible, build_time = build_dense_front(
        problem, args.grid_f2, args.grid_f3, args.grid_f4, args.max_solves, args.time_limit
    )
    if not dense_front:
        raise RuntimeError("The MILP sweep produced no feasible plans")
    dense_min = np.min(np.asarray([solution.objectives for solution in dense_front]), axis=0)
    dense_max = np.max(np.asarray([solution.objectives for solution in dense_front]), axis=0)
    _serialize(
        dense_front,
        {
            "instance": "medium",
            "grid": {"f2": args.grid_f2, "f3": args.grid_f3, "f4": args.grid_f4},
            "solves": solves,
            "feasible_solves": feasible,
            "time_limit_sec": args.time_limit,
            "ideal": np.min(_objective_matrix(dense_front), axis=0).tolist(),
            "nadir": np.max(_objective_matrix(dense_front), axis=0).tolist(),
        },
    )

    print(f"Dense MILP sweep: {solves} solves, {feasible} feasible solves, "
          f"{len(dense_front)} final-front plans, {build_time:.3f}s")
    print(f"Dense ideal={np.min(_objective_matrix(dense_front), axis=0).tolist()}")
    print(f"Dense nadir={np.max(_objective_matrix(dense_front), axis=0).tolist()}")
    print(f"Old ideal={OLD_IDEAL.tolist()}; old nadir={OLD_NADIR.tolist()}")
    print(f"Saved dense front to {OUTPUT_PATH}")

    results = _run_algorithms(problem)
    dense_actual = np.asarray([solution.objectives for solution in dense_front])
    dense_matrix = _objective_matrix(dense_front)
    missed_ranges: dict[str, list[str]] = {}
    print("\nCross-check against dense front:")
    for name, result in results.items():
        solutions = _front_solutions(result)
        _print_ranges(name, solutions, dense_actual)
        dominates, covered = _coverage_counts(solutions, dense_front)
        missed_ranges[name] = _unreached_endpoints(solutions, dense_actual)
        print(f"  evaluations={result.total_evaluations}; feasible front plans={len(solutions)}")
        print(f"  method plans that dominate at least one dense plan: {dominates} "
              "(MILP reference is not exact)")
        print(f"  dense plans matched or dominated by method: {covered}/{len(dense_front)}")

    print("\nHypervolume (benchmark_v2 normalization and reference point):")
    for name, matrix in [
        ("Dense MILP", dense_matrix),
        *[(name, _objective_matrix(_front_solutions(result))) for name, result in results.items()],
    ]:
        value = compute_hypervolume(
            matrix, ideal_point=OLD_IDEAL, nadir_point=OLD_NADIR,
            ref_point=REFERENCE_POINT, normalize=True,
        )
        print(f"  {name}: {value:.6f}")

    heuristic_hv = [
        compute_hypervolume(
            _objective_matrix(_front_solutions(result)),
            ideal_point=OLD_IDEAL, nadir_point=OLD_NADIR,
            ref_point=REFERENCE_POINT, normalize=True,
        )
        for result in results.values()
    ]
    valid = all(compute_hypervolume(
        dense_matrix, ideal_point=OLD_IDEAL, nadir_point=OLD_NADIR,
        ref_point=REFERENCE_POINT, normalize=True
    ) >= value for value in heuristic_hv)
    print("\nSummary:")
    print(f"  Dense reference is a valid yardstick by HV>=heuristics: {'yes' if valid else 'no'}")
    for name, missing in missed_ranges.items():
        print(f"  {name} misses: {', '.join(missing) if missing else 'none'}")


if __name__ == "__main__":
    main()
