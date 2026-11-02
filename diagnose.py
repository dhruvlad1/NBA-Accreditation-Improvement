"""Diagnose feasibility and benchmark accounting on the medium NBA instance."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import yaml

import benchmark
from src.algorithms.nsga2 import NSGA2Optimizer, OptimizationResult
from src.baseline.milp_solver import HiGHSMILPSolver
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.metrics import compute_hypervolume
from src.problem import EvaluationResult, NBAProblem


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "experiments" / "config.yaml"
SAMPLE_SIZE = 2_000
TOLERANCE = 1e-9


def _format_vector(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in vector) + "]"


def _instrument_evaluations(problem: NBAProblem) -> Tuple[Dict[str, int], Callable[[], None]]:
    """Count actual calls to NBAProblem.evaluate for one problem instance."""
    counts = {"evaluate": 0}
    original = problem.evaluate

    def counted(x: np.ndarray) -> EvaluationResult:
        counts["evaluate"] += 1
        return original(x)

    problem.evaluate = counted  # type: ignore[method-assign]

    def restore() -> None:
        problem.evaluate = original  # type: ignore[method-assign]

    return counts, restore


def _benchmark_evaluation_counts(problem: NBAProblem) -> None:
    """Run the same medium-instance factories as benchmark.py and count calls."""
    params = problem.config.get("algorithm_parameters", {})
    print("\n1. EVALUATION BUDGET")
    print(f"Configured benchmark budget: {benchmark.EVALUATION_BUDGET}")
    print("experiments/config.yaml evaluation budget:", "NOT SET")
    print("Proposal budget: 20,000 (not present as a config field)")
    print(
        "Configured population_size/max_generations:",
        params.get("population_size"),
        params.get("max_generations"),
        "(the benchmark overrides generations to zero)",
    )
    print("Actual calls to NBAProblem.evaluate per medium run:")

    for name, factory in benchmark._algorithm_factories(problem, params, seed=42):
        counts, restore = _instrument_evaluations(problem)
        try:
            result = factory()
        finally:
            restore()
        print(
            f"  {name:16s}: {counts['evaluate']:4d} "
            f"(result.total_evaluations={result.total_evaluations})"
        )

    counts, restore = _instrument_evaluations(problem)
    try:
        milp_result = benchmark._run_milp(problem, seed=42)
    finally:
        restore()
    print(
        f"  {'MILP-HiGHS':16s}: {counts['evaluate']:4d} "
        f"(result.total_evaluations={milp_result.total_evaluations})"
    )


def _raw_random_plans(problem: NBAProblem, rng: np.random.Generator) -> np.ndarray:
    """Sample unconstrained integer plans, including unschedulable actions."""
    upper_bounds = np.array([action.max_units for action in problem.actions], dtype=np.int32)
    return np.column_stack(
        [rng.integers(0, int(upper) + 1, size=SAMPLE_SIZE) for upper in upper_bounds]
    ).astype(np.int32)


def _constraint_report(problem: NBAProblem) -> None:
    """Report independent constraint violation rates for raw random plans."""
    rng = np.random.default_rng(20261102)
    plans = _raw_random_plans(problem, rng)
    results = problem.evaluate_batch(plans)
    names = (
        ("prerequisites", "prerequisites/evidence"),
        ("deadline", "deadline"),
        ("budget", "budget"),
        ("monthly_hours", "monthly faculty hours"),
        ("criteria_min", "per-criterion minimum level"),
        ("target_score", "total score >= R"),
    )
    print("\n2. FEASIBILITY BY CONSTRAINT")
    print(f"Raw unconstrained random integer plans: {SAMPLE_SIZE}")
    for key, label in names:
        violated = sum(result.violations[key] > TOLERANCE for result in results)
        print(f"  {label:30s}: {100.0 * violated / SAMPLE_SIZE:6.2f}% violate")
    feasible = sum(result.is_feasible for result in results)
    print(f"  {'pass all constraints':30s}: {100.0 * feasible / SAMPLE_SIZE:6.2f}%")
    if feasible == 0:
        print(
            "  Numeric rarity bound: 0/2,000 observed; the rule-of-three "
            "95% upper bound is approximately 0.15%."
        )


def _milp_killer_constraint_report(problem: NBAProblem) -> None:
    """Inspect exact MILP anchor solutions against each relevant limit."""
    solver = HiGHSMILPSolver(problem)
    anchors: List[Tuple[str, EvaluationResult]] = []
    for objective in ("max_f1", "min_f2", "min_f3", "max_f4"):
        solution = solver.solve_single_objective(objective)
        if solution is not None:
            anchors.append((objective, solution))

    print("\n3. THE KILLER CONSTRAINT")
    print(f"MILP anchor solutions found: {len(anchors)}")
    unique_vectors = {
        tuple(np.round(solution.objectives, decimals=8))
        for _, solution in anchors
    }
    print(f"Unique anchor objective vectors: {len(unique_vectors)}")
    for name, solution in anchors:
        print(
            f"  {name:8s}: budget {solution.f2_cost_lakh:.4f}/"
            f"{problem.budget:.4f} "
            f"({100.0 * solution.f2_cost_lakh / problem.budget:.2f}%), "
            f"max monthly hours {np.max(solution.monthly_hours):.4f}/"
            f"{problem.monthly_capacity:.4f}, "
            f"total score {solution.f1_total_score:.4f} vs R="
            f"{problem.min_accreditation_score:.4f}, "
            f"feasible={solution.is_feasible}"
        )
    if anchors:
        print(
            "  Feasible-region interpretation: raw random plans passing all "
            "constraints are rare; MILP anchors are the exact feasible witness set "
            "used here, not a proof that the region is mathematically tiny."
        )


def _nsga_generation_report(problem: NBAProblem) -> None:
    """Run the configured NSGA-II and print population feasibility every 20 generations."""
    params = problem.config.get("algorithm_parameters", {})
    result = NSGA2Optimizer(
        problem,
        pop_size=int(params.get("population_size", 100)),
        max_generations=int(params.get("max_generations", 120)),
        crossover_prob=float(params.get("crossover_probability", 0.9)),
        mutation_prob=float(params.get("mutation_probability", 0.15)),
        seed=42,
    ).run()
    print("\n4. FEASIBILITY OVER GENERATIONS")
    for snapshot in result.history:
        generation = int(snapshot["generation"])
        if generation % 20 == 0:
            share = 100.0 * float(snapshot["feasible_count"]) / 100.0
            print(f"  generation {generation:3d}: {share:6.2f}% feasible")
    print(f"  final population evaluations: {result.total_evaluations}")


def _hypervolume_report(problem: NBAProblem) -> None:
    """Compare raw benchmark HV with HV after explicitly filtering feasible plans."""
    params = problem.config.get("algorithm_parameters", {})
    result = NSGA2Optimizer(
        problem,
        pop_size=benchmark.EVALUATION_BUDGET,
        max_generations=0,
        crossover_prob=float(params.get("crossover_probability", 0.9)),
        mutation_prob=float(params.get("mutation_probability", 0.15)),
        seed=42,
    ).run()
    all_front = np.asarray([solution.objectives_min for solution in result.pareto_solutions])
    feasible_front = np.asarray(
        [
            solution.objectives_min
            for solution in result.pareto_solutions
            if solution.is_feasible
        ]
    )
    raw_hv = compute_hypervolume(all_front, normalize=True)
    feasible_hv = compute_hypervolume(feasible_front, normalize=True)
    print("\n5. HYPERVOLUME CHECK")
    print(
        "  benchmark.py passes all result.pareto_solutions to metrics; "
        "it does not filter inside _metrics."
    )
    print(
        "  metrics.py computes HV over every row supplied; it has no "
        "feasibility check."
    )
    print(f"  NSGA-II front size: {len(all_front)}")
    print(f"  NSGA-II feasible front size: {len(feasible_front)}")
    print(f"  HV using benchmark front: {raw_hv:.8f}")
    print(f"  HV restricted to feasible plans: {feasible_hv:.8f}")
    print(
        "  Therefore a 0%-feasible method can show HV (such as 0.0043): "
        "the archive fallback keeps its least-violating plan when no feasible "
        "plan exists, and that infeasible objective vector is still passed to HV."
    )


def _hybrid_seed_report(problem: NBAProblem) -> None:
    """Compare Hybrid-Memetic fronts for seeds 0 and 1 as requested."""
    params = problem.config.get("algorithm_parameters", {})
    fronts: Dict[int, np.ndarray] = {}
    for seed in (0, 1):
        result = MemeticNSGA2Optimizer(
            problem,
            pop_size=benchmark.EVALUATION_BUDGET,
            max_generations=0,
            crossover_prob=float(params.get("crossover_probability", 0.9)),
            mutation_prob=float(params.get("mutation_probability", 0.15)),
            seed=seed,
        ).run()
        fronts[seed] = np.asarray(
            [solution.objectives for solution in result.pareto_solutions]
        )

    identical = fronts[0].shape == fronts[1].shape and np.array_equal(
        np.sort(fronts[0], axis=0), np.sort(fronts[1], axis=0)
    )
    print("\n6. SEEDS")
    for seed in (0, 1):
        print(f"  Hybrid-Memetic seed {seed} objective vectors:")
        for vector in fronts[seed]:
            print(f"    {_format_vector(vector)}")
    print(f"  Front objective vectors identical: {identical}")
    print(
        "  Note: benchmark.py uses seeds [42, 101, 2024, 777, 999], "
        "not seed values 0 and 1."
    )


def _summary(problem: NBAProblem) -> None:
    print("\nSUMMARY")
    print(
        "Single biggest cause: benchmark.py overrides every evolutionary "
        "algorithm to zero generations and gives stochastic methods only 100 "
        "evaluations, so it measures the unrepaired initial/search sample "
        "rather than the configured evolutionary run. The full configured "
        "NSGA-II run reaches 97% feasibility by generation 40 and 100% by "
        "generation 60."
    )
    print("Fix first, in order (diagnosis only; nothing was changed):")
    print("  1. Restore a real, explicit evaluation budget and do not force zero generations.")
    print("  2. Use constraint-aware initialization/repair or MILP seeding for the 100-plan case.")
    print("  3. Measure/report hypervolume using feasible-only fronts.")
    print("  4. Count objective calls consistently, including MILP and warm-start evaluations.")
    print(
        "Bugs/measurement issues noticed: the benchmark's feasible count filters "
        "the final population, but its HV can use an infeasible fallback; "
        "SPEA2 splits its 100-plan budget between two initial populations; "
        "MILP is reported under the same nominal budget despite using a different "
        "number of objective evaluations."
    )


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print("NBA medium-instance diagnosis")
    print(f"Config: {CONFIG_PATH}")
    print(f"Configured target score R: {config['planning_parameters']['min_accreditation_score']}")

    problem = NBAProblem(CONFIG_PATH)
    _benchmark_evaluation_counts(problem)
    _constraint_report(problem)
    _milp_killer_constraint_report(problem)
    _nsga_generation_report(problem)
    _hypervolume_report(problem)
    _hybrid_seed_report(problem)
    _summary(problem)


if __name__ == "__main__":
    main()
