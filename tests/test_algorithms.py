"""
Unit tests for all optimization algorithms: NSGA-II, SPEA2, MOEA/D,
Hybrid Memetic, Greedy, and HiGHS MILP solver.
"""

from pathlib import Path
import pytest

from src.algorithms.moead import MOEADOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer
from src.algorithms.spea2 import SPEA2Optimizer
from src.baseline.greedy import GreedyBaseline
from src.baseline.milp_solver import HiGHSMILPSolver
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.problem import NBAProblem

CONFIG_PATH = Path("experiments/config.yaml")


@pytest.fixture
def problem() -> NBAProblem:
    return NBAProblem(CONFIG_PATH)


def test_highs_milp_solver(problem: NBAProblem) -> None:
    """Verify exact HiGHS solver solves single objective and finds feasible solution."""
    solver = HiGHSMILPSolver(problem)
    sol_max_f1 = solver.solve_single_objective("max_f1")
    assert sol_max_f1 is not None
    assert sol_max_f1.is_feasible
    assert sol_max_f1.f1_total_score >= 850.0
    assert sol_max_f1.f2_cost_lakh <= problem.budget

    sol_min_f2 = solver.solve_single_objective("min_f2")
    assert sol_min_f2 is not None
    assert sol_min_f2.is_feasible
    assert sol_min_f2.f1_total_score >= problem.min_accreditation_score


def test_nsga2_optimizer(problem: NBAProblem) -> None:
    """Verify NSGA-II generates feasible non-dominated solutions."""
    opt = NSGA2Optimizer(problem, pop_size=20, max_generations=10, seed=42)
    res = opt.run()
    assert len(res.pareto_solutions) > 0
    for s in res.pareto_solutions:
        assert s.is_feasible
        assert s.f1_total_score >= problem.min_accreditation_score


def test_spea2_optimizer(problem: NBAProblem) -> None:
    """Verify SPEA2 generates feasible non-dominated solutions."""
    opt = SPEA2Optimizer(problem, pop_size=20, archive_size=20, max_generations=10, seed=42)
    res = opt.run()
    assert len(res.pareto_solutions) > 0
    for s in res.pareto_solutions:
        assert s.is_feasible


def test_moead_optimizer(problem: NBAProblem) -> None:
    """Verify MOEA/D generates feasible non-dominated solutions."""
    opt = MOEADOptimizer(problem, pop_size=20, n_neighbors=5, max_generations=10, seed=42)
    res = opt.run()
    assert len(res.pareto_solutions) > 0
    for s in res.pareto_solutions:
        assert s.is_feasible


def test_hybrid_memetic_optimizer(problem: NBAProblem) -> None:
    """Verify Memetic NSGA-II runs and finds high-quality solutions."""
    opt = MemeticNSGA2Optimizer(problem, pop_size=20, max_generations=10, seed=42)
    res = opt.run()
    assert len(res.pareto_solutions) > 0
    for s in res.pareto_solutions:
        assert s.is_feasible


def test_greedy_baseline(problem: NBAProblem) -> None:
    """Verify Greedy baseline generates valid solutions."""
    baseline = GreedyBaseline(problem, seed=42)
    res = baseline.run()
    assert len(res.pareto_solutions) > 0
    for s in res.pareto_solutions:
        assert s.is_feasible
