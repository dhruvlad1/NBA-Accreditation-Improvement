from pathlib import Path

import numpy as np

from src.algorithms.nsga2 import NSGA2Optimizer
from src.hybrid.corner_nsga2 import CornerNSGA2
from src.problem import NBAProblem


CONFIG_PATH = Path("experiments/config.yaml")


def test_corner_flags_off_match_nsga2() -> None:
    plain = NSGA2Optimizer(NBAProblem(CONFIG_PATH), pop_size=20, max_generations=4, seed=7).run()
    corner = CornerNSGA2(
        NBAProblem(CONFIG_PATH), pop_size=20, max_generations=4, seed=7,
        extreme_seeding=False, corner_local_search=False,
    ).run()
    assert [solution.x.tolist() for solution in corner.pareto_solutions] == [
        solution.x.tolist() for solution in plain.pareto_solutions
    ]
    assert corner.total_evaluations == plain.total_evaluations


def test_corner_budget_is_exact() -> None:
    result = CornerNSGA2(
        NBAProblem(CONFIG_PATH), pop_size=20, max_generations=4, seed=7,
        extreme_seeding=True, corner_local_search=True,
    ).run()
    assert result.total_evaluations == 20 * 5


def test_corner_returned_plans_are_feasible() -> None:
    problem = NBAProblem(CONFIG_PATH)
    result = CornerNSGA2(
        problem, pop_size=20, max_generations=4, seed=7,
        extreme_seeding=True, corner_local_search=True,
    ).run()
    assert result.pareto_solutions
    assert all(solution.is_feasible for solution in result.pareto_solutions)
    assert all(np.isclose(solution.total_violation, 0.0) for solution in result.pareto_solutions)
