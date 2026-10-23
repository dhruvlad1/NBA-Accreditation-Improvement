"""
Unit tests for NBAProblem formulation, schedule precomputation,
and constraint verification.
"""

from pathlib import Path
import numpy as np
import pytest

from src.problem import NBAProblem

CONFIG_PATH = Path("experiments/config.yaml")


@pytest.fixture
def problem() -> NBAProblem:
    return NBAProblem(CONFIG_PATH)


def test_problem_initialization(problem: NBAProblem) -> None:
    """Verify problem loads criteria and actions properly without hardcoding."""
    assert problem.num_criteria == 10
    assert problem.num_actions == 45
    assert problem.budget == 30.0
    assert problem.planning_horizon == 12
    assert problem.monthly_capacity == 300.0
    assert problem.min_accreditation_score == 750.0
    assert len(problem.actions) == 45
    assert len(problem.criteria) == 10


def test_earliest_start_schedule(problem: NBAProblem) -> None:
    """Verify earliest finish Fa = da + max Fb over prereqs and Sa = Fa - da."""
    for a in problem.actions:
        assert a.earliest_finish == a.duration_months + (
            max([problem.actions[p].earliest_finish for p in a.prereq_indices])
            if a.prereq_indices else 0
        )
        assert a.earliest_start == a.earliest_finish - a.duration_months


def test_deadline_constraint(problem: NBAProblem) -> None:
    """Constraint 3: x_a must be 0 if F_a > T."""
    unschedulable = [a for a in problem.actions if not a.is_schedulable]
    assert len(unschedulable) >= 1
    # Choosing units on unschedulable action causes deadline violation
    x = np.zeros(problem.num_actions, dtype=int)
    x[unschedulable[0].index] = 1
    res = problem.evaluate(x)
    assert res.violations["deadline"] == 1.0
    assert not res.is_feasible


def test_prerequisite_constraint(problem: NBAProblem) -> None:
    """Constraint 4: x_a <= u_a * x_b for every prerequisite b."""
    # Action 2 requires Action 1 (E1)
    x = np.zeros(problem.num_actions, dtype=int)
    x[1] = 1  # ACT_02
    res = problem.evaluate(x)
    assert res.violations["prerequisites"] > 0
    assert not res.is_feasible

    # When prerequisite ACT_01 is fulfilled, prereq violation is 0
    x[0] = 1  # ACT_01
    res2 = problem.evaluate(x)
    assert res2.violations["prerequisites"] == 0.0


def test_budget_constraint(problem: NBAProblem) -> None:
    """Constraint 1: f2 <= B."""
    # Set all units to max to exceed budget
    x = problem.upper_bounds.copy()
    res = problem.evaluate(x)
    assert res.f2_cost_lakh > problem.budget
    assert res.violations["budget"] > 0.0
    assert not res.is_feasible


def test_domain_repair_produces_feasible_solutions(problem: NBAProblem) -> None:
    """Verify repair operator generates strictly valid feasible configurations."""
    rng = np.random.default_rng(12345)
    for _ in range(20):
        raw_x = np.array([rng.integers(0, a.max_units + 1) for a in problem.actions])
        repaired = problem.repair(raw_x, rng=rng)
        res = problem.evaluate(repaired)
        assert res.is_feasible, f"Violations: {res.violations}"
        assert res.f1_total_score >= problem.min_accreditation_score
        assert res.f2_cost_lakh <= problem.budget
        assert np.all(res.monthly_hours <= problem.monthly_capacity)
        assert res.violations["deadline"] == 0.0
        assert res.violations["prerequisites"] == 0.0
        assert np.all(res.criteria_scores >= problem.L)
