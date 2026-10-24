"""Independent plain-loop checker for the proposal formulas (Sections 6.3-6.5)."""

from pathlib import Path

import numpy as np

from src.problem import NBAProblem


def _reference(problem: NBAProblem, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    scores = []
    for criterion in problem.criteria:
        score = problem.s0[criterion.id - 1]
        for index in criterion.action_indices:
            score += problem.gains[index] * x[index]
        scores.append(min(problem.M[criterion.id - 1], score))
    scores_array = np.asarray(scores, dtype=float)
    monthly = np.zeros(problem.planning_horizon)
    for action in problem.actions:
        for month in range(1, problem.planning_horizon + 1):
            if action.earliest_start < month <= action.earliest_finish:
                monthly[month - 1] += action.faculty_hours / action.duration_months * x[action.index]

    budget = max(0.0, float(np.dot(problem.costs, x)) - problem.budget)
    monthly_violation = float(np.maximum(0.0, monthly - problem.monthly_capacity).sum())
    deadline = float(sum(x[action.index] for action in problem.actions
                         if action.earliest_finish > problem.planning_horizon))
    prerequisites = 0.0
    for action in problem.actions:
        for parent in action.prereq_indices:
            prerequisites += max(0.0, x[action.index] - action.max_units * x[parent])
    criteria_min = float(np.maximum(0.0, problem.L - scores_array).sum())
    target = max(0.0, problem.min_accreditation_score - float(scores_array.sum()))
    return scores_array, {
        "budget": budget,
        "monthly_hours": monthly_violation,
        "deadline": deadline,
        "prerequisites": prerequisites,
        "criteria_min": criteria_min,
        "target_score": target,
    }


def test_evaluate_matches_independent_reference() -> None:
    problem = NBAProblem(Path("experiments/config.yaml"))
    rng = np.random.default_rng(20261024)
    names = ("budget", "monthly_hours", "deadline", "prerequisites", "criteria_min", "target_score")
    for _ in range(2000):
        x = problem.random_solution(rng, repair=False)
        expected_scores, expected_violations = _reference(problem, x)
        actual = problem.evaluate(x)
        assert np.allclose(actual.criteria_scores, expected_scores, atol=1e-9)
        assert np.allclose(actual.objectives[0], expected_scores.sum(), atol=1e-9)
        assert np.allclose([actual.violations[name] for name in names],
                           [expected_violations[name] for name in names], atol=1e-9)
        assert actual.is_feasible == (sum(expected_violations.values()) <= 1e-7)
