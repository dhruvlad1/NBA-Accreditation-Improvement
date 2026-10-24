"""Properties for the structural repair operator (Proposal Section 10)."""

from pathlib import Path

import numpy as np

from src.problem import NBAProblem


def test_repair_properties() -> None:
    problem = NBAProblem(Path("experiments/config.yaml"))
    rng = np.random.default_rng(7)
    for _ in range(1000):
        raw = rng.integers(-3, problem.upper_bounds + 4)
        repaired = problem.repair(raw)
        assert problem.structural_violation(repaired) == 0.0
        assert np.all((0 <= repaired) & (repaired <= problem.upper_bounds))
        assert np.array_equal(problem.repair(repaired), repaired)
        feasible = problem.repair(raw)
        assert np.array_equal(problem.repair(feasible), feasible)

    raw = problem.random_solution(np.random.default_rng(11), repair=False)
    first = problem.repair(raw, np.random.default_rng(99))
    second = problem.repair(raw, np.random.default_rng(99))
    assert np.array_equal(first, second)


def test_repair_deadline_and_prerequisite_closure() -> None:
    problem = NBAProblem(Path("experiments/config.yaml"))
    original_mask = problem.schedulable_mask.copy()
    try:
        middle = next(index for index, action in enumerate(problem.actions)
                      if problem.dep[index].any())
        problem.schedulable_mask[middle] = False
        raw = problem.upper_bounds.copy()
        repaired = problem.repair(raw)
        assert repaired[middle] == 0
        assert np.all(repaired[problem.dep[middle]] == 0)
    finally:
        problem.schedulable_mask[:] = original_mask

    leaf = next(index for index, action in enumerate(problem.actions) if action.prereq_indices)
    raw = np.zeros(problem.num_actions, dtype=np.int32)
    raw[leaf] = 1
    repaired = problem.repair(raw)
    assert np.all(repaired[np.flatnonzero(problem.dep[:, leaf])] >= 1)


def test_evaluate_and_repair_do_not_modify_inputs() -> None:
    problem = NBAProblem(Path("experiments/config.yaml"))
    x = np.arange(problem.num_actions, dtype=np.int32)
    original = x.copy()
    problem.evaluate(x)
    assert np.array_equal(x, original)
    problem.repair(x)
    assert np.array_equal(x, original)
