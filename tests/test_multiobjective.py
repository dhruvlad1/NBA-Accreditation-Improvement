"""
Unit tests for dominance, sorting, crowding distance, archive, and weights.
"""

import numpy as np
import pytest

from src.multiobjective.archive import ParetoArchive
from src.multiobjective.crowding import compute_crowding_distance
from src.multiobjective.dominance import (
    constrained_dominates,
    fast_non_dominated_sort,
    pareto_dominates,
)
from src.multiobjective.weights import generate_simplex_lattice, generate_weight_vectors
from src.problem import EvaluationResult


def test_pareto_dominates() -> None:
    """Verify Pareto dominance for minimization."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.5, 2.5, 3.5])
    c = np.array([0.5, 3.0, 3.0])

    assert pareto_dominates(a, b)
    assert not pareto_dominates(b, a)
    assert not pareto_dominates(a, c)
    assert not pareto_dominates(c, a)


def test_constrained_dominates() -> None:
    """Verify Deb's constrained dominance rules."""
    feas_good = np.array([1.0, 1.0])
    feas_bad = np.array([2.0, 2.0])
    infeas_low = np.array([0.5, 0.5])
    infeas_high = np.array([0.1, 0.1])

    # Feasible dominates infeasible
    assert constrained_dominates(feas_good, 0.0, infeas_low, 5.0)
    assert not constrained_dominates(infeas_low, 5.0, feas_good, 0.0)

    # Infeasible comparison by violation magnitude
    assert constrained_dominates(infeas_high, 2.0, infeas_low, 5.0)
    assert not constrained_dominates(infeas_low, 5.0, infeas_high, 2.0)

    # Both feasible comparison by Pareto dominance
    assert constrained_dominates(feas_good, 0.0, feas_bad, 0.0)


def test_fast_non_dominated_sort() -> None:
    """Verify front partitioning."""
    objs = np.array([
        [1.0, 5.0],  # Front 0
        [2.0, 3.0],  # Front 0
        [5.0, 1.0],  # Front 0
        [3.0, 4.0],  # Dominated by [2.0, 3.0] -> Front 1
    ])
    fronts = fast_non_dominated_sort(objs)
    assert 0 in fronts[0]
    assert 1 in fronts[0]
    assert 2 in fronts[0]
    assert fronts[1] == [3]


def test_crowding_distance() -> None:
    """Verify boundary points get infinity and intermediate points are positive."""
    objs = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [3.0, 1.0],
    ])
    dists = compute_crowding_distance(objs, [0, 1, 2])
    assert np.isinf(dists[0])
    assert np.isinf(dists[2])
    assert dists[1] > 0.0


def test_weight_generation() -> None:
    """Verify Das-Dennis weight generation properties."""
    w = generate_simplex_lattice(4, 4)
    assert w.shape[1] == 4
    np.testing.assert_allclose(np.sum(w, axis=1), 1.0, atol=1e-5)

    sampled = generate_weight_vectors(4, 50)
    assert len(sampled) == 50
    np.testing.assert_allclose(np.sum(sampled, axis=1), 1.0, atol=1e-5)
