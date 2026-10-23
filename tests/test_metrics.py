"""
Unit tests for quality indicators and metrics (HV, Spacing, Spread, IGD, Coverage).
"""

import numpy as np
import pytest

from src.metrics import (
    compute_coverage,
    compute_hypervolume,
    compute_igd,
    compute_spacing,
    compute_spread,
)


def test_hypervolume_calculation() -> None:
    """Verify pymoo HV wrapper produces strictly positive value for valid fronts."""
    # Min objectives: [-f1, f2, f3, -f4]
    front = np.array([
        [-880.0, 25.0, 1200.0, -0.7],
        [-850.0, 18.0, 900.0, -0.65],
        [-790.0, 12.0, 600.0, -0.58],
    ])
    hv_val = compute_hypervolume(front, normalize=True)
    assert hv_val > 0.0


def test_spacing_metric() -> None:
    """Verify spacing calculation."""
    front = np.array([
        [1.0, 5.0],
        [2.0, 4.0],
        [3.0, 3.0],
        [4.0, 2.0],
    ])
    s = compute_spacing(front)
    assert s >= 0.0


def test_spread_metric() -> None:
    """Verify generalized spread."""
    front = np.array([
        [1.0, 4.0],
        [2.0, 3.0],
        [3.0, 2.0],
    ])
    delta = compute_spread(front)
    assert 0.0 <= delta <= 2.0


def test_coverage_metric() -> None:
    """Verify C-metric."""
    front_a = np.array([[1.0, 1.0]])
    front_b = np.array([[2.0, 2.0], [3.0, 3.0]])
    # front_a dominates both points in front_b
    cov_ab = compute_coverage(front_a, front_b)
    assert cov_ab == 1.0

    cov_ba = compute_coverage(front_b, front_a)
    assert cov_ba == 0.0
