"""
Multi-objective performance indicators and quality metrics.
Integrates pymoo's Hypervolume indicator (HV) as specified,
and implements Spacing (S), Spread (Delta), Inverted Generational Distance (IGD),
and Set Coverage (C-metric) from scratch.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from pymoo.indicators.hv import HV

from src.problem import EvaluationResult


def compute_hypervolume(
    objective_matrix_min: np.ndarray,
    ref_point: np.ndarray | None = None,
    ideal_point: np.ndarray | None = None,
    nadir_point: np.ndarray | None = None,
    normalize: bool = True
) -> float:
    """
    Compute hypervolume using pymoo's HV indicator.

    Args:
        objective_matrix_min: 2D array (N, M) of minimization objectives [-f1, f2, f3, -f4].
        ref_point: Reference point for HV calculation. If normalize=True and None,
                   defaults to np.array([1.1, 1.1, 1.1, 1.1]).
        ideal_point: Lower bound vector for normalization.
        nadir_point: Upper bound vector for normalization.
        normalize: Whether to normalize objectives into [0, 1] prior to HV computation.

    Returns:
        Hypervolume value (float). Returns 0.0 if input is empty.
    """
    if len(objective_matrix_min) == 0:
        return 0.0

    objs = np.asarray(objective_matrix_min, dtype=np.float64)

    if normalize:
        if ideal_point is None:
            # Synthetic default ideal: [f1=900, f2=9.0, f3=400, f4=0.8]
            ideal = np.array([-900.0, 9.0, 400.0, -0.8], dtype=np.float64)
        else:
            ideal = np.asarray(ideal_point, dtype=np.float64)

        if nadir_point is None:
            # Synthetic default nadir: [f1=750, f2=30.0, f3=1600, f4=0.4]
            nadir = np.array([-750.0, 30.0, 1600.0, -0.4], dtype=np.float64)
        else:
            nadir = np.asarray(nadir_point, dtype=np.float64)

        ranges = np.where((nadir - ideal) > 1e-12, nadir - ideal, 1.0)
        norm_objs = (objs - ideal) / ranges

        # Filter out points that are worse than reference point
        standard_ref = ref_point if ref_point is not None else np.array([1.1, 1.1, 1.1, 1.1], dtype=np.float64)
        valid_mask = np.all(norm_objs <= standard_ref, axis=1)
        valid_objs = norm_objs[valid_mask]

        if len(valid_objs) == 0:
            return 0.0

        indicator = HV(ref_point=standard_ref)
        try:
            return float(indicator(valid_objs))
        except Exception:
            return 0.0
    else:
        if ref_point is None:
            # Absolute default nadir reference point
            ref_point = np.array([-740.0, 31.0, 1700.0, -0.35], dtype=np.float64)
        valid_mask = np.all(objs <= ref_point, axis=1)
        valid_objs = objs[valid_mask]
        if len(valid_objs) == 0:
            return 0.0
        indicator = HV(ref_point=ref_point)
        try:
            return float(indicator(valid_objs))
        except Exception:
            return 0.0


def compute_spacing(objective_matrix: np.ndarray) -> float:
    """
    Spacing metric (Schott, 1995).
    Measures the standard deviation of Manhattan distances to the nearest neighbor.
    Lower spacing indicates more uniform distribution.

    Args:
        objective_matrix: 2D array of shape (N, M).

    Returns:
        Spacing S (float).
    """
    n = len(objective_matrix)
    if n <= 1:
        return 0.0

    # Normalize objectives to avoid objective scaling dominance
    mins = np.min(objective_matrix, axis=0)
    maxs = np.max(objective_matrix, axis=0)
    ranges = np.where((maxs - mins) > 1e-12, maxs - mins, 1.0)
    norm = (objective_matrix - mins) / ranges

    # Pairwise Manhattan (L1) distances
    diff = norm[:, np.newaxis, :] - norm[np.newaxis, :, :]
    dist_mat = np.sum(np.abs(diff), axis=-1)
    np.fill_diagonal(dist_mat, np.inf)

    d_i = np.min(dist_mat, axis=1)
    d_mean = np.mean(d_i)
    spacing = np.sqrt(np.sum((d_i - d_mean) ** 2) / (n - 1))
    return float(spacing)


def compute_spread(
    objective_matrix: np.ndarray,
    reference_extremes: np.ndarray | None = None
) -> float:
    """
    Generalized Spread metric Delta (Deb et al., 2002).
    Measures extent and uniformity of the non-dominated set.

    Args:
        objective_matrix: 2D array of shape (N, M).
        reference_extremes: Optional extreme points of the reference front.

    Returns:
        Delta (float). 0.0 indicates ideal uniform distribution covering extremes.
    """
    n = len(objective_matrix)
    if n <= 1:
        return 1.0

    mins = np.min(objective_matrix, axis=0)
    maxs = np.max(objective_matrix, axis=0)
    ranges = np.where((maxs - mins) > 1e-12, maxs - mins, 1.0)
    norm = (objective_matrix - mins) / ranges

    # Pairwise Euclidean distances
    diff = norm[:, np.newaxis, :] - norm[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist_mat, np.inf)

    d_i = np.min(dist_mat, axis=1)
    d_mean = np.mean(d_i)

    if reference_extremes is not None and len(reference_extremes) >= 2:
        norm_ref = (reference_extremes - mins) / ranges
        # Distances to boundary extremes
        d_f = np.min(np.sqrt(np.sum((norm - norm_ref[0]) ** 2, axis=1)))
        d_l = np.min(np.sqrt(np.sum((norm - norm_ref[-1]) ** 2, axis=1)))
    else:
        d_f = 0.0
        d_l = 0.0

    numerator = d_f + d_l + np.sum(np.abs(d_i - d_mean))
    denominator = d_f + d_l + (n - 1) * d_mean

    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def compute_igd(
    objective_matrix: np.ndarray,
    reference_front: np.ndarray
) -> float:
    """
    Inverted Generational Distance (IGD).
    Average Euclidean distance from each point in the reference front to its nearest point in the obtained front.

    Args:
        objective_matrix: Obtained front (N, M).
        reference_front: True/Reference Pareto front (K, M).

    Returns:
        IGD (float). Lower is better.
    """
    if len(objective_matrix) == 0 or len(reference_front) == 0:
        return float("inf")

    # Normalize based on reference front bounds
    mins = np.min(reference_front, axis=0)
    maxs = np.max(reference_front, axis=0)
    ranges = np.where((maxs - mins) > 1e-12, maxs - mins, 1.0)

    norm_objs = (objective_matrix - mins) / ranges
    norm_ref = (reference_front - mins) / ranges

    # Distance from each ref point to nearest obtained point
    diff = norm_ref[:, np.newaxis, :] - norm_objs[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    min_dists = np.min(dists, axis=1)

    return float(np.mean(min_dists))


def compute_coverage(front_a: np.ndarray, front_b: np.ndarray) -> float:
    """
    Coverage metric C(A, B) (Zitzler, 1999).
    Fraction of solutions in B that are weakly dominated by at least one solution in A.

    Args:
        front_a: 2D array of minimization objectives from set A.
        front_b: 2D array of minimization objectives from set B.

    Returns:
        Float in [0, 1]. C(A, B) = 1 means all solutions in B are dominated by A.
    """
    if len(front_b) == 0:
        return 0.0
    if len(front_a) == 0:
        return 0.0

    dominated_count = 0
    for b in front_b:
        # Check if any a in A dominates or equals b
        is_dom = False
        for a in front_a:
            if np.all(a <= b) and np.any(a < b):
                is_dom = True
                break
        if is_dom:
            dominated_count += 1

    return float(dominated_count / len(front_b))
