"""
Crowding distance computation for non-dominated fronts (Deb et al., 2002).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compute_crowding_distance(
    objective_matrix: np.ndarray,
    front_indices: Sequence[int]
) -> np.ndarray:
    """
    Compute crowding distances for solutions in a given front.

    Args:
        objective_matrix: 2D array of shape (N, M) of objective values.
        front_indices: 1D sequence of integer indices belonging to the front.

    Returns:
        1D array of shape (len(front_indices),) with crowding distance values.
        Boundary points have distance np.inf.
    """
    front_len = len(front_indices)
    if front_len == 0:
        return np.array([], dtype=np.float64)
    if front_len <= 2:
        return np.full(front_len, np.inf, dtype=np.float64)

    distances = np.zeros(front_len, dtype=np.float64)
    sub_objs = objective_matrix[list(front_indices)]
    num_objs = sub_objs.shape[1]

    for m in range(num_objs):
        # Sort indices of the front according to objective m
        sorted_order = np.argsort(sub_objs[:, m])
        f_min = sub_objs[sorted_order[0], m]
        f_max = sub_objs[sorted_order[-1], m]

        # Boundary points get infinite distance
        distances[sorted_order[0]] = np.inf
        distances[sorted_order[-1]] = np.inf

        denom = f_max - f_min
        if denom <= 1e-12:
            continue

        for i in range(1, front_len - 1):
            if np.isneginf(distances[sorted_order[i]]) or np.isposinf(distances[sorted_order[i]]):
                continue
            prev_val = sub_objs[sorted_order[i - 1], m]
            next_val = sub_objs[sorted_order[i + 1], m]
            distances[sorted_order[i]] += (next_val - prev_val) / denom

    return distances
