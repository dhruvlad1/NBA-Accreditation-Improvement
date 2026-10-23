"""
Weight vector generation for decomposition-based multi-objective optimization (MOEA/D).
Implements the Das-Dennis simplex lattice design and uniform Dirichlet sampling.
"""

from __future__ import annotations

from typing import List
import numpy as np


def generate_simplex_lattice(num_objectives: int, divisions: int) -> np.ndarray:
    """
    Generate uniformly distributed weight vectors on unit simplex using Das-Dennis method.

    Args:
        num_objectives: Number of objectives M.
        divisions: Number of divisions H along each axis.

    Returns:
        2D numpy array of shape (N, M) where sum of each row is 1.0.
    """
    weights: List[List[float]] = []

    def _generate(current_weight: List[float], left: int, total: int) -> None:
        if left == 1:
            current_weight.append(total / divisions)
            weights.append(list(current_weight))
            current_weight.pop()
            return
        for i in range(total + 1):
            current_weight.append(i / divisions)
            _generate(current_weight, left - 1, total - i)
            current_weight.pop()

    _generate([], num_objectives, divisions)
    W = np.array(weights, dtype=np.float64)
    # Avoid exact zeros for Tchebycheff division stability
    W = np.maximum(W, 1e-6)
    W = W / np.sum(W, axis=1, keepdims=True)
    return W


def generate_weight_vectors(num_objectives: int, target_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate approximately target_size weight vectors.
    Finds the closest Das-Dennis division, or supplements via uniform Dirichlet sampling.

    Args:
        num_objectives: Number of objectives M.
        target_size: Desired number of weight vectors.
        rng: Optional seeded numpy random generator.

    Returns:
        2D numpy array of shape (K, M) where each row sums to 1.0.
    """
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    # Find division H that gives count closest to target_size
    best_h = 1
    best_diff = float("inf")
    best_w = None

    for h in range(1, 20):
        w = generate_simplex_lattice(num_objectives, h)
        diff = abs(len(w) - target_size)
        if diff < best_diff:
            best_diff = diff
            best_h = h
            best_w = w
        if len(w) >= target_size * 2:
            break

    assert best_w is not None

    if len(best_w) == target_size:
        return best_w

    if len(best_w) > target_size:
        # Subsample deterministically or with rng
        if rng is not None:
            indices = rng.choice(len(best_w), size=target_size, replace=False)
        else:
            indices = np.linspace(0, len(best_w) - 1, target_size, dtype=int)
        return best_w[indices]

    # If fewer, add Dirichlet samples
    needed = target_size - len(best_w)
    local_rng = rng if rng is not None else np.random.default_rng(42)
    extra = local_rng.dirichlet(np.ones(num_objectives), size=needed)
    combined = np.vstack([best_w, extra])
    return combined
