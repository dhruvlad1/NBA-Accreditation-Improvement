"""
Genetic operators for integer-encoded multi-objective optimization.
Includes simulated binary crossover (SBX), uniform crossover,
polynomial mutation, unit flip mutation, and binary tournament selection.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np


def integer_sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    upper_bounds: np.ndarray,
    eta_c: float = 15.0,
    crossover_prob: float = 0.9,
    rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) adapted for integer decision variables.

    Args:
        parent1: First parent array (integers).
        parent2: Second parent array (integers).
        upper_bounds: Upper bound for each variable.
        eta_c: Distribution index for crossover.
        crossover_prob: Probability of performing crossover.
        rng: Seeded numpy random generator.

    Returns:
        Two offspring integer arrays bounded within [0, upper_bounds].
    """
    if rng is None:
        rng = np.random.default_rng()

    c1 = parent1.copy().astype(np.float64)
    c2 = parent2.copy().astype(np.float64)

    if rng.random() > crossover_prob:
        return np.rint(c1).astype(np.int32), np.rint(c2).astype(np.int32)

    n_vars = len(parent1)
    for i in range(n_vars):
        if rng.random() <= 0.5 and abs(parent1[i] - parent2[i]) > 1e-6:
            y1 = min(parent1[i], parent2[i])
            y2 = max(parent1[i], parent2[i])
            lb = 0.0
            ub = float(upper_bounds[i])

            rand = rng.random()
            # Beta calculation for bounded SBX
            beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1 + 1e-12))
            alpha = 2.0 - beta ** -(eta_c + 1.0)
            if rand <= (1.0 / alpha):
                beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            c1_val = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

            beta = 1.0 + (2.0 * (ub - y2) / (y2 - y1 + 1e-12))
            alpha = 2.0 - beta ** -(eta_c + 1.0)
            if rand <= (1.0 / alpha):
                beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            c2_val = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

            if rng.random() <= 0.5:
                c1[i] = c2_val
                c2[i] = c1_val
            else:
                c1[i] = c1_val
                c2[i] = c2_val

    c1 = np.clip(np.rint(c1), 0, upper_bounds).astype(np.int32)
    c2 = np.clip(np.rint(c2), 0, upper_bounds).astype(np.int32)
    return c1, c2


def integer_uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_prob: float = 0.9,
    swap_prob: float = 0.5,
    rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover for integer vectors."""
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    mask = rng.random(len(parent1)) < swap_prob
    c1 = np.where(mask, parent2, parent1).astype(np.int32)
    c2 = np.where(mask, parent1, parent2).astype(np.int32)
    return c1, c2


def integer_polynomial_mutation(
    individual: np.ndarray,
    upper_bounds: np.ndarray,
    eta_m: float = 20.0,
    mutation_prob: float | None = None,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Polynomial mutation adapted for integer representation.

    Args:
        individual: 1D integer array.
        upper_bounds: Upper bound per decision variable.
        eta_m: Distribution index for mutation.
        mutation_prob: Mutation probability per gene (defaults to 1 / N).
        rng: Seeded numpy random generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    mutated = individual.copy().astype(np.float64)
    n_vars = len(individual)
    prob = mutation_prob if mutation_prob is not None else (1.0 / n_vars)

    for i in range(n_vars):
        if rng.random() <= prob:
            y = mutated[i]
            lb = 0.0
            ub = float(upper_bounds[i])
            if ub <= lb:
                continue

            delta1 = (y - lb) / (ub - lb)
            delta2 = (ub - y) / (ub - lb)
            rand = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                delta_q = 1.0 - val ** mut_pow

            y = y + delta_q * (ub - lb)
            mutated[i] = y

    return np.clip(np.rint(mutated), 0, upper_bounds).astype(np.int32)


def integer_unit_flip_mutation(
    individual: np.ndarray,
    upper_bounds: np.ndarray,
    mutation_prob: float = 0.15,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Discrete unit perturb mutation: randomly adds +1 or -1 or resets unit.
    Very effective for knapsack and discrete scheduling problems.
    """
    if rng is None:
        rng = np.random.default_rng()

    mutated = individual.copy()
    n_vars = len(individual)

    for i in range(n_vars):
        if rng.random() < mutation_prob:
            ub = upper_bounds[i]
            if ub <= 0:
                mutated[i] = 0
                continue
            choice = rng.choice([-1, 1, 999])
            if choice == 999:
                # Random reset
                mutated[i] = rng.integers(0, ub + 1)
            else:
                mutated[i] = np.clip(mutated[i] + choice, 0, ub)

    return mutated


def binary_tournament_selection(
    ranks: np.ndarray,
    crowding_distances: np.ndarray,
    rng: np.random.Generator | None = None
) -> int:
    """
    Standard NSGA-II crowded comparison binary tournament selection.
    Prefers lower rank (better front); ties broken by higher crowding distance.

    Args:
        ranks: 1D array of Pareto front ranks (0 = best).
        crowding_distances: 1D array of crowding distances.
        rng: Seeded numpy random generator.

    Returns:
        Index of the winning individual.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(ranks)
    i1, i2 = rng.choice(n, size=2, replace=False)

    # 1. Compare ranks (lower rank is better)
    if ranks[i1] < ranks[i2]:
        return int(i1)
    if ranks[i2] < ranks[i1]:
        return int(i2)

    # 2. Compare crowding distances (larger distance is better)
    if crowding_distances[i1] > crowding_distances[i2]:
        return int(i1)
    return int(i2)
