"""
Multi-objective dominance and sorting operators.
Includes standard Pareto dominance, Deb's constrained dominance,
and fast non-dominated sorting (Deb et al., 2002).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np


def pareto_dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """
    Check if obj1 Pareto-dominates obj2 (assuming all objectives are to be minimized).

    Args:
        obj1: Objective vector of first solution.
        obj2: Objective vector of second solution.

    Returns:
        True if obj1 <= obj2 everywhere and obj1 < obj2 in at least one objective.
    """
    return bool(np.all(obj1 <= obj2) and np.any(obj1 < obj2))


def constrained_dominates(
    obj1: np.ndarray,
    viol1: float,
    obj2: np.ndarray,
    viol2: float,
    tol: float = 1e-7
) -> bool:
    """
    Deb's constrained dominance rule:
    1. If sol1 is feasible and sol2 is infeasible: sol1 dominates sol2.
    2. If both are infeasible: sol1 dominates sol2 if viol1 < viol2.
    3. If both are feasible: sol1 dominates sol2 if obj1 Pareto-dominates obj2.

    Args:
        obj1: Objective vector 1 (minimization).
        viol1: Total constraint violation of solution 1.
        obj2: Objective vector 2 (minimization).
        viol2: Total constraint violation of solution 2.
        tol: Tolerance for feasibility.

    Returns:
        True if solution 1 dominates solution 2 under Deb's rule.
    """
    feas1 = viol1 <= tol
    feas2 = viol2 <= tol

    if feas1 and not feas2:
        return True
    if not feas1 and feas2:
        return False
    if not feas1 and not feas2:
        return viol1 < (viol2 - tol)

    # Both are feasible: standard Pareto dominance
    return pareto_dominates(obj1, obj2)


def fast_non_dominated_sort(
    objective_matrix: np.ndarray,
    violations: Sequence[float] | np.ndarray | None = None,
    tol: float = 1e-7
) -> List[List[int]]:
    """
    Fast Non-Dominated Sorting algorithm (Deb et al., 2002).
    Supports both unconstrained and constrained dominance.

    Args:
        objective_matrix: 2D array of shape (N, M) of minimization objectives.
        violations: 1D array of shape (N,) with total constraint violations (0 if feasible).
                    If None, standard unconstrained Pareto dominance is used.
        tol: Tolerance for constraint violation comparison.

    Returns:
        List of fronts, where each front is a list of integer indices [F_1, F_2, ...].
        F_1 contains indices of non-dominated (rank 0) solutions.
    """
    N = len(objective_matrix)
    if N == 0:
        return []

    use_constraints = violations is not None
    viols = np.asarray(violations, dtype=np.float64) if use_constraints else np.zeros(N)

    # S[p]: list of solutions that solution p dominates
    # n[p]: domination counter (number of solutions that dominate p)
    S: List[List[int]] = [[] for _ in range(N)]
    n: np.ndarray = np.zeros(N, dtype=np.int32)
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(p + 1, N):
            p_dom_q = False
            q_dom_p = False

            if use_constraints:
                p_dom_q = constrained_dominates(
                    objective_matrix[p], viols[p],
                    objective_matrix[q], viols[q], tol=tol
                )
                q_dom_p = constrained_dominates(
                    objective_matrix[q], viols[q],
                    objective_matrix[p], viols[p], tol=tol
                )
            else:
                p_dom_q = pareto_dominates(objective_matrix[p], objective_matrix[q])
                q_dom_p = pareto_dominates(objective_matrix[q], objective_matrix[p])

            if p_dom_q:
                S[p].append(q)
                n[q] += 1
            elif q_dom_p:
                S[q].append(p)
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and len(fronts[i]) > 0:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts
