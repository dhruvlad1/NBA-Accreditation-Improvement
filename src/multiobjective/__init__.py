"""
Multi-objective optimization utilities and primitives.
"""

from src.multiobjective.archive import ParetoArchive
from src.multiobjective.crowding import compute_crowding_distance
from src.multiobjective.dominance import (
    constrained_dominates,
    fast_non_dominated_sort,
    pareto_dominates,
)
from src.multiobjective.weights import generate_simplex_lattice, generate_weight_vectors

__all__ = [
    "ParetoArchive",
    "compute_crowding_distance",
    "constrained_dominates",
    "fast_non_dominated_sort",
    "pareto_dominates",
    "generate_simplex_lattice",
    "generate_weight_vectors",
]
