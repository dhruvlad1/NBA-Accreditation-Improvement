"""
NBA Accreditation Improvement Planning optimization framework.
"""

from src.metrics import (
    compute_coverage,
    compute_hypervolume,
    compute_igd,
    compute_spacing,
    compute_spread,
)
from src.problem import Action, Criterion, EvaluationResult, NBAProblem

__all__ = [
    "NBAProblem",
    "Action",
    "Criterion",
    "EvaluationResult",
    "compute_hypervolume",
    "compute_spacing",
    "compute_spread",
    "compute_igd",
    "compute_coverage",
]
