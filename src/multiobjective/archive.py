"""
Bounded non-dominated external archive for multi-objective optimization.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from src.multiobjective.crowding import compute_crowding_distance
from src.multiobjective.dominance import constrained_dominates, fast_non_dominated_sort
from src.problem import EvaluationResult


class ParetoArchive:
    """
    Maintains a bounded set of non-dominated solutions.
    """

    def __init__(self, max_size: int = 100) -> None:
        """
        Initialize archive with a maximum capacity.

        Args:
            max_size: Maximum number of non-dominated solutions to retain.
        """
        self.max_size = max_size
        self.solutions: List[EvaluationResult] = []

    def __len__(self) -> int:
        return len(self.solutions)

    def add(self, candidate: EvaluationResult) -> bool:
        """
        Attempt to add a candidate evaluation result to the archive.

        Args:
            candidate: Candidate EvaluationResult.

        Returns:
            True if candidate was added to the archive, False otherwise.
        """
        # Duplicate check on decision vector or objectives
        for s in self.solutions:
            if np.array_equal(candidate.x, s.x):
                return False
            if np.allclose(candidate.objectives, s.objectives, atol=1e-5):
                return False

        # Domination test against existing solutions
        survivors: List[EvaluationResult] = []
        for s in self.solutions:
            # If s dominates candidate, candidate is rejected
            if constrained_dominates(s.objectives_min, s.total_violation, candidate.objectives_min, candidate.total_violation):
                return False
            # If candidate dominates s, s is discarded
            if constrained_dominates(candidate.objectives_min, candidate.total_violation, s.objectives_min, s.total_violation):
                continue
            survivors.append(s)

        survivors.append(candidate)
        self.solutions = survivors

        # Prune if over capacity
        if len(self.solutions) > self.max_size:
            self._prune()

        return True

    def add_batch(self, candidates: List[EvaluationResult]) -> int:
        """Add multiple candidates; returns count of successful additions."""
        added = 0
        for cand in candidates:
            if self.add(cand):
                added += 1
        return added

    def _prune(self) -> None:
        """Prune archive down to max_size using crowding distance."""
        # Separate feasible and infeasible
        feas = [s for s in self.solutions if s.is_feasible]
        if len(feas) >= self.max_size:
            self.solutions = feas
        elif len(feas) > 0:
            # Keep all feasible, prune from infeasible by lowest violation
            infeas = [s for s in self.solutions if not s.is_feasible]
            infeas.sort(key=lambda s: s.total_violation)
            self.solutions = feas + infeas[: (self.max_size - len(feas))]
            return

        objs = np.array([s.objectives_min for s in self.solutions], dtype=np.float64)
        front_indices = list(range(len(self.solutions)))
        distances = compute_crowding_distance(objs, front_indices)

        # Sort descending by distance (keep boundary & high crowding distance)
        ranked_indices = np.argsort(-distances)
        keep_indices = ranked_indices[: self.max_size]
        self.solutions = [self.solutions[i] for i in sorted(keep_indices)]

    def get_objective_matrix(self, minimization: bool = False) -> np.ndarray:
        """
        Return 2D numpy array of objectives of all archive solutions.

        Args:
            minimization: If True, returns [-f1, f2, f3, -f4]. If False, returns [f1, f2, f3, f4].
        """
        if not self.solutions:
            return np.empty((0, 4), dtype=np.float64)
        if minimization:
            return np.array([s.objectives_min for s in self.solutions], dtype=np.float64)
        return np.array([s.objectives for s in self.solutions], dtype=np.float64)

    def get_feasible_solutions(self) -> List[EvaluationResult]:
        """Return all feasible solutions in the archive."""
        return [s for s in self.solutions if s.is_feasible]
