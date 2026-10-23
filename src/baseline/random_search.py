"""
Random Search / Monte Carlo baseline for multi-objective NBA problem.
Samples random integer configurations and constructs the empirical Pareto front.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
import numpy as np

from src.algorithms.nsga2 import OptimizationResult
from src.multiobjective.archive import ParetoArchive
from src.problem import EvaluationResult, NBAProblem


class RandomSearchBaseline:
    """
    Monte Carlo random search baseline with constraint repair.
    """

    def __init__(self, problem: NBAProblem, seed: int = 42) -> None:
        self.problem = problem
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self, n_samples: int = 500) -> OptimizationResult:
        """Run random sampling."""
        start_time = time.time()
        archive = ParetoArchive(max_size=100)
        evals: List[EvaluationResult] = []

        for _ in range(n_samples):
            raw_x = self.problem.random_solution(self.rng, repair=True)
            res = self.problem.evaluate(raw_x)
            evals.append(res)
            archive.add(res)

        elapsed = time.time() - start_time
        pareto_sols = archive.get_feasible_solutions()
        if not pareto_sols:
            evals.sort(key=lambda s: s.total_violation)
            pareto_sols = [evals[0]]

        return OptimizationResult(
            algorithm_name="Random-Baseline",
            seed=self.seed,
            elapsed_time=elapsed,
            final_population=evals,
            pareto_solutions=pareto_sols,
            history=[],
            total_evaluations=n_samples,
        )
