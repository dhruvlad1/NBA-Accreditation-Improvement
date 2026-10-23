"""
MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) written from scratch.
Implements Tchebycheff decomposition, Das-Dennis weight vector generation,
neighborhood mating, and external Pareto archive tracking.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.algorithms.nsga2 import OptimizationResult
from src.algorithms.operators import (
    integer_sbx_crossover,
    integer_unit_flip_mutation,
)
from src.multiobjective.archive import ParetoArchive
from src.multiobjective.weights import generate_weight_vectors
from src.problem import EvaluationResult, NBAProblem


class MOEADOptimizer:
    """
    MOEA/D with Tchebycheff scalarization and constraint penalty.
    """

    def __init__(
        self,
        problem: NBAProblem,
        pop_size: int = 100,
        n_neighbors: int = 15,
        max_generations: int = 120,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        max_replacements: int = 2,
        seed: int = 42,
        apply_repair: bool = True
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.n_neighbors = min(n_neighbors, pop_size)
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_replacements = max_replacements
        self.seed = seed
        self.apply_repair = apply_repair
        self.rng = np.random.default_rng(seed)

    def _tchebycheff(
        self,
        objs_min: np.ndarray,
        weight: np.ndarray,
        z_ideal: np.ndarray,
        violation: float
    ) -> float:
        """
        Penalty-based Tchebycheff scalar function.
        g_te = max_j ( weight_j * |f_j - z_j*| ) + 10^4 * violation
        """
        diff = np.abs(objs_min - z_ideal)
        te_val = float(np.max(weight * diff))
        return te_val + 10000.0 * violation

    def run(self) -> OptimizationResult:
        """Execute MOEA/D search."""
        start_time = time.time()
        num_objs = 4
        archive = ParetoArchive(max_size=self.pop_size)
        history: List[Dict[str, Any]] = []

        # 1. Generate weight vectors on 4D simplex
        weights = generate_weight_vectors(num_objs, self.pop_size, rng=self.rng)
        actual_pop_size = len(weights)
        self.n_neighbors = min(self.n_neighbors, actual_pop_size)

        # 2. Compute Euclidean distance matrix of weights to find neighborhoods
        diff = weights[:, np.newaxis, :] - weights[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
        neighborhoods = np.argsort(dist_mat, axis=1)[:, : self.n_neighbors]

        # 3. Initialize population and ideal point
        pop_genes: List[np.ndarray] = [
            self.problem.random_solution(self.rng, repair=self.apply_repair)
            for _ in range(actual_pop_size)
        ]
        pop_evals = self.problem.evaluate_batch(np.array(pop_genes))
        total_evals = len(pop_evals)
        archive.add_batch(pop_evals)

        # Ideal point z* for minimization
        all_objs = np.array([res.objectives_min for res in pop_evals], dtype=np.float64)
        z_ideal = np.min(all_objs, axis=0)

        for gen in range(self.max_generations):
            feasible_count = sum(1 for res in pop_evals if res.is_feasible)
            f1_max = max(res.f1_total_score for res in pop_evals)
            f2_min = min(res.f2_cost_lakh for res in pop_evals)
            history.append({
                "generation": gen,
                "feasible_count": feasible_count,
                "f1_max": f1_max,
                "f2_min": f2_min,
                "archive_size": len(archive),
            })

            # Iterate over each subproblem
            for i in range(actual_pop_size):
                # Select two random parents from neighborhood B(i)
                neigh = neighborhoods[i]
                p1_idx, p2_idx = self.rng.choice(neigh, size=2, replace=False)
                p1 = pop_genes[p1_idx]
                p2 = pop_genes[p2_idx]

                # Crossover
                c1, _ = integer_sbx_crossover(
                    p1, p2,
                    upper_bounds=self.problem.upper_bounds,
                    crossover_prob=self.crossover_prob,
                    rng=self.rng
                )
                # Mutation
                c1 = integer_unit_flip_mutation(c1, self.problem.upper_bounds, self.mutation_prob, self.rng)
                # Repair
                if self.apply_repair:
                    c1 = self.problem.repair(c1, self.rng)

                # Evaluate offspring
                offspring_res = self.problem.evaluate(c1)
                total_evals += 1
                archive.add(offspring_res)

                # Update ideal point
                z_ideal = np.minimum(z_ideal, offspring_res.objectives_min)

                # Update neighboring solutions (up to max_replacements)
                replacements = 0
                # Shuffle neighbors to avoid directional bias
                shuffled_neigh = self.rng.permutation(neigh)
                for j in shuffled_neigh:
                    current_res = pop_evals[j]
                    current_te = self._tchebycheff(
                        current_res.objectives_min, weights[j], z_ideal, current_res.total_violation
                    )
                    offspring_te = self._tchebycheff(
                        offspring_res.objectives_min, weights[j], z_ideal, offspring_res.total_violation
                    )

                    if offspring_te <= current_te:
                        pop_genes[j] = c1.copy()
                        pop_evals[j] = offspring_res
                        replacements += 1
                        if replacements >= self.max_replacements:
                            break

        elapsed = time.time() - start_time
        pareto_solutions = archive.get_feasible_solutions()
        if not pareto_solutions:
            pop_evals.sort(key=lambda s: s.total_violation)
            pareto_solutions = [pop_evals[0]]

        return OptimizationResult(
            algorithm_name="MOEA/D",
            seed=self.seed,
            elapsed_time=elapsed,
            final_population=pop_evals,
            pareto_solutions=pareto_solutions,
            history=history,
            total_evaluations=total_evals,
        )
