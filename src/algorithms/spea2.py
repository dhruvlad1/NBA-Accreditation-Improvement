"""
SPEA2 (Strength Pareto Evolutionary Algorithm 2) written from scratch (Zitzler et al., 2001).
Implements fine-grained fitness assignment, k-th nearest neighbor density estimation,
and iterative archive truncation for diversity preservation.
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
from src.multiobjective.dominance import constrained_dominates
from src.problem import EvaluationResult, NBAProblem


class SPEA2Optimizer:
    """
    SPEA2 multi-objective evolutionary algorithm written from scratch.
    """

    def __init__(
        self,
        problem: NBAProblem,
        pop_size: int = 100,
        archive_size: int = 100,
        max_generations: int = 120,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        seed: int = 42,
        apply_repair: bool = True
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.seed = seed
        self.apply_repair = apply_repair
        self.rng = np.random.default_rng(seed)

    def _compute_fitness(
        self,
        eval_results: List[EvaluationResult]
    ) -> np.ndarray:
        """
        Compute SPEA2 fitness: F(i) = RawFitness(i) + Density(i).
        Smaller fitness is better. Non-dominated individuals have F(i) < 1.0.
        """
        n = len(eval_results)
        if n == 0:
            return np.array([])

        obj_matrix = np.array([res.objectives_min for res in eval_results], dtype=np.float64)
        viols = np.array([res.total_violation for res in eval_results], dtype=np.float64)

        # 1. Strength S(i): number of solutions that i dominates
        strength = np.zeros(n, dtype=np.int32)
        domination_matrix = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(n):
                if i != j and constrained_dominates(obj_matrix[i], viols[i], obj_matrix[j], viols[j]):
                    domination_matrix[i, j] = True
                    strength[i] += 1

        # 2. Raw fitness R(i): sum of strengths of solutions that dominate i
        raw_fitness = np.zeros(n, dtype=np.float64)
        for i in range(n):
            dominators = np.where(domination_matrix[:, i])[0]
            raw_fitness[i] = float(np.sum(strength[dominators]))

        # 3. Density estimation: D(i) = 1 / (sigma_i^k + 2)
        # Normalize objectives to avoid objective scale bias in Euclidean distance
        min_vals = np.min(obj_matrix, axis=0)
        max_vals = np.max(obj_matrix, axis=0)
        ranges = np.where((max_vals - min_vals) > 1e-12, max_vals - min_vals, 1.0)
        norm_objs = (obj_matrix - min_vals) / ranges

        # Pairwise Euclidean distances
        diff = norm_objs[:, np.newaxis, :] - norm_objs[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Sort distances for each individual
        sorted_dists = np.sort(dist_matrix, axis=1)
        k = max(1, int(np.sqrt(n)))
        sigma_k = sorted_dists[:, k]
        density = 1.0 / (sigma_k + 2.0)

        total_fitness = raw_fitness + density
        return total_fitness

    def _environmental_selection(
        self,
        pop_solutions: List[EvaluationResult],
        pop_genes: List[np.ndarray],
        fitness: np.ndarray
    ) -> Tuple[List[EvaluationResult], List[np.ndarray]]:
        """
        Select archive_size individuals using SPEA2 environmental selection and truncation.
        """
        n = len(pop_solutions)
        non_dominated_indices = np.where(fitness < 1.0)[0].tolist()

        if len(non_dominated_indices) == self.archive_size:
            selected_indices = non_dominated_indices
        elif len(non_dominated_indices) < self.archive_size:
            # Fill with best dominated (lowest fitness)
            dominated_indices = np.where(fitness >= 1.0)[0]
            sorted_dominated = dominated_indices[np.argsort(fitness[dominated_indices])]
            needed = self.archive_size - len(non_dominated_indices)
            selected_indices = non_dominated_indices + sorted_dominated[:needed].tolist()
        else:
            # Truncation operator: iteratively remove individuals with closest neighbors
            selected = list(non_dominated_indices)
            obj_matrix = np.array([pop_solutions[i].objectives_min for i in selected], dtype=np.float64)

            min_v = np.min(obj_matrix, axis=0)
            max_v = np.max(obj_matrix, axis=0)
            rngs = np.where((max_v - min_v) > 1e-12, max_v - min_v, 1.0)
            norm_objs = (obj_matrix - min_v) / rngs

            # Distance matrix within non-dominated set
            diff = norm_objs[:, np.newaxis, :] - norm_objs[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)

            while len(selected) > self.archive_size:
                # Find pair with minimum distance
                sorted_dists = np.sort(dists, axis=1)
                # Lexicographically sort by smallest distance to any neighbor
                min_neighbor_dist = sorted_dists[:, 0]
                worst_idx = int(np.argmin(min_neighbor_dist))

                selected.pop(worst_idx)
                dists = np.delete(dists, worst_idx, axis=0)
                dists = np.delete(dists, worst_idx, axis=1)

            selected_indices = selected

        archive_solutions = [pop_solutions[i] for i in selected_indices]
        archive_genes = [pop_genes[i] for i in selected_indices]
        return archive_solutions, archive_genes

    def run(self) -> OptimizationResult:
        """Execute the SPEA2 search."""
        start_time = time.time()
        archive_tracker = ParetoArchive(max_size=self.archive_size)
        history: List[Dict[str, Any]] = []

        # 1. Initialize population P0 and archive P_bar_0
        pop_genes: List[np.ndarray] = [
            self.problem.random_solution(self.rng, repair=self.apply_repair)
            for _ in range(self.pop_size)
        ]
        pop_evals = self.problem.evaluate_batch(np.array(pop_genes))
        total_evals = len(pop_evals)
        archive_tracker.add_batch(pop_evals)

        archive_genes: List[np.ndarray] = [
            self.problem.random_solution(self.rng, repair=self.apply_repair)
            for _ in range(self.archive_size)
        ]
        archive_evals = self.problem.evaluate_batch(np.array(archive_genes))
        total_evals += len(archive_evals)
        archive_tracker.add_batch(archive_evals)

        for gen in range(self.max_generations):
            # Combine P_t and archive
            union_evals = pop_evals + archive_evals
            union_genes = pop_genes + archive_genes

            # Fitness assignment
            fitness = self._compute_fitness(union_evals)

            # Environmental selection to form next archive
            archive_evals, archive_genes = self._environmental_selection(
                union_evals, union_genes, fitness
            )
            archive_tracker.add_batch(archive_evals)

            # Snapshot
            feasible_count = sum(1 for res in archive_evals if res.is_feasible)
            f1_max = max(res.f1_total_score for res in archive_evals)
            f2_min = min(res.f2_cost_lakh for res in archive_evals)
            history.append({
                "generation": gen,
                "feasible_count": feasible_count,
                "f1_max": f1_max,
                "f2_min": f2_min,
                "archive_size": len(archive_tracker),
            })

            # Mating selection (binary tournament on SPEA2 fitness)
            archive_fitness = self._compute_fitness(archive_evals)
            next_pop_genes: List[np.ndarray] = []

            while len(next_pop_genes) < self.pop_size:
                # Tournament among archive members
                i1, i2 = self.rng.choice(len(archive_evals), size=2, replace=False)
                p1 = archive_genes[i1 if archive_fitness[i1] <= archive_fitness[i2] else i2]

                j1, j2 = self.rng.choice(len(archive_evals), size=2, replace=False)
                p2 = archive_genes[j1 if archive_fitness[j1] <= archive_fitness[j2] else j2]

                c1, c2 = integer_sbx_crossover(
                    p1, p2,
                    upper_bounds=self.problem.upper_bounds,
                    crossover_prob=self.crossover_prob,
                    rng=self.rng
                )
                c1 = integer_unit_flip_mutation(c1, self.problem.upper_bounds, self.mutation_prob, self.rng)
                c2 = integer_unit_flip_mutation(c2, self.problem.upper_bounds, self.mutation_prob, self.rng)

                if self.apply_repair:
                    c1 = self.problem.repair(c1, self.rng)
                    c2 = self.problem.repair(c2, self.rng)

                next_pop_genes.append(c1)
                if len(next_pop_genes) < self.pop_size:
                    next_pop_genes.append(c2)

            pop_genes = next_pop_genes
            pop_evals = self.problem.evaluate_batch(np.array(pop_genes))
            total_evals += len(pop_evals)

        elapsed = time.time() - start_time
        pareto_solutions = archive_tracker.get_feasible_solutions()
        if not pareto_solutions:
            archive_evals.sort(key=lambda s: s.total_violation)
            pareto_solutions = [archive_evals[0]]

        return OptimizationResult(
            algorithm_name="SPEA2",
            seed=self.seed,
            elapsed_time=elapsed,
            final_population=archive_evals,
            pareto_solutions=pareto_solutions,
            history=history,
            total_evaluations=total_evals,
        )
