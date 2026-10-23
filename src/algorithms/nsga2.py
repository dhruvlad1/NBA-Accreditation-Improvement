"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) written from scratch.
Implements fast non-dominated sorting, crowding distance assignment,
Deb's constrained dominance, and integer genetic variation with domain repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.algorithms.operators import (
    binary_tournament_selection,
    integer_sbx_crossover,
    integer_unit_flip_mutation,
)
from src.multiobjective.archive import ParetoArchive
from src.multiobjective.crowding import compute_crowding_distance
from src.multiobjective.dominance import fast_non_dominated_sort
from src.problem import EvaluationResult, NBAProblem


@dataclass
class OptimizationResult:
    """Standardized result returned by any multi-objective optimizer."""
    algorithm_name: str
    seed: int
    elapsed_time: float
    final_population: List[EvaluationResult]
    pareto_solutions: List[EvaluationResult]
    history: List[Dict[str, Any]]
    total_evaluations: int


class NSGA2Optimizer:
    """
    NSGA-II algorithm implemented from scratch for integer-constrained optimization.
    """

    def __init__(
        self,
        problem: NBAProblem,
        pop_size: int = 100,
        max_generations: int = 120,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        seed: int = 42,
        apply_repair: bool = True
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.seed = seed
        self.apply_repair = apply_repair
        self.rng = np.random.default_rng(seed)

    def run(self) -> OptimizationResult:
        """Execute the NSGA-II search."""
        start_time = time.time()
        archive = ParetoArchive(max_size=self.pop_size)
        history: List[Dict[str, Any]] = []

        # 1. Initialize population
        pop: List[np.ndarray] = []
        for _ in range(self.pop_size):
            sol = self.problem.random_solution(self.rng, repair=self.apply_repair)
            pop.append(sol)

        eval_results = self.problem.evaluate_batch(np.array(pop))
        total_evals = len(eval_results)
        archive.add_batch(eval_results)

        for gen in range(self.max_generations):
            obj_matrix = np.array([res.objectives_min for res in eval_results], dtype=np.float64)
            violations = np.array([res.total_violation for res in eval_results], dtype=np.float64)

            # Fast non-dominated sort with constrained dominance
            fronts = fast_non_dominated_sort(obj_matrix, violations=violations)

            # Compute ranks and crowding distances
            ranks = np.zeros(len(eval_results), dtype=np.int32)
            crowding = np.zeros(len(eval_results), dtype=np.float64)
            for rank_idx, front in enumerate(fronts):
                if len(front) == 0:
                    continue
                ranks[front] = rank_idx
                dist = compute_crowding_distance(obj_matrix, front)
                crowding[front] = dist

            # Record generation snapshot
            feasible_count = sum(1 for res in eval_results if res.is_feasible)
            f1_max = max(res.f1_total_score for res in eval_results)
            f2_min = min(res.f2_cost_lakh for res in eval_results)
            history.append({
                "generation": gen,
                "feasible_count": feasible_count,
                "f1_max": f1_max,
                "f2_min": f2_min,
                "archive_size": len(archive),
            })

            # Offspring generation
            offspring_pop: List[np.ndarray] = []
            while len(offspring_pop) < self.pop_size:
                # Tournament selection
                p1_idx = binary_tournament_selection(ranks, crowding, self.rng)
                p2_idx = binary_tournament_selection(ranks, crowding, self.rng)
                parent1 = pop[p1_idx]
                parent2 = pop[p2_idx]

                # Crossover
                c1, c2 = integer_sbx_crossover(
                    parent1, parent2,
                    upper_bounds=self.problem.upper_bounds,
                    crossover_prob=self.crossover_prob,
                    rng=self.rng
                )

                # Mutation
                c1 = integer_unit_flip_mutation(c1, self.problem.upper_bounds, self.mutation_prob, self.rng)
                c2 = integer_unit_flip_mutation(c2, self.problem.upper_bounds, self.mutation_prob, self.rng)

                # Repair
                if self.apply_repair:
                    c1 = self.problem.repair(c1, self.rng)
                    c2 = self.problem.repair(c2, self.rng)

                offspring_pop.append(c1)
                if len(offspring_pop) < self.pop_size:
                    offspring_pop.append(c2)

            offspring_evals = self.problem.evaluate_batch(np.array(offspring_pop))
            total_evals += len(offspring_evals)
            archive.add_batch(offspring_evals)

            # Combined pool (2N)
            combined_pop = pop + offspring_pop
            combined_evals = eval_results + offspring_evals

            combined_objs = np.array([res.objectives_min for res in combined_evals], dtype=np.float64)
            combined_viols = np.array([res.total_violation for res in combined_evals], dtype=np.float64)

            # Sort combined 2N pool
            combined_fronts = fast_non_dominated_sort(combined_objs, violations=combined_viols)

            # Environmental selection: select N individuals
            next_pop: List[np.ndarray] = []
            next_evals: List[EvaluationResult] = []

            for front in combined_fronts:
                if len(next_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        next_pop.append(combined_pop[idx])
                        next_evals.append(combined_evals[idx])
                else:
                    # Truncate splitting front based on crowding distance
                    needed = self.pop_size - len(next_pop)
                    front_dist = compute_crowding_distance(combined_objs, front)
                    sorted_front = [front[i] for i in np.argsort(-front_dist)]
                    for idx in sorted_front[:needed]:
                        next_pop.append(combined_pop[idx])
                        next_evals.append(combined_evals[idx])
                    break

            pop = next_pop
            eval_results = next_evals

        elapsed = time.time() - start_time
        pareto_solutions = archive.get_feasible_solutions()
        if not pareto_solutions:
            # Fallback if no feasible found in archive: pick least violating from final pop
            eval_results.sort(key=lambda s: s.total_violation)
            pareto_solutions = [eval_results[0]]

        return OptimizationResult(
            algorithm_name="NSGA-II",
            seed=self.seed,
            elapsed_time=elapsed,
            final_population=eval_results,
            pareto_solutions=pareto_solutions,
            history=history,
            total_evaluations=total_evals,
        )
