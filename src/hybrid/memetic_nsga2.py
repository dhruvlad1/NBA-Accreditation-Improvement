"""
Hybrid Memetic Multi-Objective Algorithm (Memetic NSGA-II).
Combines NSGA-II with:
1. Exact MILP warm-start seeding (anchors for f1, f2, f3, f4).
2. Domain-specific local search (surplus trimming and bottleneck criterion boost).
3. Repair-guided variation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.algorithms.nsga2 import OptimizationResult
from src.algorithms.operators import (
    binary_tournament_selection,
    integer_sbx_crossover,
    integer_unit_flip_mutation,
)
from src.baseline.milp_solver import HiGHSMILPSolver
from src.multiobjective.archive import ParetoArchive
from src.multiobjective.crowding import compute_crowding_distance
from src.multiobjective.dominance import constrained_dominates, fast_non_dominated_sort
from src.problem import EvaluationResult, NBAProblem


class MemeticNSGA2Optimizer:
    """
    Hybrid Memetic NSGA-II with exact MILP seeding and localized Pareto refinement.
    """

    def __init__(
        self,
        problem: NBAProblem,
        pop_size: int = 100,
        max_generations: int = 120,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        local_search_freq: int = 5,
        local_search_candidates: int = 5,
        warm_start_milp: bool = True,
        seed: int = 42
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.local_search_freq = local_search_freq
        self.local_search_candidates = local_search_candidates
        self.warm_start_milp = warm_start_milp
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _local_search(self, x: np.ndarray) -> np.ndarray:
        """
        Domain-aware local refinement:
        1. Trims surplus points: if a criterion has exceeded max marks M_c, reduce units.
        2. Bottleneck criterion boost: identify worst criterion and attempt to add eligible actions.
        """
        sol = x.copy()
        eval_before = self.problem.evaluate(sol)

        # 1. Check for criterion saturation (surplus waste)
        raw_scores = self.problem.s0 + self.problem.criterion_action_mask @ sol
        for c in self.problem.criteria:
            c_idx = c.id - 1
            if raw_scores[c_idx] > c.max_score:
                # We have wasted points; try decrementing non-evidence actions
                for a_idx in c.action_indices:
                    a = self.problem.actions[a_idx]
                    if not a.is_evidence and sol[a_idx] > 0:
                        sol[a_idx] -= 1
                        eval_test = self.problem.evaluate(sol)
                        if eval_test.is_feasible and eval_test.f1_total_score >= eval_before.f1_total_score - 1e-4:
                            eval_before = eval_test
                        else:
                            sol[a_idx] += 1

        # 2. Bottleneck criterion boost
        eval_mid = self.problem.evaluate(sol)
        worst_c_idx = int(np.argmin(eval_mid.criteria_ratios))
        target_criterion = self.problem.criteria[worst_c_idx]

        for a_idx in target_criterion.action_indices:
            a = self.problem.actions[a_idx]
            if a.is_schedulable and sol[a_idx] < a.max_units and all(sol[p] >= 1 for p in a.prereq_indices):
                sol[a_idx] += 1
                sol_repaired = self.problem.repair(sol, self.rng)
                eval_test = self.problem.evaluate(sol_repaired)
                if eval_test.is_feasible:
                    # Accept if it dominates or improves worst ratio without losing score
                    if constrained_dominates(eval_test.objectives_min, eval_test.total_violation, eval_before.objectives_min, eval_before.total_violation):
                        return sol_repaired
                    if eval_test.f4_worst_criterion_ratio > eval_before.f4_worst_criterion_ratio and eval_test.f1_total_score >= eval_before.f1_total_score:
                        return sol_repaired
                sol[a_idx] -= 1

        return sol

    def run(self) -> OptimizationResult:
        """Execute Hybrid Memetic NSGA-II search."""
        start_time = time.time()
        archive = ParetoArchive(max_size=self.pop_size)
        history: List[Dict[str, Any]] = []

        pop: List[np.ndarray] = []

        # 1. Optional warm-start seeding via HiGHS MILP anchors
        if self.warm_start_milp:
            milp_solver = HiGHSMILPSolver(self.problem)
            for obj in ["max_f1", "min_f2", "min_f3", "max_f4"]:
                res = milp_solver.solve_single_objective(obj)
                if res and res.is_feasible:
                    pop.append(res.x.copy())
                    archive.add(res)

        # Fill remaining population
        while len(pop) < self.pop_size:
            pop.append(self.problem.random_solution(self.rng, repair=True))

        eval_results = self.problem.evaluate_batch(np.array(pop))
        total_evals = len(eval_results)
        archive.add_batch(eval_results)

        for gen in range(self.max_generations):
            obj_matrix = np.array([res.objectives_min for res in eval_results], dtype=np.float64)
            violations = np.array([res.total_violation for res in eval_results], dtype=np.float64)

            fronts = fast_non_dominated_sort(obj_matrix, violations=violations)

            ranks = np.zeros(len(eval_results), dtype=np.int32)
            crowding = np.zeros(len(eval_results), dtype=np.float64)
            for rank_idx, front in enumerate(fronts):
                if len(front) == 0:
                    continue
                ranks[front] = rank_idx
                dist = compute_crowding_distance(obj_matrix, front)
                crowding[front] = dist

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
                p1_idx = binary_tournament_selection(ranks, crowding, self.rng)
                p2_idx = binary_tournament_selection(ranks, crowding, self.rng)
                p1 = pop[p1_idx]
                p2 = pop[p2_idx]

                c1, c2 = integer_sbx_crossover(
                    p1, p2,
                    upper_bounds=self.problem.upper_bounds,
                    crossover_prob=self.crossover_prob,
                    rng=self.rng
                )
                c1 = integer_unit_flip_mutation(c1, self.problem.upper_bounds, self.mutation_prob, self.rng)
                c2 = integer_unit_flip_mutation(c2, self.problem.upper_bounds, self.mutation_prob, self.rng)

                c1 = self.problem.repair(c1, self.rng)
                c2 = self.problem.repair(c2, self.rng)

                offspring_pop.append(c1)
                if len(offspring_pop) < self.pop_size:
                    offspring_pop.append(c2)

            # Periodic local search on top rank-0 individuals
            if gen % self.local_search_freq == 0 and len(fronts) > 0:
                first_front = fronts[0]
                n_ls = min(self.local_search_candidates, len(first_front))
                chosen = self.rng.choice(first_front, size=n_ls, replace=False)
                for idx in chosen:
                    refined = self._local_search(pop[idx])
                    offspring_pop.append(refined)

            offspring_evals = self.problem.evaluate_batch(np.array(offspring_pop))
            total_evals += len(offspring_evals)
            archive.add_batch(offspring_evals)

            # Combine and truncate
            combined_pop = pop + offspring_pop
            combined_evals = eval_results + offspring_evals

            combined_objs = np.array([res.objectives_min for res in combined_evals], dtype=np.float64)
            combined_viols = np.array([res.total_violation for res in combined_evals], dtype=np.float64)

            combined_fronts = fast_non_dominated_sort(combined_objs, violations=combined_viols)

            next_pop: List[np.ndarray] = []
            next_evals: List[EvaluationResult] = []

            for front in combined_fronts:
                if len(next_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        next_pop.append(combined_pop[idx])
                        next_evals.append(combined_evals[idx])
                else:
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
            eval_results.sort(key=lambda s: s.total_violation)
            pareto_solutions = [eval_results[0]]

        return OptimizationResult(
            algorithm_name="Hybrid-Memetic",
            seed=self.seed,
            elapsed_time=elapsed,
            final_population=eval_results,
            pareto_solutions=pareto_solutions,
            history=history,
            total_evaluations=total_evals,
        )
