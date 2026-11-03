"""Opt-in corner-seeking extension of the repository's NSGA-II optimizer."""

from __future__ import annotations

from typing import List

import numpy as np

from src.algorithms.nsga2 import NSGA2Optimizer, OptimizationResult
from src.multiobjective.archive import ParetoArchive
from src.problem import EvaluationResult, NBAProblem


class CornerNSGA2(NSGA2Optimizer):
    """NSGA-II with optional heuristic extreme seeds and corner local search."""

    def __init__(
        self,
        problem: NBAProblem,
        pop_size: int = 100,
        max_generations: int = 120,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        seed: int = 42,
        apply_repair: bool = True,
        extreme_seeding: bool = False,
        corner_local_search: bool = False,
        corner_frequency: int = 10,
        corner_moves: int = 30,
    ) -> None:
        super().__init__(
            problem, pop_size, max_generations, crossover_prob, mutation_prob,
            seed, apply_repair
        )
        if corner_frequency < 1:
            raise ValueError("corner_frequency must be at least 1")
        if corner_moves < 1:
            raise ValueError("corner_moves must be at least 1")
        self.extreme_seeding = extreme_seeding
        self.corner_local_search = corner_local_search
        self.corner_frequency = corner_frequency
        self.corner_moves = corner_moves

    def _greedy_extreme_plan(self, objective: int) -> np.ndarray:
        plan = np.zeros(self.problem.num_actions, dtype=np.int32)
        actions = list(self.problem.actions)

        def rank(action: object) -> float:
            item = action
            gain = float(self.problem.gains[item.index])
            if objective == 0:
                return gain / (float(self.problem.costs[item.index]) + 0.1)
            if objective == 1:
                return -float(self.problem.costs[item.index])
            if objective == 2:
                return -float(self.problem.hours[item.index])
            return gain / (float(self.problem.hours[item.index]) + 0.1)

        for action in sorted(actions, key=rank, reverse=True):
            if not action.is_schedulable:
                continue
            plan[action.index] = action.max_units
            for prerequisite in action.prereq_indices:
                plan[prerequisite] = max(plan[prerequisite], 1)
        return self.problem.repair(plan, self.rng)

    def _initial_population(self) -> List[np.ndarray]:
        if not self.extreme_seeding:
            return super()._initial_population()
        seeds = [self._greedy_extreme_plan(objective) for objective in range(4)]
        heuristic_candidates: List[tuple[tuple[float, float], np.ndarray]] = []
        for _ in range(64):
            candidate = np.zeros(self.problem.num_actions, dtype=np.int32)
            for action in self.problem.actions:
                if action.is_schedulable and self.rng.random() < self.rng.random():
                    candidate[action.index] = self.rng.integers(1, action.max_units + 1)
            candidate = self.problem.repair(
                self.problem._legacy_repair(candidate, self.rng), self.rng
            )
            scores = np.minimum(
                self.problem.M,
                self.problem.s0 + self.problem.criterion_action_mask @ candidate,
            )
            total_score = float(np.sum(scores))
            deficit = float(np.sum(np.maximum(0.0, self.problem.L - scores)))
            target_deficit = max(0.0, self.problem.min_accreditation_score - total_score)
            heuristic_candidates.append(((deficit + target_deficit, -total_score), candidate))
        seeds.append(min(heuristic_candidates, key=lambda item: item[0])[1])
        perturbations = max(1, int(round(self.pop_size * 0.10)))
        seeded: List[np.ndarray] = list(seeds[: self.pop_size])
        for index in range(perturbations):
            base = seeds[index % len(seeds)].copy()
            action_index = int(self.rng.integers(0, self.problem.num_actions))
            if self.problem.actions[action_index].is_schedulable:
                base[action_index] = min(
                    self.problem.upper_bounds[action_index],
                    base[action_index] + int(self.rng.integers(1, 3)),
                )
            seeded.append(self.problem.repair(base, self.rng))
        while len(seeded) < self.pop_size:
            seeded.append(self.problem.random_solution(self.rng, repair=self.apply_repair))
        return seeded[: self.pop_size]

    def _generation_extra_evaluations(
        self,
        pop: List[np.ndarray],
        eval_results: List[EvaluationResult],
        archive: ParetoArchive,
        generation: int,
        max_extra: int,
    ) -> List[EvaluationResult]:
        if not self.corner_local_search or generation % self.corner_frequency != 0:
            return []
        archive_solutions = archive.get_feasible_solutions()
        if not archive_solutions:
            return []

        extra: List[EvaluationResult] = []
        for objective in range(4):
            if len(extra) >= max_extra:
                break
            current = min(
                archive_solutions,
                key=lambda solution: solution.objectives_min[objective],
            )
            for _ in range(self.corner_moves):
                if len(extra) >= max_extra:
                    break
                candidates: List[np.ndarray] = []
                action_index = int(self.rng.integers(0, self.problem.num_actions))
                for delta in (-1, 1):
                    candidate = current.x.copy()
                    candidate[action_index] = max(
                        0,
                        min(
                            self.problem.upper_bounds[action_index],
                            candidate[action_index] + delta,
                        ),
                    )
                    candidates.append(self.problem.repair(candidate, self.rng))
                candidate = current.x.copy()
                other_index = int(self.rng.integers(0, self.problem.num_actions))
                candidate[action_index] = 0
                candidate[other_index] = min(
                    self.problem.upper_bounds[other_index], 1
                )
                candidates.append(self.problem.repair(candidate, self.rng))
                evaluations: List[EvaluationResult] = []
                for candidate in candidates:
                    if len(extra) + len(evaluations) >= max_extra:
                        break
                    evaluations.append(self.problem.evaluate(candidate))
                if not evaluations:
                    break
                best = min(evaluations, key=lambda solution: solution.objectives_min[objective])
                extra.extend(evaluations)
                if best.objectives_min[objective] < current.objectives_min[objective] - 1e-9:
                    current = best
                else:
                    break
        return extra

    def run(self) -> OptimizationResult:
        result = super().run()
        if result.pareto_solutions and all(
            solution.is_feasible for solution in result.pareto_solutions
        ):
            return result
        feasible = [solution for solution in result.final_population if solution.is_feasible]
        if feasible:
            result.pareto_solutions = feasible
        return result
