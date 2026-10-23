"""
Greedy heuristic baseline for NBA Accreditation Planning.
Prioritizes actions based on multi-criteria efficiency (marginal score gain / cost,
marginal score gain / hours, and criterion deficit weighting).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
import numpy as np

from src.algorithms.nsga2 import OptimizationResult
from src.multiobjective.archive import ParetoArchive
from src.problem import EvaluationResult, NBAProblem


class GreedyBaseline:
    """
    Greedy heuristic generating diverse solutions across the cost-effort spectrum.
    """

    def __init__(self, problem: NBAProblem, seed: int = 42) -> None:
        self.problem = problem
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self, n_samples: int = 100) -> OptimizationResult:
        """
        Run greedy heuristic across multiple preference weightings.

        Args:
            n_samples: Number of greedy variants to evaluate.
        """
        start_time = time.time()
        archive = ParetoArchive(max_size=100)
        all_evals: List[EvaluationResult] = []

        # Sweep cost vs hour preference alpha in [0, 1]
        alphas = np.linspace(0.0, 1.0, 25)
        # Sweep target budget caps from 10 to B
        budget_caps = np.linspace(10.0, self.problem.budget, 4)

        for alpha in alphas:
            for b_cap in budget_caps:
                x = np.zeros(self.problem.num_actions, dtype=np.int32)

                # 1. First ensure all evidence packages for criteria below L_c or initial pass
                for c in self.problem.criteria:
                    for a_idx in c.action_indices:
                        if self.problem.actions[a_idx].is_evidence and self.problem.actions[a_idx].is_schedulable:
                            x[a_idx] = 1

                # 2. Greedily add actions by weighted efficiency:
                # efficiency = alpha * (gain / cost) + (1 - alpha) * (gain / (hours/50))
                # multiplied by deficit priority if criterion is below target
                while True:
                    scores = np.minimum(self.problem.M, self.problem.s0 + self.problem.criterion_action_mask @ x)
                    cost = float(np.dot(self.problem.costs, x))
                    monthly = self.problem.schedule_matrix @ x

                    best_idx = -1
                    best_score = -1.0

                    for a in self.problem.actions:
                        if not a.is_schedulable or x[a.index] >= a.max_units:
                            continue
                        if not all(x[p] >= 1 for p in a.prereq_indices):
                            continue
                        if cost + a.cost_lakh > b_cap:
                            continue

                        # Check monthly capacity
                        a_rate = a.faculty_hours / max(a.duration_months, 1)
                        new_monthly = monthly.copy()
                        for m in range(1, self.problem.planning_horizon + 1):
                            if a.earliest_start < m <= a.earliest_finish:
                                new_monthly[m - 1] += a_rate
                        if np.any(new_monthly > self.problem.monthly_capacity):
                            continue

                        # Marginal gain
                        c_idx = a.crit_id - 1
                        curr_s = scores[c_idx]
                        potential_s = min(self.problem.M[c_idx], curr_s + a.score_gain)
                        gain = potential_s - curr_s
                        if gain <= 1e-4:
                            continue

                        cost_eff = gain / (a.cost_lakh + 0.1)
                        hour_eff = gain / (a.faculty_hours / 40.0 + 0.1)
                        score = alpha * cost_eff + (1.0 - alpha) * hour_eff

                        # Prioritize criteria with lower score/max ratio
                        deficit_mult = 2.0 - (scores[c_idx] / self.problem.M[c_idx])
                        score *= deficit_mult

                        if score > best_score:
                            best_score = score
                            best_idx = a.index

                    if best_idx >= 0:
                        x[best_idx] += 1
                    else:
                        break

                x_repaired = self.problem.repair(x, self.rng)
                eval_res = self.problem.evaluate(x_repaired)
                all_evals.append(eval_res)
                archive.add(eval_res)

        elapsed = time.time() - start_time
        pareto_sols = archive.get_feasible_solutions()
        if not pareto_sols:
            all_evals.sort(key=lambda s: s.total_violation)
            pareto_sols = [all_evals[0]]

        return OptimizationResult(
            algorithm_name="Greedy-Baseline",
            seed=self.seed,
            elapsed_time=elapsed,
            final_population=all_evals,
            pareto_solutions=pareto_sols,
            history=[],
            total_evaluations=len(all_evals),
        )
