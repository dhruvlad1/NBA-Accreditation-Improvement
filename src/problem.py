"""
Problem formulation for NBA Accreditation Improvement Planning.
Multi-objective optimization under budget, faculty hour capacity, scheduling deadlines,
prerequisite dependencies, and accreditation threshold constraints.

All parameters are loaded dynamically from configuration files; no values are hardcoded.
All synthetic datasets and benchmarks are explicitly marked synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Action:
    """Represents a single synthetic NBA improvement action."""
    action_id: str
    index: int
    crit_id: int
    name: str
    is_evidence: bool
    cost_lakh: float
    faculty_hours: float
    duration_months: int
    prerequisites: Tuple[str, ...]
    prereq_indices: Tuple[int, ...]
    max_units: int
    score_gain: float
    earliest_start: int
    earliest_finish: int
    is_schedulable: bool


@dataclass(frozen=True)
class Criterion:
    """Represents one of the 10 NBA criteria."""
    id: int
    code: str
    name: str
    max_score: float
    baseline_score: float
    min_threshold: float
    action_indices: Tuple[int, ...]


@dataclass
class EvaluationResult:
    """Complete evaluation of a candidate decision vector x."""
    x: np.ndarray
    objectives: np.ndarray          # [f1 (max), f2 (min), f3 (min), f4 (max)]
    objectives_min: np.ndarray      # [-f1, f2, f3, -f4] for standard minimization algorithms
    criteria_scores: np.ndarray     # Score_c for c = 1..10
    criteria_ratios: np.ndarray     # Score_c / M_c
    monthly_hours: np.ndarray       # Faculty hours per month 1..T
    violations: Dict[str, float]    # Individual constraint violations (>= 0)
    total_violation: float          # Sum of all constraint violations
    is_feasible: bool               # True if total_violation == 0.0

    @property
    def f1_total_score(self) -> float:
        """Total NBA Score (maximize)."""
        return float(self.objectives[0])

    @property
    def f2_cost_lakh(self) -> float:
        """Total Cost in Lakh INR (minimize)."""
        return float(self.objectives[1])

    @property
    def f3_faculty_hours(self) -> float:
        """Total Faculty Hours (minimize)."""
        return float(self.objectives[2])

    @property
    def f4_worst_criterion_ratio(self) -> float:
        """Worst-off Criterion Score / Max Ratio (maximize)."""
        return float(self.objectives[3])


class NBAProblem:
    """
    Formal NBA Accreditation Improvement Planning Problem.

    Decision Variables:
        x_a in {0, ..., u_a} for each action a in {1, ..., N}.

    Score for Criterion c:
        Score_c(x) = min(M_c, s0_c + sum_{a in crit c} g_a * x_a)

    Objectives:
        f1(x) = sum_c Score_c(x)                       [max total NBA marks]
        f2(x) = sum_a q_a * x_a                        [min cost in Lakh INR]
        f3(x) = sum_a h_a * x_a                        [min total faculty hours]
        f4(x) = min_c (Score_c(x) / M_c)               [max worst-off criterion balance]

    Constraints:
        (1) f2(x) <= B                                 [Budget]
        (2) sum_{a: S_a < m <= F_a} (h_a/d_a)*x_a <= K_m for m in 1..T [Monthly faculty hours]
        (3) x_a = 0 if F_a > T                         [Deadline]
        (4) x_a <= u_a * x_b for every prereq b        [Prerequisite dependency]
        (5) Score_c(x) >= L_c for all c in 1..10       [Minimum criterion thresholds]
        (6) f1(x) >= R                                 [Accreditation total threshold]
    """

    def __init__(self, config_path: str | Path) -> None:
        """
        Initialize problem by loading config and action data.
        Never hardcodes parameters.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Base directories
        self.root_dir = self.config_path.parent.parent

        # Planning parameters
        params = self.config["planning_parameters"]
        self.budget: float = float(params["budget_lakh"])
        self.planning_horizon: int = int(params["planning_horizon_months"])
        self.monthly_capacity: float = float(params["monthly_faculty_capacity"])
        self.min_accreditation_score: float = float(params["min_accreditation_score"])

        # Load raw criteria
        self._load_criteria()

        # Load actions and compute earliest-start schedule
        self._load_actions()

        # Precompute index lookup arrays for fast vectorised evaluations
        self._precompute_structures()

    def _load_criteria(self) -> None:
        """Load 10 NBA criteria definitions from config."""
        criteria_list = self.config["criteria"]
        self.num_criteria = len(criteria_list)
        self.criteria_raw = criteria_list

        self.M: np.ndarray = np.array([c["max_score"] for c in criteria_list], dtype=np.float64)
        self.s0: np.ndarray = np.array([c["baseline_score"] for c in criteria_list], dtype=np.float64)
        self.L: np.ndarray = np.array([c["min_threshold"] for c in criteria_list], dtype=np.float64)

    def _load_actions(self) -> None:
        """Load action data from CSV and resolve prerequisites and schedule."""
        actions_rel_path = self.config["dataset_metadata"]["actions_file"]
        actions_csv_path = self.root_dir / actions_rel_path
        if not actions_csv_path.exists():
            raise FileNotFoundError(f"Actions CSV not found at {actions_csv_path}")

        df = pd.read_csv(actions_csv_path)
        self.num_actions = len(df)

        id_to_index: Dict[str, int] = {row["action_id"]: idx for idx, row in df.iterrows()}
        self.id_to_index = id_to_index

        # Parse prerequisites
        prereqs_by_idx: List[List[int]] = []
        for _, row in df.iterrows():
            raw_p = str(row["prerequisites"])
            if raw_p and raw_p.strip() and raw_p != "nan":
                p_ids = [p.strip() for p in raw_p.split(";") if p.strip()]
                p_idxs = [id_to_index[pid] for pid in p_ids if pid in id_to_index]
            else:
                p_idxs = []
            prereqs_by_idx.append(p_idxs)

        # Precompute Earliest Start & Earliest Finish Schedule:
        # F_a = d_a + max_{b in prereq(a)} F_b; S_a = F_a - d_a.
        # Compute in topological order / memoized recursion.
        durations = df["duration_months"].values.astype(int)
        finish_times = np.zeros(self.num_actions, dtype=int)

        def compute_finish(idx: int) -> int:
            if finish_times[idx] > 0:
                return finish_times[idx]
            prereq_finishes = [compute_finish(p) for p in prereqs_by_idx[idx]]
            max_prereq = max(prereq_finishes) if prereq_finishes else 0
            finish_times[idx] = durations[idx] + max_prereq
            return finish_times[idx]

        for i in range(self.num_actions):
            compute_finish(i)

        start_times = finish_times - durations

        # Build Action objects
        self.actions: List[Action] = []
        for idx, row in df.iterrows():
            f_a = int(finish_times[idx])
            s_a = int(start_times[idx])
            p_list = tuple(row["prerequisites"].split(";")) if (str(row["prerequisites"]) and str(row["prerequisites"]) != "nan") else ()
            p_indices = tuple(prereqs_by_idx[idx])

            action = Action(
                action_id=str(row["action_id"]),
                index=idx,
                crit_id=int(row["crit_id"]),
                name=str(row["name"]),
                is_evidence=bool(row["is_evidence"]),
                cost_lakh=float(row["cost_lakh"]),
                faculty_hours=float(row["faculty_hours"]),
                duration_months=int(row["duration_months"]),
                prerequisites=p_list,
                prereq_indices=p_indices,
                max_units=int(row["max_units"]),
                score_gain=float(row["score_gain"]),
                earliest_start=s_a,
                earliest_finish=f_a,
                is_schedulable=(f_a <= self.planning_horizon),
            )
            self.actions.append(action)

        # Build Criterion objects
        self.criteria: List[Criterion] = []
        for c_info in self.criteria_raw:
            c_id = int(c_info["id"])
            c_actions = tuple(a.index for a in self.actions if a.crit_id == c_id)
            crit = Criterion(
                id=c_id,
                code=str(c_info["code"]),
                name=str(c_info["name"]),
                max_score=float(c_info["max_score"]),
                baseline_score=float(c_info["baseline_score"]),
                min_threshold=float(c_info["min_threshold"]),
                action_indices=c_actions,
            )
            self.criteria.append(crit)

    def _precompute_structures(self) -> None:
        """Precompute static arrays and schedule masks for high-speed evaluation."""
        self.upper_bounds = np.array([a.max_units for a in self.actions], dtype=np.int32)
        self.costs = np.array([a.cost_lakh for a in self.actions], dtype=np.float64)
        self.hours = np.array([a.faculty_hours for a in self.actions], dtype=np.float64)
        self.gains = np.array([a.score_gain for a in self.actions], dtype=np.float64)
        self.durations = np.array([a.duration_months for a in self.actions], dtype=np.float64)
        self.schedulable_mask = np.array([a.is_schedulable for a in self.actions], dtype=bool)

        # Monthly effort rate per unit for each action: h_a / d_a
        self.monthly_rate = self.hours / np.maximum(self.durations, 1.0)

        # Schedule incidence matrix: shape (planning_horizon, num_actions)
        # Entry (m-1, a) is monthly_rate[a] if S_a < m <= F_a, else 0.0
        T = self.planning_horizon
        self.schedule_matrix = np.zeros((T, self.num_actions), dtype=np.float64)
        for a in self.actions:
            s_a = a.earliest_start
            f_a = a.earliest_finish
            rate = a.faculty_hours / max(a.duration_months, 1)
            for m in range(1, T + 1):
                if s_a < m <= f_a:
                    self.schedule_matrix[m - 1, a.index] = rate

        # Criteria incidence mapping
        self.criterion_action_mask = np.zeros((self.num_criteria, self.num_actions), dtype=np.float64)
        for c in self.criteria:
            c_idx = c.id - 1
            for a_idx in c.action_indices:
                self.criterion_action_mask[c_idx, a_idx] = self.gains[a_idx]

        # Prerequisite pairs: list of (action_index, prereq_index, u_a)
        self.prereq_pairs: List[Tuple[int, int, int]] = []
        for a in self.actions:
            for p_idx in a.prereq_indices:
                self.prereq_pairs.append((a.index, p_idx, a.max_units))

    def evaluate(self, x: np.ndarray) -> EvaluationResult:
        """
        Evaluate a candidate solution x.

        Args:
            x: 1D integer numpy array of shape (num_actions,) where x[a] in {0..u_a}.

        Returns:
            EvaluationResult dataclass with objectives, criteria scores, monthly hours, and violations.
        """
        x = np.asarray(x, dtype=np.int32)

        # 1. Criteria scores: Score_c(x) = min(M_c, s0_c + sum_{a in c} g_a * x_a)
        added_gains = self.criterion_action_mask @ x
        scores = np.minimum(self.M, self.s0 + added_gains)
        ratios = scores / self.M

        # 2. Objectives
        f1 = float(np.sum(scores))
        f2 = float(np.dot(self.costs, x))
        f3 = float(np.dot(self.hours, x))
        f4 = float(np.min(ratios))

        objectives = np.array([f1, f2, f3, f4], dtype=np.float64)
        objectives_min = np.array([-f1, f2, f3, -f4], dtype=np.float64)

        # 3. Monthly faculty hours: H_m = schedule_matrix @ x
        monthly_hours = self.schedule_matrix @ x

        # 4. Check constraints
        # (1) Budget: f2 <= B
        budget_viol = max(0.0, f2 - self.budget)

        # (2) Monthly faculty capacity: H_m <= K_m
        monthly_viols = np.maximum(0.0, monthly_hours - self.monthly_capacity)
        monthly_viol_total = float(np.sum(monthly_viols))

        # (3) Deadline: x_a = 0 if F_a > T
        unschedulable_units = x[~self.schedulable_mask]
        deadline_viol = float(np.sum(unschedulable_units))

        # (4) Prerequisites: x_a <= u_a * x_b
        prereq_viol = 0.0
        for a_idx, b_idx, u_a in self.prereq_pairs:
            diff = x[a_idx] - u_a * x[b_idx]
            if diff > 0:
                prereq_viol += float(diff)

        # (5) Minimum criterion thresholds: Score_c >= L_c
        crit_viols = np.maximum(0.0, self.L - scores)
        crit_viol_total = float(np.sum(crit_viols))

        # (6) Accreditation target: f1 >= R
        target_viol = max(0.0, self.min_accreditation_score - f1)

        violations = {
            "budget": budget_viol,
            "monthly_hours": monthly_viol_total,
            "deadline": deadline_viol,
            "prerequisites": prereq_viol,
            "criteria_min": crit_viol_total,
            "target_score": target_viol,
        }

        total_violation = sum(violations.values())
        is_feasible = (total_violation <= 1e-7)

        return EvaluationResult(
            x=x.copy(),
            objectives=objectives,
            objectives_min=objectives_min,
            criteria_scores=scores,
            criteria_ratios=ratios,
            monthly_hours=monthly_hours,
            violations=violations,
            total_violation=total_violation,
            is_feasible=is_feasible,
        )

    def evaluate_batch(self, X: np.ndarray) -> List[EvaluationResult]:
        """Evaluate a 2D array of solutions of shape (pop_size, num_actions)."""
        return [self.evaluate(sol) for sol in X]

    def random_solution(self, rng: np.random.Generator, repair: bool = True) -> np.ndarray:
        """
        Generate a random solution within bounds, optionally repaired for feasibility.
        Uses seeded numpy Generator.
        """
        x = np.zeros(self.num_actions, dtype=np.int32)
        # Randomly choose units between 0 and u_a
        for a in self.actions:
            if a.is_schedulable:
                x[a.index] = rng.integers(0, a.max_units + 1)
        if repair:
            x = self.repair(x, rng=rng)
        return x

    def repair(self, x: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Domain-aware repair operator.
        Enforces:
          1. Deadlines (x_a = 0 if F_a > T).
          2. Prerequisite compliance (x_a <= u_a * x_b).
          3. Budget constraint (f2 <= B) by pruning least cost-effective units.
          4. Monthly faculty hour capacity (H_m <= K_m).
          5. Minimum criterion threshold satisfaction (Score_c >= L_c).
          6. Target accreditation threshold (f1 >= R) if feasible within remaining budget.
        """
        sol = np.array(x, dtype=np.int32)
        sol = np.clip(sol, 0, self.upper_bounds)

        # Step 1: Zero out unschedulable actions
        sol[~self.schedulable_mask] = 0

        # Step 2: Ensure evidence packages are active if any action in criterion is chosen
        for c in self.criteria:
            crit_actions = [self.actions[idx] for idx in c.action_indices if self.actions[idx].is_schedulable]
            evidence_actions = [a for a in crit_actions if a.is_evidence]
            non_evidence = [a for a in crit_actions if not a.is_evidence]

            # If any non-evidence action has x > 0, evidence package must be at least 1
            if any(sol[a.index] > 0 for a in non_evidence):
                for ea in evidence_actions:
                    if sol[ea.index] == 0:
                        sol[ea.index] = 1

        # Step 3: Top-down prerequisite enforcement
        for a in self.actions:
            if sol[a.index] > 0 and a.prereq_indices:
                for p_idx in a.prereq_indices:
                    if sol[p_idx] == 0:
                        sol[a.index] = 0
                        break

        # Step 4: Budget & monthly hours reduction if exceeded
        # Prune units with lowest marginal efficiency (gain / cost)
        while True:
            cost = float(np.dot(self.costs, sol))
            monthly_hours = self.schedule_matrix @ sol
            budget_exceeded = cost > self.budget
            hours_exceeded = np.any(monthly_hours > self.monthly_capacity)

            if not budget_exceeded and not hours_exceeded:
                break

            # Find active non-evidence actions or evidence actions without dependent children
            candidates: List[int] = []
            for a in self.actions:
                if sol[a.index] > 0:
                    # Check if any other action with sol > 0 depends on a
                    is_depended_upon = False
                    for other in self.actions:
                        if sol[other.index] > 0 and a.index in other.prereq_indices:
                            is_depended_upon = True
                            break
                    if not is_depended_upon:
                        candidates.append(a.index)

            if not candidates:
                # Fallback: force reduction on any positive action
                candidates = [a.index for a in self.actions if sol[a.index] > 0]
                if not candidates:
                    break

            # Choose candidate with smallest score_gain / (cost_lakh + 0.01)
            effs = [self.gains[idx] / (self.costs[idx] + 0.05) for idx in candidates]
            worst_candidate = candidates[int(np.argmin(effs))]
            sol[worst_candidate] -= 1

            # Cascading prune of dependent actions if parent became 0
            if sol[worst_candidate] == 0:
                for a in self.actions:
                    if worst_candidate in a.prereq_indices and sol[a.index] > 0:
                        sol[a.index] = 0

        # Step 5: Boost solution towards min criteria L_c and target score R if budget and capacity allow
        # Greedily consider actions that improve criteria furthest below threshold or with highest gain/cost
        for _ in range(50):
            scores = np.minimum(self.M, self.s0 + self.criterion_action_mask @ sol)
            f1 = float(np.sum(scores))
            cost = float(np.dot(self.costs, sol))
            monthly = self.schedule_matrix @ sol

            under_l = np.where(scores < self.L)[0]
            if len(under_l) == 0 and f1 >= self.min_accreditation_score:
                break  # Target criteria met!

            # Look for viable candidate action to increment
            best_action_idx = -1
            best_merit = -1.0

            for a in self.actions:
                if not a.is_schedulable or sol[a.index] >= a.max_units:
                    continue

                # Check if prerequisites are satisfied
                prereqs_ok = all(sol[p] >= 1 for p in a.prereq_indices)
                if not prereqs_ok:
                    continue

                # Check budget and monthly capacity
                if cost + a.cost_lakh > self.budget:
                    continue
                a_monthly_rate = a.faculty_hours / max(a.duration_months, 1)
                new_monthly = monthly.copy()
                for m in range(1, self.planning_horizon + 1):
                    if a.earliest_start < m <= a.earliest_finish:
                        new_monthly[m - 1] += a_monthly_rate
                if np.any(new_monthly > self.monthly_capacity):
                    continue

                # Calculate marginal score improvement
                c_idx = a.crit_id - 1
                curr_c_score = scores[c_idx]
                potential_c_score = min(self.M[c_idx], curr_c_score + a.score_gain)
                marginal_gain = potential_c_score - curr_c_score

                if marginal_gain <= 0.0:
                    continue

                # Priority multiplier if criterion is below minimum threshold L_c
                priority = 3.0 if curr_c_score < self.L[c_idx] else 1.0
                merit = (marginal_gain * priority) / (a.cost_lakh + 0.1)

                if merit > best_merit:
                    best_merit = merit
                    best_action_idx = a.index

            if best_action_idx >= 0:
                sol[best_action_idx] += 1
            else:
                break

        return sol
