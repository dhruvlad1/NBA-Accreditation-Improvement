"""
Exact MILP Baseline using HiGHS solver via scipy.optimize.milp.
Formulates single-objective and epsilon-constraint multi-objective models
for exact benchmark comparisons.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import LinearConstraint, milp

from src.algorithms.nsga2 import OptimizationResult
from src.multiobjective.archive import ParetoArchive
from src.problem import EvaluationResult, NBAProblem


class HiGHSMILPSolver:
    """
    MILP Solver using SciPy's HiGHS wrapper for the NBA Accreditation Problem.
    """

    def __init__(self, problem: NBAProblem) -> None:
        self.problem = problem
        self.N = problem.num_actions
        self.C = problem.num_criteria
        # Variable layout: [x_0..x_{N-1}, y_0..y_{C-1}, z]
        # x: actions (integer, 0..u_a)
        # y: criteria scores (continuous, L_c..M_c)
        # z: min_c (y_c / M_c) (continuous, 0..1)
        self.total_vars = self.N + self.C + 1
        self.x_idx = slice(0, self.N)
        self.y_idx = slice(self.N, self.N + self.C)
        self.z_idx = self.N + self.C

        self._build_static_bounds_and_constraints()

    def _build_static_bounds_and_constraints(self) -> None:
        """Construct static bounds and constraints."""
        # Variable bounds
        lb = np.zeros(self.total_vars, dtype=np.float64)
        ub = np.zeros(self.total_vars, dtype=np.float64)

        # x bounds
        for a in self.problem.actions:
            lb[a.index] = 0.0
            ub[a.index] = float(a.max_units) if a.is_schedulable else 0.0

        # y bounds (L_c .. M_c)
        for c_idx in range(self.C):
            lb[self.N + c_idx] = float(self.problem.L[c_idx])
            ub[self.N + c_idx] = float(self.problem.M[c_idx])

        # z bounds (0 .. 1)
        lb[self.z_idx] = 0.0
        ub[self.z_idx] = 1.0

        self.lb = lb
        self.ub = ub

        # Integrality: 1 for integers (x_a), 0 for continuous (y_c, z)
        self.integrality = np.zeros(self.total_vars, dtype=int)
        self.integrality[: self.N] = 1

    def _assemble_constraints(
        self,
        max_budget: Optional[float] = None,
        max_total_hours: Optional[float] = None,
        min_total_score: Optional[float] = None,
        min_worst_ratio: Optional[float] = None,
    ) -> LinearConstraint:
        """Assemble all linear constraints into a SciPy LinearConstraint."""
        rows: List[np.ndarray] = []
        lhs: List[float] = []
        rhs: List[float] = []

        # 1. Score definition: y_c <= s0_c + sum_{a in c} g_a * x_a
        # => y_c - sum_{a in c} g_a * x_a <= s0_c
        for c in self.problem.criteria:
            c_idx = c.id - 1
            row = np.zeros(self.total_vars, dtype=np.float64)
            row[self.N + c_idx] = 1.0
            for a_idx in c.action_indices:
                row[a_idx] = -float(self.problem.gains[a_idx])
            rows.append(row)
            lhs.append(-np.inf)
            rhs.append(float(self.problem.s0[c_idx]))

        # 2. Worst-off ratio definition: z <= y_c / M_c
        # => z - (1 / M_c) * y_c <= 0
        for c_idx in range(self.C):
            row = np.zeros(self.total_vars, dtype=np.float64)
            row[self.z_idx] = 1.0
            row[self.N + c_idx] = -1.0 / float(self.problem.M[c_idx])
            rows.append(row)
            lhs.append(-np.inf)
            rhs.append(0.0)

        # 3. Budget: sum_a q_a * x_a <= B
        row = np.zeros(self.total_vars, dtype=np.float64)
        row[: self.N] = self.problem.costs
        b_val = max_budget if max_budget is not None else self.problem.budget
        rows.append(row)
        lhs.append(0.0)
        rhs.append(float(b_val))

        # 4. Monthly faculty hours: schedule_matrix @ x <= K_m
        for m in range(self.problem.planning_horizon):
            row = np.zeros(self.total_vars, dtype=np.float64)
            row[: self.N] = self.problem.schedule_matrix[m]
            rows.append(row)
            lhs.append(0.0)
            rhs.append(float(self.problem.monthly_capacity))

        # 5. Prerequisites: x_a - u_a * x_b <= 0
        for a_idx, b_idx, u_a in self.problem.prereq_pairs:
            row = np.zeros(self.total_vars, dtype=np.float64)
            row[a_idx] = 1.0
            row[b_idx] = -float(u_a)
            rows.append(row)
            lhs.append(-np.inf)
            rhs.append(0.0)

        # 6. Total accreditation score: sum_c y_c >= R
        row = np.zeros(self.total_vars, dtype=np.float64)
        row[self.N : self.N + self.C] = 1.0
        r_val = min_total_score if min_total_score is not None else self.problem.min_accreditation_score
        rows.append(row)
        lhs.append(float(r_val))
        rhs.append(np.inf)

        # Optional total hours bound
        if max_total_hours is not None:
            row = np.zeros(self.total_vars, dtype=np.float64)
            row[: self.N] = self.problem.hours
            rows.append(row)
            lhs.append(0.0)
            rhs.append(float(max_total_hours))

        # Optional worst ratio bound
        if min_worst_ratio is not None:
            row = np.zeros(self.total_vars, dtype=np.float64)
            row[self.z_idx] = 1.0
            rows.append(row)
            lhs.append(float(min_worst_ratio))
            rhs.append(1.0)

        A = np.array(rows, dtype=np.float64)
        return LinearConstraint(A, lhs, rhs)

    def solve_single_objective(
        self,
        objective_name: str = "max_f1",
        max_budget: Optional[float] = None,
        max_total_hours: Optional[float] = None,
        min_total_score: Optional[float] = None,
    ) -> Optional[EvaluationResult]:
        """
        Solve single-objective optimization using HiGHS.

        Args:
            objective_name: "max_f1", "min_f2", "min_f3", or "max_f4".
            max_budget: Optional budget override.
            max_total_hours: Optional faculty hours limit override.
            min_total_score: Optional score threshold override.

        Returns:
            EvaluationResult if optimal/feasible solution found, else None.
        """
        c = np.zeros(self.total_vars, dtype=np.float64)

        if objective_name == "max_f1":
            # Maximize sum_c y_c => Minimize -sum_c y_c
            c[self.N : self.N + self.C] = -1.0
        elif objective_name == "min_f2":
            # Minimize sum_a q_a * x_a
            c[: self.N] = self.problem.costs
        elif objective_name == "min_f3":
            # Minimize sum_a h_a * x_a
            c[: self.N] = self.problem.hours
        elif objective_name == "max_f4":
            # Maximize z => Minimize -z
            c[self.z_idx] = -1.0
        else:
            raise ValueError(f"Unknown objective: {objective_name}")

        constraints = self._assemble_constraints(
            max_budget=max_budget,
            max_total_hours=max_total_hours,
            min_total_score=min_total_score,
        )

        from scipy.optimize import Bounds
        bounds = Bounds(self.lb, self.ub)

        res = milp(
            c=c,
            integrality=self.integrality,
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "presolve": True}
        )

        if res.success and res.x is not None:
            x_sol = np.rint(res.x[: self.N]).astype(np.int32)
            eval_res = self.problem.evaluate(x_sol)
            return eval_res
        return None

    def solve_epsilon_constraint_front(
        self,
        n_budget_steps: int = 6,
        n_hour_steps: int = 5,
    ) -> List[EvaluationResult]:
        """
        Generate exact Pareto-optimal points using epsilon-constraint MILP.
        Sweeps budget epsilon in [min_feasible_budget, B] and hours epsilon in [min_hours, max_hours].

        Returns:
            List of non-dominated feasible EvaluationResults.
        """
        archive = ParetoArchive(max_size=150)

        # 1. Find anchor point for min cost
        min_cost_sol = self.solve_single_objective("min_f2")
        min_budget = min_cost_sol.f2_cost_lakh if min_cost_sol else 10.0
        max_budget = self.problem.budget

        budget_grid = np.linspace(min_budget, max_budget, n_budget_steps)

        # 2. Find anchor point for min hours
        min_hours_sol = self.solve_single_objective("min_f3")
        min_h = min_hours_sol.f3_faculty_hours if min_hours_sol else 400.0
        max_h = 1600.0
        hours_grid = np.linspace(min_h, max_h, n_hour_steps)

        # Add pure anchors
        for obj in ["max_f1", "min_f2", "min_f3", "max_f4"]:
            sol = self.solve_single_objective(obj)
            if sol and sol.is_feasible:
                archive.add(sol)

        # Grid sweep: maximize f1 subject to budget <= eps_b and hours <= eps_h
        for eps_b in budget_grid:
            for eps_h in hours_grid:
                sol = self.solve_single_objective(
                    "max_f1", max_budget=float(eps_b), max_total_hours=float(eps_h)
                )
                if sol and sol.is_feasible:
                    archive.add(sol)

        return archive.get_feasible_solutions()
