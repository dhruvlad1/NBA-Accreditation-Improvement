"""Independent brute-force validation for both small NBA instances."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline.milp_solver import HiGHSMILPSolver
from src.problem import NBAProblem


CHUNK_SIZE = 100_000
BASE_BUDGET = 30.0
BUDGETS = (15.0, 20.0, 30.0, 40.0)
TOLERANCES = {1: 1e-6, 3: 1e-6}


class Enumeration:
    """Chunked independent enumeration of plans satisfying constraints 2-6."""

    def __init__(self, problem: NBAProblem) -> None:
        self.problem = problem
        self.data = pd.read_csv(ROOT / problem.config["data"]["actions_file"])
        self.action_count = len(self.data)
        self.upper_bounds = self.data["max_units"].to_numpy(dtype=np.int64)
        self.radices = self.upper_bounds + 1
        self.total_plans = int(np.prod(self.radices, dtype=np.int64))
        self.action_ids = self.data["action_id"].tolist()
        self.index_by_id = {action_id: i for i, action_id in enumerate(self.action_ids)}
        self.prerequisites = [
            [self.index_by_id[parent] for parent in str(raw).split(";") if parent]
            for raw in self.data["prerequisites"].fillna("")
        ]
        self.finishes = self._compute_finishes()
        self.schedule = self._build_schedule()
        self.incidence = np.zeros((problem.num_criteria, self.action_count))
        criteria = self.data["crit_id"].to_numpy(dtype=int) - 1
        self.incidence[criteria, np.arange(self.action_count)] = (
            self.data["marks_per_unit"].to_numpy(dtype=float)
        )
        self.objectives: np.ndarray
        self.plans: np.ndarray
        self._enumerate()

    def _compute_finishes(self) -> np.ndarray:
        durations = self.data["duration_months"].to_numpy(dtype=int)
        finishes: list[int | None] = [None] * self.action_count

        def finish(index: int) -> int:
            if finishes[index] is None:
                finishes[index] = durations[index] + max(
                    (finish(parent) for parent in self.prerequisites[index]),
                    default=0,
                )
            return finishes[index]  # type: ignore[return-value]

        return np.array([finish(index) for index in range(self.action_count)], dtype=int)

    def _build_schedule(self) -> np.ndarray:
        schedule = np.zeros((self.problem.planning_horizon, self.action_count))
        hours = self.data["faculty_hours"].to_numpy(dtype=float)
        durations = self.data["duration_months"].to_numpy(dtype=float)
        starts = self.finishes - durations.astype(int)
        for action in range(self.action_count):
            for month in range(1, self.problem.planning_horizon + 1):
                if starts[action] < month <= self.finishes[action]:
                    schedule[month - 1, action] = hours[action] / durations[action]
        return schedule

    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        plans = np.empty((len(encoded), self.action_count), dtype=np.int8)
        remaining = encoded.copy()
        for index, radix in enumerate(self.radices):
            plans[:, index] = remaining % radix
            remaining //= radix
        return plans

    def evaluate_independent(self, plans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate formulas independently of src.problem (Proposal Section 6.3-6.5)."""
        scores = np.minimum(
            self.problem.M,
            self.problem.s0 + plans @ self.incidence.T,
        )
        costs = plans @ self.problem.costs
        hours = plans @ self.problem.hours
        worst_ratio = (scores / self.problem.M).min(axis=1)
        objectives = np.column_stack((scores.sum(axis=1), costs, hours, worst_ratio))

        feasible = np.all(plans @ self.schedule.T <= self.problem.monthly_capacity + 1e-9, axis=1)
        feasible &= plans[:, self.finishes > self.problem.planning_horizon].sum(axis=1) == 0
        for action, parents in enumerate(self.prerequisites):
            for parent in parents:
                feasible &= plans[:, action] <= self.upper_bounds[action] * plans[:, parent]
        feasible &= np.all(scores >= self.problem.L - 1e-9, axis=1)
        feasible &= objectives[:, 0] >= self.problem.min_accreditation_score - 1e-9
        return objectives, feasible

    def _enumerate(self) -> None:
        plans: list[np.ndarray] = []
        objectives: list[np.ndarray] = []
        for start in range(0, self.total_plans, CHUNK_SIZE):
            encoded = np.arange(
                start,
                min(start + CHUNK_SIZE, self.total_plans),
                dtype=np.int64,
            )
            chunk = self._decode(encoded)
            chunk_objectives, feasible = self.evaluate_independent(chunk)
            if feasible.any():
                plans.append(chunk[feasible])
                objectives.append(chunk_objectives[feasible])
        self.plans = np.vstack(plans) if plans else np.empty((0, self.action_count), dtype=np.int8)
        self.objectives = (
            np.vstack(objectives) if objectives else np.empty((0, 4), dtype=float)
        )


def best(objectives: np.ndarray, objective: str, budget: float | None) -> np.ndarray | None:
    selected = objectives if budget is None else objectives[objectives[:, 1] <= budget + 1e-9]
    if len(selected) == 0:
        return None
    if objective == "min_f2":
        index = np.lexsort((selected[:, 2], selected[:, 1]))[0]
    elif objective == "min_f3":
        index = np.lexsort((selected[:, 1], selected[:, 2]))[0]
    elif objective == "max_f1":
        index = int(np.argmax(selected[:, 0]))
    else:
        index = int(np.argmax(selected[:, 3]))
    return selected[index]


def pareto_front(objectives: np.ndarray, budget: float) -> np.ndarray:
    selected = objectives[objectives[:, 1] <= budget + 1e-9]
    minimization = np.column_stack(
        (-selected[:, 0], selected[:, 1], selected[:, 2], -selected[:, 3])
    )
    keep = np.ones(len(selected), dtype=bool)
    for index in range(len(selected)):
        dominates = np.all(minimization <= minimization[index] + 1e-9, axis=1)
        dominates &= np.any(minimization < minimization[index] - 1e-9, axis=1)
        keep[index] = not np.any(dominates)
    return selected[keep]


def same_objectives(actual: np.ndarray, expected: np.ndarray) -> bool:
    return (
        actual[0] == expected[0]
        and actual[2] == expected[2]
        and np.isclose(actual[1], expected[1], atol=1e-6)
        and np.isclose(actual[3], expected[3], atol=1e-6)
    )


def print_case(label: str, expected: np.ndarray | None, result, index: int, problem: NBAProblem) -> None:
    if expected is None and result is None:
        print(f"{label}: PASS (both report no feasible plan)")
        return
    if expected is None or result is None:
        print(f"{label}: FAIL (brute={expected is not None}, HiGHS={result is not None})")
        return
    evaluated = problem.evaluate(result.x)
    actual = evaluated.objectives
    tolerance = TOLERANCES.get(index, 0.0)
    passed = abs(float(actual[index]) - float(expected[index])) <= tolerance
    print(
        f"{label}: {'PASS' if passed else 'FAIL'} "
        f"(brute={expected[index]:.9f}, HiGHS={actual[index]:.9f}, "
        f"evaluate=[{actual[0]:.0f}, {actual[1]:.6f}, {actual[2]:.0f}, {actual[3]:.9f}])"
    )


def validate(config_name: str) -> None:
    problem = NBAProblem(ROOT / "experiments" / config_name)
    enumeration = Enumeration(problem)
    print(f"\n{config_name}: plans={enumeration.total_plans}, feasible_checked={len(enumeration.plans)}")

    rng = np.random.default_rng(20261024)
    disagreements = 0
    random_feasible = 0
    for _ in range(1000):
        plan = np.array(
            [rng.integers(0, int(bound) + 1) for bound in enumeration.upper_bounds],
            dtype=np.int32,
        )
        expected, flags = enumeration.evaluate_independent(plan[None, :])
        evaluated = problem.evaluate(plan)
        random_feasible += int(flags[0] and expected[0, 1] <= BASE_BUDGET + 1e-9)
        expected_feasible = bool(flags[0] and expected[0, 1] <= BASE_BUDGET + 1e-9)
        disagreements += int(
            not (
                same_objectives(evaluated.objectives, expected[0])
                and evaluated.is_feasible == expected_feasible
            )
        )
    supplemental = 0
    if random_feasible < 50:
        supplemental = min(200, len(enumeration.plans))
        for plan, expected in zip(
            enumeration.plans[:supplemental],
            enumeration.objectives[:supplemental],
        ):
            evaluated = problem.evaluate(plan.astype(np.int32))
            disagreements += int(not same_objectives(evaluated.objectives, expected))
    print(
        f"cross-check: {'PASS' if disagreements == 0 else 'FAIL'} "
        f"(random_feasible={random_feasible}, supplemental={supplemental}, "
        f"disagreements={disagreements}, feasible_plans_checked={len(enumeration.plans)})"
    )

    front = pareto_front(enumeration.objectives, BASE_BUDGET)
    print(f"front: {len(front)} points")
    solver = HiGHSMILPSolver(problem)
    print_case(
        "min f2 with f1>=R",
        best(enumeration.objectives, "min_f2", BASE_BUDGET),
        solver.solve_single_objective("min_f2"),
        1,
        problem,
    )
    print_case(
        "min f3 with f1>=R",
        best(enumeration.objectives, "min_f3", BASE_BUDGET),
        solver.solve_single_objective("min_f3"),
        2,
        problem,
    )
    for budget in (15, 20, 30, 40):
        print_case(
            f"max f1 with B={budget}",
            best(enumeration.objectives, "max_f1", budget),
            solver.solve_single_objective("max_f1", max_budget=budget),
            0,
            problem,
        )
        print_case(
            f"max f4 with B={budget}",
            best(enumeration.objectives, "max_f4", budget),
            solver.solve_single_objective("max_f4", max_budget=budget),
            3,
            problem,
        )


if __name__ == "__main__":
    validate("config_small.yaml")
    validate("config_small_f4.yaml")
