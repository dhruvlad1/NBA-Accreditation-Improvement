"""Save and verify reduced-budget golden outputs for algorithm regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np

from src.algorithms.moead import MOEADOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer
from src.algorithms.spea2 import SPEA2Optimizer
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.problem import NBAProblem


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "small": ROOT / "experiments" / "config_small.yaml",
    "medium": ROOT / "experiments" / "config.yaml",
}
ALGORITHMS = ("NSGA-II", "SPEA2", "MOEA/D", "Hybrid-Memetic")
POPULATION = 40
GENERATIONS = 10
SEEDS = (0, 1)


def run_one(instance: str, algorithm: str, seed: int) -> Dict[str, Any]:
    problem = NBAProblem(CONFIGS[instance])
    params = problem.config.get("algorithm_parameters", {})
    common = {
        "crossover_prob": params.get("crossover_probability", 0.9),
        "mutation_prob": params.get("mutation_probability", 0.15),
    }
    if algorithm == "NSGA-II":
        optimizer = NSGA2Optimizer(problem, POPULATION, GENERATIONS, seed=seed, **common)
    elif algorithm == "SPEA2":
        optimizer = SPEA2Optimizer(
            problem, POPULATION, POPULATION, GENERATIONS, seed=seed, **common
        )
    elif algorithm == "MOEA/D":
        optimizer = MOEADOptimizer(
            problem, POPULATION,
            n_neighbors=min(params.get("moead_neighbors", 15), POPULATION),
            max_generations=GENERATIONS, seed=seed, **common
        )
    else:
        optimizer = MemeticNSGA2Optimizer(
            problem, POPULATION, GENERATIONS, seed=seed, **common
        )
    result = optimizer.run()
    return {
        "objectives": [solution.objectives.tolist() for solution in result.pareto_solutions],
        "decisions": [solution.x.tolist() for solution in result.pareto_solutions],
    }


def build_golden() -> Dict[str, Any]:
    golden: Dict[str, Any] = {
        "population": POPULATION,
        "generations": GENERATIONS,
        "seeds": list(SEEDS),
        "runs": {},
    }
    for instance in CONFIGS:
        golden["runs"][instance] = {}
        for algorithm in ALGORITHMS:
            golden["runs"][instance][algorithm] = {}
            for seed in SEEDS:
                print(f"Saving {instance}/{algorithm}/seed={seed}")
                golden["runs"][instance][algorithm][str(seed)] = run_one(
                    instance, algorithm, seed
                )
    return golden


def check_golden(golden: Dict[str, Any]) -> bool:
    passed = True
    for instance in CONFIGS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                current = run_one(instance, algorithm, seed)
                expected = golden["runs"][instance][algorithm][str(seed)]
                objectives_equal = np.allclose(
                    current["objectives"], expected["objectives"],
                    rtol=0.0, atol=1e-9,
                )
                decisions_equal = current["decisions"] == expected["decisions"]
                result = objectives_equal and decisions_equal
                print(
                    f"{'PASS' if result else 'FAIL'} "
                    f"{instance}/{algorithm}/seed={seed}"
                )
                passed = passed and result
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.save == args.check:
        parser.error("choose exactly one of --save or --check")

    start = time.perf_counter()
    path = ROOT / "results" / "golden_front.json"
    if args.save:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_golden(), indent=2), encoding="utf-8")
        print(f"Saved {path}")
        passed = True
    else:
        if not path.exists():
            parser.error(f"golden file not found: {path}")
        passed = check_golden(json.loads(path.read_text(encoding="utf-8")))
    print(f"TOTAL: {'PASS' if passed else 'FAIL'}")
    print(f"Total runtime: {time.perf_counter() - start:.3f}s")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
