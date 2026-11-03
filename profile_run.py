"""Profile one reduced-budget medium-instance NSGA-II run."""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from src.algorithms.nsga2 import NSGA2Optimizer
from src.problem import NBAProblem


ROOT = Path(__file__).resolve().parent


def main() -> None:
    problem = NBAProblem(ROOT / "experiments" / "config.yaml")
    profiler = cProfile.Profile()
    profiler.enable()
    NSGA2Optimizer(problem, pop_size=40, max_generations=10, seed=0).run()
    profiler.disable()

    print("TOP 15 BY CUMULATIVE TIME")
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(15)
    print("TOP 15 BY OWN TIME")
    pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
