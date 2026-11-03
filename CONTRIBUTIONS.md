# Contributions

This project plans feasible NBA accreditation improvement actions with multi-objective optimization.
The numbers below were computed from the reachable `HEAD` history using the git commands listed at the end; line counts exclude `results/`, `figures/`, and `data/*.csv`.
Generated on 2026-11-04 from HEAD `1b376bd` (`1b376bdd54c70dd3c65838f4eccfa834845a2188`); this document's own commit is not included.

## Team at a glance

| GitHub user | Role | Non-merge commits | Merge commits | Lines added / removed* | Active dates | Main areas |
|---|---|---:|---:|---:|---|---|
| [Tirth-1612](https://github.com/dhruvlad1/NBA-Accreditation-Improvement) | Core architecture and algorithms | 8 | 0 | 4,513 / 0 | 2026-10-23 | formulation, algorithms, metrics, tests, visualization |
| [RishitModi](https://github.com/dhruvlad1/NBA-Accreditation-Improvement) | Data engineering and validation | 5 | 2 | 951 / 268 | 2026-10-24 | datasets, repair hardening, brute-force validation |
| [kavish310107](https://github.com/dhruvlad1/NBA-Accreditation-Improvement) | Frontend and decision support | 3 | 0 | 1,795 / 299 | 2026-10-28 | Streamlit, Plotly, configuration fixes |
| [dhruvlad1](https://github.com/dhruvlad1/NBA-Accreditation-Improvement) | Optimization, benchmarking, evaluation and release | 18 | 3 | 3,043 / 301 | 2026-11-02 to 2026-11-04 | benchmarks, profiling, references, statistics, documentation |

\* `Dhruv` and `Dhruv Lad` are the same person, and `kavish310107` and `Kavish Desai` are the same person. Counts are for commits reachable from `HEAD`; `git shortlog -sne --all` also sees six agent checkpoint refs outside the project history and therefore reports extra commits.

## Project phases

| Phase | Lead contributor | Dates | Key commits |
|---|---|---|---|
| Setup and algorithms | Tirth-1612 | 2026-10-23 | [`fdc8676`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/fdc8676aea5579d4b31f9793afe3ad66cfc81d56), [`7eba55e`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/7eba55ea3124f2da5926b9199d9e8f3fa3edd385) |
| Data and validation | RishitModi | 2026-10-24 | [`534b011`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/534b011fb214cb23045b1b18ca2dc3f7dd1245a5), [`60d223d`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/60d223dcbda04afaf11829941d37c43370046b76) |
| Frontend | kavish310107 / Kavish Desai | 2026-10-28 | [`36badde`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/36badde6de0ed1aec4b354d8d983b2352b292c1d), [`4749c99`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/4749c99299622a01ea947bead37942080b52e70d), [`dca3818`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/dca3818f49b3f048b880c5e890bcadbabb6d14c6) |
| Optimization and evaluation | dhruvlad1 | 2026-11-02 to 2026-11-04 | [`31392e3`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/31392e363bd066a94d75a41a2849aa8b81745350), [`5c6eeb5`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/5c6eeb56f7e6af881209a8459271e9aef1789fbd), [`44e4f00`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/44e4f000eb7eec002dc60e89c8fc5080c8b5d899) |
| Documentation and release | dhruvlad1 | 2026-11-04 | [`43da14a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/43da14a3c9aab4d7a69f9a79cb4a77cc5fe2a45d), [`407b66f`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/407b66ff3ab56eb412178fe0c45ad34df6c14926), [`1b376bd`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/1b376bdd54c70dd3c65838f4eccfa834845a2188) |

## [@Tirth-1612](https://github.com/Tirth-1612)

**Role:** Core System Architect and Algorithm Developer

### Highlights

- Bootstrapped the repository, configuration, datasets and test infrastructure.
- Designed the formal NBA problem engine, repair operator and four-objective metrics.
- Implemented NSGA-II, SPEA2, MOEA/D, the memetic hybrid and baseline solvers.
- Added the initial tests, report, benchmark pipeline and visualization module.

<details>
<summary>Detailed contributions</summary>

1. **Project Scaffolding and Configuration** — [`fdc8676`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/fdc8676aea5579d4b31f9793afe3ad66cfc81d56), *chore: add .gitignore, requirements and pytest configuration*
   - Bootstrapped the repository with `.gitignore`, `requirements.txt`, and `pytest.ini`.
   - Defined the full Python dependency stack: `numpy`, `scipy`, `pandas`, `matplotlib`, `pymoo`, `pyyaml`, `pytest`.
   - Set up the `pytest` test infrastructure with `testpaths = tests`.
2. **Dataset, Generator and Experiment Configuration** — [`fe11021`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/fe11021192b656610544b80e99dde2d9b809e0d0)
   - Authored `data/generate_dataset.py`, a deterministic seeded script that programmatically generates all NBA SAR-compliant action and profile data.
   - Created `data/synthetic_nba_actions.csv`, the initial synthetic action table.
   - Created `experiments/config.yaml`, defining planning parameters ($B = 30$ Lakh, $T = 12$ months, $K_m = 300$ hours/month, $R = 750$ marks), data paths and algorithm hyperparameters.
3. **Problem Formulation, Metrics and Multi-Objective Utilities** — [`41bd28a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/41bd28afa69b2d23eb3e83db5e02f907d98e83)
   - Designed and implemented `src/problem.py`, the formal `NBAProblem` domain engine including `Action` and `Criterion` dataclasses, `EvaluationResult` with objectives $f_1$--$f_4$, constraint violations and feasibility.
   - Implemented critical-path earliest-start schedule computation ($F_a = d_a + \max_{b \in \text{prereq}(a)} F_b$), vectorized evaluation over all objectives and constraints, incidence matrices, prerequisite pairs, a transitive dependency matrix and domain-aware repair.
   - Implemented `src/metrics.py`: hypervolume via `pymoo`, Spacing, Spread ($\Delta$), IGD and C-Metric.
   - Implemented `src/multiobjective/`: `dominance.py` for Pareto and Deb constrained dominance plus fast non-dominated sorting; `crowding.py`; `archive.py`; and `weights.py` for Das-Dennis simplex lattice vectors.
4. **All Optimization Algorithms** — [`7eba55e`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/7eba55ea3124f2da5926b9199d9e8f3fa3edd385)
   - **`src/algorithms/nsga2.py`** — custom NSGA-II with fast non-dominated sorting, Deb's constrained dominance, crowding distance selection and `OptimizationResult`.
   - **`src/algorithms/spea2.py`** — custom SPEA2 with strength-based raw fitness, k-nearest-neighbour density estimation in normalized objective space and iterative archive truncation.
   - **`src/algorithms/moead.py`** — custom MOEA/D with Tchebycheff decomposition, neighbourhood-restricted reproduction and penalty constraint handling.
   - **`src/algorithms/operators.py`** — integer SBX crossover, polynomial mutation and unit-flip mutation operators.
   - **`src/hybrid/memetic_nsga2.py`** — Hybrid Memetic NSGA-II combining global search with exact MILP warm-start seeding and domain-aware local refinement.
   - **`src/baseline/milp_solver.py`** — exact MILP solver via `scipy.optimize.milp` (HiGHS), including single-objective models and $\epsilon$-constraint Pareto sweeps with $(x,y,z)$ variables.
   - **`src/baseline/greedy.py`** — multi-attribute greedy efficiency baseline with gain, cost and hours sorting plus prerequisite respect.
   - **`src/baseline/random_search.py`** — Monte Carlo random search baseline with constraint repair.
5. **Unit Tests, README and Project Report** — [`530147d`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/530147d349aef19537444c0aae96ffbeab337831)
   - Wrote `tests/test_algorithms.py`, `tests/test_metrics.py`, `tests/test_multiobjective.py` and `tests/test_problem.py`.
   - Authored the comprehensive `README.md` and `report/report.md`.
6. **Visualization Module and Main Benchmark Pipeline** — [`61c32ee`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/61c32eedfdc8d615938013f87fd9dfd372fcc768)
   - Implemented `src/visualization.py`: Pareto projections, 3D surfaces, criterion radar, schedule Gantt/workload, convergence history and comparison bars.
   - Implemented `main.py`: multi-seed benchmark loop, quality indicators, CSV export and figure generation.
   - Created `figures/.gitkeep` and `results/.gitkeep`.
7. **`.gitignore` Hardening** — [`1668bbf`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/1668bbff030de873bad7b2c91a1dd62251a97eee), [`d0c7c42`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/d0c7c42fb5736f2533d81e0642bda28849cf74c5)
   - Updated `.gitignore` to exclude Python caches, build artifacts, generated output directories and virtual environments.
   - Removed already-tracked `__pycache__` entries from the index.

</details>

## [@RishitModi](https://github.com/RishitModi)

**Role:** Data Engineering, Model Hardening and Validation Testing

### Highlights

- Replaced the placeholder action list with seeded NBA SAR-compliant criteria, profile and action tables.
- Hardened capped scoring, feasibility checking, HiGHS formulation and repair behavior.
- Added small exhaustive-validation instances and independent correctness oracles.
- Updated documentation and merged the first two pull requests.

<details>
<summary>Detailed contributions</summary>

1. **Official NBA SAR Dataset and Data Integrity Tests** — [`534b011`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/534b011fb214cb23045b1b18ca2dc3f7dd1245a5)
   - Replaced the initial synthetic action placeholder with production-quality, NBA SAR-compliant datasets:
     - `data/actions.csv` — 45 improvement actions (E1--E10 evidence packages and A01--A35 interventions) with criterion assignments, cost, faculty hours, durations, prerequisites and SAR references.
     - `data/nba_criteria.csv` — 10 official NBA Tier-I criterion codes and maximum marks (totalling 1,000).
     - `data/nba_subcriteria.csv` — 75 SAR sub-criteria with traceability marks per criterion.
     - `data/profile.csv` — synthetic programme baseline profile (current marks = 650, confirming a 100-mark deficit to $R = 750$).
   - Authored `tests/test_data.py`, validating criterion totals, sub-criteria totals, profile totals and ranges, the Criterion 5 floor breach, valid prerequisite IDs and DAG acyclicity, valid SAR references and generator reproducibility.
2. **Model Hardening: Capped Scoring, Feasibility Checker, HiGHS Fixes and Repair Upgrades** — [`9d465bd`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/9d465bd7d1ebe868598f344c9cb313dcaa5880f3)
   - Hardened `NBAProblem.evaluate()` with piecewise-linear score saturation $\min(M_c, s0_c + \text{gains})$.
   - Fixed `HiGHSMILPSolver` linearization of criterion score constraints and worst-ratio variable definition.
   - Authored `tests/test_checker.py`, an independent checker that evaluates solutions using manually written formulas and asserts agreement.
   - Authored `tests/test_repair.py`, verifying structural feasibility, bounds $0 \le x_a \le u_a$, idempotency, seeded determinism, input immutability, prerequisite closure and deadline propagation.
3. **Small Validation Instances for Brute-Force Correctness Verification** — [`c375a8a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/c375a8aa6c23f46955dc19fd289ce238c849c60f)
   - Created `data/actions_small.csv`, a reduced 20-action instance.
   - Created `data/actions_small_f4.csv`, a 21-action variant emphasizing the $f_4$ objective.
   - Created `experiments/config_small.yaml` and `experiments/config_small_f4.yaml`.
4. **Brute-Force Validator** — [`60d223d`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/60d223dcbda04afaf11829941d37c43370046b76)
   - Authored `tests/validate_small.py`, a complete independent brute-force correctness validator.
   - Its `Enumeration` class performs chunked exhaustive enumeration of all plans (up to 3.5M for small and 7M for the f4 variant) using mixed-radix decoding.
   - `evaluate_independent()` manually computes all four objectives and feasibility flags outside `src/problem.py`.
   - `pareto_front()` extracts an independent non-dominated set.
   - Cross-checks against `NBAProblem.evaluate()` covered 1,000 random plans plus supplemental feasible plans, with 0 disagreements for both configurations.
   - Validated HiGHS `solve_single_objective()` against brute-force optima for min $f_2$, min $f_3$, max $f_1$ and max $f_4$ at $B = 15/20/30/40$; all passed.
5. **Documentation Update and PR Merges** — [`ac3737b`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/ac3737bd9d05988b7a5ddfb92000e0d8cb4a8375), [`5a9c0e8`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/5a9c0e8435c03f84099ab85671a13e41e8595fcc), [`879ca3a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/879ca3afbc9e2713a4cc998901109b7e1f2769d0)
   - Updated `README.md` with the A12 criterion note, proposal criterion assignment versus SAR references 2.2.4/2.2.5, dataset disclaimer and corrected layout.
   - Merged PR [#1](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/pull/1) and PR [#2](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/pull/2) from `dhruvlad1/partial-week3` into `main`.

</details>

## [@kavish310107](https://github.com/kavish310107)

**Role:** Streamlit Frontend Developer and Decision Support System

### Highlights

- Built the initial Streamlit decision-support application and interactive optimization views.
- Added Plotly charts, schedules, what-if controls and BOQ export.
- Fixed configuration switching, cache isolation, widget synchronization and fallback behavior.
- Created the first contribution record; the commit is [`dca3818`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/dca3818f49b3f048b880c5e890bcadbabb6d14c6), dated 2026-10-28, with message “addition of contributions.md, documenting member wise contributions”.

<details>
<summary>Detailed contributions</summary>

1. **Streamlit Interactive Frontend (Prototype)** — [`36badde`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/36badde6de0ed1aec4b354d8d983b2352b292c1d)
   - Created `app.py`, an interactive Decision Support System featuring an Executive Overview, Action Catalog and Schedule, Optimization Studio, Pareto Frontiers, Implementation Roadmap and What-If Scenario Simulator.
   - The views include KPI cards, Plotly criteria charts, filterable action inventory, earliest-start Gantt charts, live NSGA-II/SPEA2/MOEA/D/Hybrid-Memetic/HiGHS/Greedy/Random runners, 3D and 2D trade-off plots, radar and workload charts, Gantt timelines, BOQ export, per-action sliders and real-time constraint evaluation.
   - Added `streamlit` and `plotly` to the dependency stack.
2. **Frontend Bug Fixes** — [`4749c99`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/4749c99299622a01ea947bead37942080b52e70d)
   - Fixed `KeyError: 'subcriteria_file'` when switching configuration profiles by using `dict.get()` with safe fallback paths.
   - Added `subcriteria_file` to `experiments/config_small.yaml` and `experiments/config_small_f4.yaml`.
   - Isolated results caches by configuration profile to prevent dimension mismatches between 45-action and 20-action configurations.
   - Fixed What-If widget desynchronization with nonce-based dynamic widget keys so Auto-Repair Plan and Reset to Baseline immediately update all `st.number_input` controls.
   - Added HiGHS and Greedy automatic fallback in Tab 4 and a prerequisite dependency inspector in Tab 2.
3. **Project Contributions Documentation** — [`dca3818`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/dca3818f49b3f048b880c5e890bcadbabb6d14c6)
   - Authored `CONTRIBUTIONS.md` as a detailed record of contributors' work across the 17 commits then present, sourced directly from `git log`.
   - Documented Tirth-1612, RishitModi and kavish310107 with per-commit breakdowns, files created and technical descriptions, plus a chronological timeline.

</details>

## [@dhruvlad1](https://github.com/dhruvlad1)

**Role:** Optimization, benchmarking, evaluation and release

### Highlights

- Found and fixed a flawed first benchmark and restored the full evaluation budget.
- Made the core repair operator about 1.9 times faster with identical results.
- Built an exact dense reference front and ran a 30-seed statistically tested comparison.
- Added corner-targeted methods, ablation tooling and reproducible result figures.
- Wrote the final README and release documentation.

<details>
<summary>Detailed contributions</summary>

1. **Baseline benchmark** — [`31392e3`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/31392e363bd066a94d75a41a2849aa8b81745350) (2026-11-02): added `benchmark.py`, the first comparison of random, greedy, NSGA-II, SPEA2, MOEA/D, Hybrid-Memetic and MILP on small and medium instances.
2. **Diagnosis** — [`daf75ea`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/daf75ea0a66b19edd37f6fa79ed71c734a6a1031): added `diagnose.py`. Found that the first benchmark forced zero generations and only 100 evaluations, hypervolume was computed on infeasible fallback plans, and Hybrid-Memetic returned identical fronts for every seed because of its MILP warm start. A real NSGA-II run reached 97 percent feasibility by generation 40.
3. **Corrected benchmark** — [`ed49172`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/ed4917295456a640e431607dbbb7a18541092098): added `benchmark_v2.py` with the real 12,100-evaluation budget, feasible-only hypervolume, IGD and match percentage against the MILP front, and a Hybrid-Memetic variant without MILP seed.
4. **Profiling and speed-up** — [`bccde9c`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/bccde9c8684c461c3b866cd9ca1e71cda0ccb52f), [`11fb79f`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/11fb79f75765aa6ec8e4b878fc5f58f2e665a61f) and [`de21ab7`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/de21ab79426caae40c03ff01a6dddcd469f0d2b8): added `regression_check.py` and `profile_run.py`, optimized `src/problem.py`, and updated regression checks. Profiling showed `Problem.repair` used most run time with `np.isclose` as the main cost; function calls fell from 2,423,209 to 1,186,640, repair time from 1.79 s to 0.83 s, and one NSGA-II run from 2.06 s to 1.07 s, about 1.9 times faster. Bit-identical behavior was verified on 16 regression cases.
5. **Parallel benchmark** — [`79f02d8`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/79f02d896b02252e80156e16922b4281ca0dc6b7): added `--jobs`, `--methods`, `--resume` and incremental CSV saving in `benchmark_v2.py`.
6. **Dense exact reference** — [`5c6eeb5`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/5c6eeb56f7e6af881209a8459271e9aef1789fbd): added `dense_reference.py`, 180 MILP solves and a 29-plan exact front, verified because no heuristic plan dominated a reference plan; replaced the sparse 18-plan reference.
7. **Corner-targeted improvements and ablation** — [`5c999f5`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/5c999f5d5c8a7b6a4d0a4d0ae551bd61d2d9e312): added `src/hybrid/corner_nsga2.py`, `GreedyRepaired` in `src/baseline/greedy.py`, a small opt-in addition to `src/algorithms/nsga2.py`, `ablation.py` and `tests/test_corner.py`. The real code change was 85 insertions and 7 deletions; larger raw numbers came from CRLF line endings, restored in [`f4e2fbf`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/f4e2fbfb005da5ea7f8edfb38097ba218b99bf08c). Across 10 seeds, no variant significantly improved hypervolume.
8. **Final benchmark and statistics** — [`44e4f00`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/44e4f000eb7eec002dc60e89c8fc5080c8b5d899) and [`6275be0`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/6275be0dc9eecf3040253254767ecd0580c876e9): added `benchmark_final.py` and `corner_stats.py`, then added tolerance for small evaluation overshoot and the `over_budget` flag. The final comparison used 30 seeds, 8 methods and Mann-Whitney U tests with Holm-Bonferroni correction.
9. **Results data** — [`b739d27`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/b739d27cedce41f08e3c351c3fb025877c9b5c54): added `results/final_summary.csv`, `results/final_stats.csv` and `results/milp_dense_front.json`. SPEA2 was significantly better than NSGA-II on hypervolume (0.134 versus 0.117, Holm $p = 0.021$) and IGD; Hybrid-Memetic was better but not significant after correction ($p = 0.053$); CornerNSGA2 improved only the minimum-cost corner (Holm $p = 0.003$); MOEA/D found the best minimum hours; and the exact MILP front still had the highest hypervolume (0.167).
10. **Documentation and release** — [`43da14a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/43da14a3c9aab4d7a69f9a79cb4a77cc5fe2a45d) rewrote the README with GitHub-rendered formulas, an optimization timeline, results and `make_figures.py`; [`407b66f`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/407b66ff3ab56eb412178fe0c45ad34df6c14926) added result figures; [`e893cc0`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/e893cc0f5a3304df2cab7845daa46642d0ea6e5b), [`1b376bd`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/1b376bdd54c70dd3c65838f4eccfa834845a2188) and [`23edf3b`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/23edf3b96270e757c1e55a0d9d0d600e89a6a04c) fixed GitHub math rendering in `README.md` and `report/report.md`; [`f4e2fbf`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/f4e2fbfb005da5ea7f8edfb38097ba218b99bf08c) restored LF line endings in `nsga2.py` and `greedy.py` without a code change.
11. **Integration** — PR merges #4, #5 and #6 ([`409a7d5`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/409a7d5f4df97301b162a3bbed9a90e8cf8837ad), [`1164e79`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/1164e79739296daa9478ef1528c967c3eaeb005a), [`4ebc667`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/4ebc66796724fc31ae8c5502beb6a3320879d41a)) were made as “Dhruv Lad”; [`f87abd1`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/f87abd1f53d340815ff0fb04dfa8a8758cb38b2c) merged `origin/main` into the working branch.

</details>

## Pull requests and merges

| PR | Branch | Merged by | Date | Commit |
|---|---|---|---|---|
| [#1](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/pull/1) | `partial-week3` | RishitModi | 2026-10-24 | [`5a9c0e8`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/5a9c0e8435c03f84099ab85671a13e41e8595fcc) |
| [#2](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/pull/2) | `partial-week3` | RishitModi | 2026-10-24 | [`879ca3a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/879ca3afbc9e2713a4cc998901109b7e1f2769d0) |
| [#3](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/pull/3) | `kavish310107/main` | Kavish Desai | 2026-10-28 | [`8b01957`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/8b01957fef1622cd95bb2dfcde4ec4e10729a5c1) |
| #4 | `optimize-algorithms` | Dhruv Lad | 2026-11-03 | [`409a7d5`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/409a7d5f4df97301b162a3bbed9a90e8cf8837ad) |
| #5 | `optimize-algorithms` | Dhruv Lad | 2026-11-03 | [`1164e79`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/1164e79739296daa9478ef1528c967c3eaeb005a) |
| #6 | `optimize-algorithms` | Dhruv Lad | 2026-11-03 | [`4ebc667`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/4ebc66796724fc31ae8c5502beb6a3320879d41a) |

Plain merge commit: [`f87abd1`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/f87abd1f53d340815ff0fb04dfa8a8758cb38b2c), `Merge remote-tracking branch 'origin/main' into optimize-algorithms`.

## Who owns which files

| File or folder | Original author | Later contributors |
|---|---|---|
| `src/problem.py` | Tirth-1612 | RishitModi, dhruvlad1 |
| `src/algorithms/` | Tirth-1612 | dhruvlad1 |
| `src/hybrid/` | Tirth-1612 | dhruvlad1 |
| `src/baseline/` | Tirth-1612 | RishitModi, dhruvlad1 |
| `src/multiobjective/` | Tirth-1612 | — |
| `src/metrics.py` | Tirth-1612 | — |
| `src/visualization.py` | Tirth-1612 | — |
| `data/` | Tirth-1612 | RishitModi |
| `tests/` | Tirth-1612 | RishitModi, dhruvlad1 |
| `app.py` | kavish310107 | — |
| `main.py` | Tirth-1612 | — |
| `benchmark_v2.py` | dhruvlad1 | — |
| `benchmark_final.py` | dhruvlad1 | — |
| `dense_reference.py` | dhruvlad1 | — |
| `ablation.py` | dhruvlad1 | — |
| `regression_check.py` | dhruvlad1 | — |
| `profile_run.py` | dhruvlad1 | — |
| `make_figures.py` | dhruvlad1 | — |
| `README.md` | Tirth-1612 | RishitModi, kavish310107, dhruvlad1 |
| `report/report.md` | Tirth-1612 | dhruvlad1 |
| `CONTRIBUTIONS.md` | kavish310107 | dhruvlad1 |

## Full commit timeline

<details>
<summary>41 commits, newest first</summary>

| Date | Commit | Author | Type | Message |
|---|---|---|---|---|
| 2026-11-04 | [`1b376bd`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/1b376bdd54c70dd3c65838f4eccfa834845a2188) | Dhruv | fix | Fix README math: add spaces around < so GitHub does not parse it as HTML |
| 2026-11-04 | [`23edf3b`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/23edf3b96270e757c1e55a0d9d0d600e89a6a04c) | Dhruv | fix | Fix report math for GitHub: use lbrace/rbrace instead of escaped braces |
| 2026-11-04 | [`e893cc0`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/e893cc0f5a3304df2cab7845daa46642d0ea6e5b) | Dhruv | fix | Fix README math for GitHub: replace operatorname with text, use lbrace/rbrace |
| 2026-11-04 | [`f4e2fbf`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/f4e2fbfb005da5ea7f8edfb38097ba218b99bf08) | Dhruv | chore | Restore LF line endings in nsga2.py and greedy.py (no code change) |
| 2026-11-04 | [`407b66f`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/407b66ff3ab56eb412178fe0c45ad34df6c14926) | Dhruv | data | Add result figures referenced by README |
| 2026-11-04 | [`43da14a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/43da14a3c9aab4d7a69f9a79cb4a77cc5fe2a45d) | Dhruv | docs | Rewrite README with formulation, optimization timeline, results and figures |
| 2026-11-04 | [`b739d27`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/b739d27cedce41f08e3c351c3fb025877c9b5c54) | Dhruv | data | Add final benchmark summary, statistics and dense reference front |
| 2026-11-04 | [`6275be0`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/6275be0dc9eecf3040253254767ecd0580c876e9) | Dhruv | fix | Final benchmark: tolerate small evaluation overshoot, record over_budget flag |
| 2026-11-04 | [`44e4f00`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/44e4f000eb7eec002dc60e89c8fc5080c8b5d899) | Dhruv | feat | Add final benchmark against dense reference with Holm-corrected tests and corner statistics |
| 2026-11-03 | [`5c999f5`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/5c999f5d5c8a7b6a4d0a4d0ae551bd61d2d9e312) | Dhruv | feat | Add CornerNSGA2 (extreme seeding, corner local search), GreedyRepaired and ablation script |
| 2026-11-03 | [`4ebc667`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/4ebc66796724fc31ae8c5502beb6a3320879d41a) | Dhruv Lad | merge | Merge pull request #6 from dhruvlad1/optimize-algorithms |
| 2026-11-03 | [`f87abd1`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/f87abd1f53d340815ff0fb04dfa8a8758cb38b2c) | Dhruv | merge | Merge remote-tracking branch 'origin/main' into optimize-algorithms |
| 2026-11-03 | [`5c6eeb5`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/5c6eeb56f7e6af881209a8459271e9aef1789fbd) | Dhruv | feat | Add dense epsilon-constraint reference front and cross-check against heuristics |
| 2026-11-03 | [`1164e79`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/1164e79739296daa9478ef1528c967c3eaeb005a) | Dhruv Lad | merge | Merge pull request #5 from dhruvlad1/optimize-algorithms |
| 2026-11-03 | [`79f02d8`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/79f02d896b02252e80156e16922b4281ca0dc6b7) | Dhruv | feat | Benchmark: parallel runs, incremental saving, resume and method selection |
| 2026-11-03 | [`409a7d5`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/409a7d5f4df97301b162a3bbed9a90e8cf8837ad) | Dhruv Lad | merge | Merge pull request #4 from dhruvlad1/optimize-algorithms |
| 2026-11-02 | [`de21ab7`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/de21ab79426caae40c03ff01a6dddcd469f0d2b8) | Dhruv | fix | Update regression check: front-size handling, determinism mode, --only filter |
| 2026-11-02 | [`11fb79f`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/11fb79f75765aa6ec8e4b878fc5f58f2e665a61f) | Dhruv | perf | Speed up repair and evaluation hot paths with identical results |
| 2026-11-02 | [`bccde9c`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/bccde9c8684c461c3b866cd9ca1e71cda0ccb52f) | Dhruv | feat | Add regression check and profiling scripts for optimization work |
| 2026-11-02 | [`ed49172`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/ed4917295456a640e431607dbbb7a18541092098) | Dhruv | feat | Add corrected benchmark: real budget, feasible-only hypervolume, IGD vs MILP |
| 2026-11-02 | [`daf75ea`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/daf75ea0a66b19edd37f6fa79ed71c734a6a1031) | Dhruv | feat | Add diagnostic script for feasibility, budget and hypervolume checks |
| 2026-11-02 | [`31392e3`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/31392e363bd066a94d75a41a2849aa8b81745350) | Dhruv | feat | Add baseline benchmark script and results before optimization |
| 2026-10-28 | [`8b01957`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/8b01957fef1622cd95bb2dfcde4ec4e10729a5c1) | Kavish Desai | merge | Merge pull request #3 from kavish310107/main |
| 2026-10-28 | [`dca3818`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/dca3818f49b3f048b880c5e890bcadbabb6d14c6) | kavish310107 | docs | addition of contributions.md, documenting member wise contributions |
| 2026-10-28 | [`4749c99`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/4749c99299622a01ea947bead37942080b52e70d) | kavish310107 | fix | fixed function/file loading errors in frontend |
| 2026-10-28 | [`36badde`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/36badde6de0ed1aec4b354d8d983b2352b292c1d) | kavish310107 | feat | prototype frontend code |
| 2026-10-24 | [`879ca3a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/879ca3afbc9e2713a4cc998901109b7e1f2769d0) | RishitModi | merge | Merge pull request #2 from dhruvlad1/partial-week3 |
| 2026-10-24 | [`60d223d`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/60d223dcbda04afaf11829941d37c43370046b76) | RishitModi | test | add small instance brute force validator |
| 2026-10-24 | [`5a9c0e8`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/5a9c0e8435c03f84099ab85671a13e41e8595fcc) | RishitModi | merge | Merge pull request #1 from dhruvlad1/partial-week3 |
| 2026-10-24 | [`ac3737b`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/ac3737bd9d05988b7a5ddfb92000e0d8cb4a8375) | RishitModi | docs | update README |
| 2026-10-24 | [`c375a8a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/c375a8aa6c23f46955dc19fd289ce238c849c60f) | RishitModi | data | add small and small-f4 validation instances for brute-force checks |
| 2026-10-24 | [`9d465bd`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/9d465bd7d1ebe868598f344c9cb313dcaa5880f3) | RishitModi | fix | capped scoring, feasibility checker, HiGHS fixes; repair operator is work in progress (not yet score-aware) |
| 2026-10-24 | [`534b011`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/534b011fb214cb23045b1b18ca2dc3f7dd1245a5) | RishitModi | data | replace synthetic action list with seeded criteria, profile and actions tables |
| 2026-10-23 | [`d0c7c42`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/d0c7c42fb5736f2533d81e0642bda28849cf74c5) | Tirth-1612 | chore | stop tracking Python cache files |
| 2026-10-23 | [`1668bbf`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/1668bbff030de873bad7b2c91a1dd62251a97eee) | Tirth-1612 | chore | update .gitignore to exclude caches, build artifacts and generated outputs |
| 2026-10-23 | [`61c32ee`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/61c32eedfdc8d615938013f87fd9dfd372fcc768) | Tirth-1612 | feat | add visualization module and main benchmark pipeline |
| 2026-10-23 | [`530147d`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/530147d349aef19537444c0aae96ffbeab337831) | Tirth-1612 | test/docs | add unit tests, README and project report |
| 2026-10-23 | [`7eba55e`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/7eba55ea3124f2da5926b9199d9e8f3fa3edd385) | Tirth-1612 | feat | implement NSGA-II, SPEA2, MOEA/D, memetic hybrid and baselines (greedy, random search, MILP) |
| 2026-10-23 | [`41bd28a`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/41bd28afa69b2d23eb3e83db5e9b02f907d98e83) | Tirth-1612 | feat | add NBA problem formulation, metrics and multi-objective utilities (dominance, crowding, archive, weights) |
| 2026-10-23 | [`fe11021`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/fe11021192b656610544b80e99dde2d9b809e0d0) | Tirth-1612 | feat | add synthetic NBA actions dataset, generator script and experiment config |
| 2026-10-23 | [`fdc8676`](https://github.com/dhruvlad1/NBA-Accreditation-Improvement/commit/fdc8676aea5579d4b31f9793afe3ad66cfc81d56) | Tirth-1612 | chore | add .gitignore, requirements and pytest configuration |

</details>

## How these numbers were produced

```text
git log --reverse --format='%H|%h|%an|%ae|%ad|%s' --date=short
git shortlog -sne --all
git rev-list --count HEAD
git log --no-merges --author=<name> --ignore-space-at-eol --numstat --format=
```

For line totals, records for `results/`, `figures/` and `data/*.csv` were excluded. The full timeline is the 41-row `git log` history reachable from `HEAD`; the commit that adds this revised file is intentionally not included.
