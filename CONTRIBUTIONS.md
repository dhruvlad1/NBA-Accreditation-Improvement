# Contributions

This document tracks what each contributor has built and committed to the **NBA Accreditation Improvement Planning** project, derived directly from the commit history of the project.

---

## Contributors at a Glance

| GitHub Username | Commits | Primary Role |
| :--- | :---: | :--- |
| [@Tirth-1612](https://github.com/Tirth-1612) | 8 | Core system architect — problem formulation, algorithms, benchmarking |
| [@RishitModi](https://github.com/RishitModi) | 7 | Data engineering, model hardening, validation & testing |
| [@kavish310107](https://github.com/kavish310107) | 3 | Streamlit frontend, bug visualizations & Decision Support System | project documentation |

---

## [@Tirth-1612](https://github.com/Tirth-1612)

> **Role:** Core System Architect & Algorithm Developer  
> **Total Commits:** 8 (2026-10-23)

### Contributions

#### 1. Project Scaffolding & Configuration
> Commit [`fdc8676`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/fdc8676aea5579d4b31f9793afe3ad66cfc81d56) — *chore: add .gitignore, requirements and pytest configuration*

- Bootstrapped the repository with `.gitignore`, `requirements.txt`, and `pytest.ini`
- Defined the full Python dependency stack: `numpy`, `scipy`, `pandas`, `matplotlib`, `pymoo`, `pyyaml`, `pytest`
- Set up the `pytest` test infrastructure with `testpaths = tests`

---

#### 2. Dataset, Generator & Experiment Configuration
> Commit [`fe11021`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/fe11021192b656610544b80e99dde2d9b809e0d0) — *feat: add synthetic NBA actions dataset, generator script and experiment config*

- Authored `data/generate_dataset.py` — a deterministic seeded script that programmatically generates all NBA SAR-compliant action and profile data
- Created `data/synthetic_nba_actions.csv` — initial synthetic action table
- Created `experiments/config.yaml` — master configuration file defining planning parameters (B=30 Lakh, T=12 months, K_m=300 hrs/mo, R=750 marks), data paths and all algorithm hyperparameters

---

#### 3. Problem Formulation, Metrics & Multi-Objective Utilities
> Commit [`41bd28a`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/41bd28afa69b2d23eb3e83db5e9b02f907d98e83) — *feat: add NBA problem formulation, metrics and multi-objective utilities (dominance, crowding, archive, weights)*

- Designed and implemented `src/problem.py` — the formal `NBAProblem` domain engine including:
  - `Action` and `Criterion` dataclasses
  - `EvaluationResult` dataclass (objectives f1–f4, constraint violations, feasibility flag)
  - Critical-path earliest-start schedule computation ($F_a = d_a + \max_{b \in \text{prereq}(a)} F_b$)
  - Vectorized evaluation over all 4 objectives and all 6 constraints
  - Precomputed incidence matrices, prerequisite pair arrays, and transitive dependency matrix
  - Domain-aware repair operator enforcing budget, monthly capacity, deadlines and prerequisite closures
- Implemented `src/metrics.py`:
  - Hypervolume (HV) via `pymoo`
  - Spacing (Schott, 1995), Spread (Δ), Inverted Generational Distance (IGD) and C-Metric from scratch
- Implemented `src/multiobjective/`:
  - `dominance.py` — Pareto dominance, Deb's constrained dominance, fast non-dominated sort
  - `crowding.py` — Crowding distance assignment
  - `archive.py` — Bounded external non-dominated Pareto archive
  - `weights.py` — Das-Dennis simplex lattice weight vector generation for MOEA/D

---

#### 4. All Optimization Algorithms (From Scratch)
> Commit [`7eba55e`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/7eba55ea3124f2da5926b9199d9e8f3fa3edd385) — *feat: implement NSGA-II, SPEA2, MOEA/D, memetic hybrid and baselines (greedy, random search, MILP)*

- **`src/algorithms/nsga2.py`** — Custom NSGA-II with fast non-dominated sorting, Deb's constrained dominance, crowding distance selection and `OptimizationResult` dataclass
- **`src/algorithms/spea2.py`** — Custom SPEA2 with strength-based raw fitness, k-NN density estimation in normalized objective space and iterative archive truncation
- **`src/algorithms/moead.py`** — Custom MOEA/D with Tchebycheff decomposition, neighbourhood-restricted reproduction and penalty constraint handling
- **`src/algorithms/operators.py`** — Integer SBX crossover, polynomial mutation and unit-flip mutation operators
- **`src/hybrid/memetic_nsga2.py`** — Hybrid Memetic NSGA-II combining NSGA-II global search with exact MILP warm-start seeding (HiGHS anchor initialization) and domain-aware local refinement (surplus trimming + bottleneck criterion boost)
- **`src/baseline/milp_solver.py`** — Exact MILP solver via `scipy.optimize.milp` (HiGHS): single-objective models and ε-constraint Pareto sweeps, with full MILP variable layout (x, y, z)
- **`src/baseline/greedy.py`** — Multi-attribute greedy efficiency baseline (gain/cost/hours sorting with prerequisite respect)
- **`src/baseline/random_search.py`** — Monte Carlo random search baseline with constraint repair

---

#### 5. Unit Tests, README & Project Report
> Commit [`530147d`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/530147d349aef19537444c0aae96ffbeab337831) — *test/docs: add unit tests, README and project report*

- Wrote `tests/test_algorithms.py` — tests for NSGA-II, SPEA2, MOEA/D, Hybrid Memetic, Greedy and MILP optimizer execution
- Wrote `tests/test_metrics.py` — validates HV, Spacing, Spread and Coverage indicators
- Wrote `tests/test_multiobjective.py` — tests for Pareto dominance, constrained dominance, non-dominated sorting, crowding distance and weight generation
- Wrote `tests/test_problem.py` — constraint and schedule tests (budget, prerequisites, deadlines, repair feasibility)
- Authored comprehensive `README.md` — mathematical formulation (LaTeX), repository layout, tech stack, algorithm descriptions, quick-start guide and figure legend
- Authored `report/report.md` — full academic project report with institutional context, formulation, algorithm architecture and benchmark methodology

---

#### 6. Visualization Module & Main Benchmark Pipeline
> Commit [`61c32ee`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/61c32eedfdc8d615938013f87fd9dfd372fcc768) — *feat: add visualization module and main benchmark pipeline*

- Implemented `src/visualization.py`:
  - `plot_pareto_2d_projections()` — 2×2 grid of 2D trade-off scatter plots (f1 vs f2, f1 vs f3, f2 vs f4, f3 vs f4)
  - `plot_pareto_3d()` — 3D Pareto surface (Cost × Hours × Marks)
  - `plot_criteria_radar()` — Polar radar chart comparing 10 criteria scores across plans
  - `plot_schedule_gantt_and_workload()` — Earliest-start Gantt chart with monthly faculty workload vs. 300 hrs capacity
  - `plot_convergence_history()` — Convergence of f1 and archive size over generations
  - `plot_algorithm_comparison_bars()` — Side-by-side bar charts for HV, Spacing, Spread and Runtime
- Implemented `main.py` — CLI benchmark runner: multi-seed loop, quality indicator computation (HV, Spacing, Spread, IGD, Coverage), CSV export for raw and summary results, and figure generation pipeline
- Created `figures/.gitkeep` and `results/.gitkeep` placeholder directories

---

#### 7. `.gitignore` Hardening
> Commits [`1668bbf`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/1668bbff030de873bad7b2c91a1dd62251a97eee), [`d0c7c42`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/d0c7c42fb5736f2533d81e0642bda28849cf74c5)

- Updated `.gitignore` to exclude Python caches, build artifacts (`__pycache__/`, `*.pyc`), generated output directories (`results/`, `figures/`) and virtual environment folders
- Removed already-tracked `__pycache__` entries from the index

---

## [@RishitModi](https://github.com/RishitModi)

> **Role:** Data Engineering, Model Hardening & Validation Testing  
> **Total Commits:** 7 (2026-10-24)

### Contributions

#### 1. Official NBA SAR Dataset & Data Integrity Tests
> Commit [`534b011`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/534b011fb214cb23045b1b18ca2dc3f7dd1245a5) — *data: replace synthetic action list with seeded criteria, profile and actions tables*

- Replaced the initial synthetic action placeholder with production-quality, NBA SAR-compliant datasets:
  - `data/actions.csv` — 45 improvement actions (E1–E10 evidence packages + A01–A35 interventions) with correct criterion assignments, cost, faculty hours, durations, prerequisites and SAR references
  - `data/nba_criteria.csv` — 10 official NBA Tier-I criterion codes and maximum marks (totalling 1,000)
  - `data/nba_subcriteria.csv` — 75 SAR sub-criteria with traceability marks per criterion
  - `data/profile.csv` — Synthetic programme baseline profile (current marks = 650, confirming 100-mark deficit to R=750)
- Authored `tests/test_data.py` — data integrity tests validating:
  - Criterion marks sum to 1,000; sub-criteria marks match criterion totals
  - Profile baseline totals to 650; all scores within valid ranges
  - Criterion 5 (Faculty) is the sole floor breach at baseline
  - All prerequisite action IDs are valid; no cycles in the dependency DAG
  - All 45 actions have valid SAR references
  - Generator reproducibility (deterministic output across runs)

---

#### 2. Model Hardening: Capped Scoring, Feasibility Checker, HiGHS Fixes & Repair Upgrades
> Commit [`9d465bd`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/9d465bd7d1ebe868598f344c9cb313dcaa5880f3) — *model: capped scoring, feasibility checker, HiGHS fixes; repair operator is work in progress (not yet score-aware)*

- Hardened `NBAProblem.evaluate()` — enforced the piecewise-linear score saturation cap $\min(M_c, s0_c + \text{gains})$ to prevent over-investment in a single criterion from inflating total scores
- Fixed `HiGHSMILPSolver` formulation — corrected linearization of criterion score constraints and worst-ratio variable definition
- Authored `tests/test_checker.py` — independent reference checker that evaluates solutions using manually written formulas (separate from `src/problem.py`) and asserts agreement
- Authored `tests/test_repair.py` — property-based tests verifying the repair operator:
  - Structural feasibility (budget, monthly hours, deadline, prerequisites) guaranteed post-repair
  - Bounds compliance ($0 \le x_a \le u_a$)
  - Idempotency (repairing an already-repaired solution is a no-op)
  - Seeded determinism (same seed → same repair output)
  - Input immutability (evaluate and repair do not mutate the original array)
  - Prerequisite closure and deadline propagation

---

#### 3. Small Validation Instances for Brute-Force Correctness Verification
> Commit [`c375a8a`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/c375a8aa6c23f46955dc19fd289ce238c849c60f) — *data: add small and small-f4 validation instances for brute-force checks*

- Created `data/actions_small.csv` — reduced 20-action instance to enable exhaustive enumeration
- Created `data/actions_small_f4.csv` — 21-action variant emphasizing the f4 (balance) objective
- Created `experiments/config_small.yaml` — configuration profile for the small instance
- Created `experiments/config_small_f4.yaml` — configuration profile for the f4 variant

---

#### 4. Brute-Force Validator (`tests/validate_small.py`)
> Commit [`60d223d`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/60d223dcbda04afaf11829941d37c43370046b76) — *test: add small instance brute force validator*

- Authored `tests/validate_small.py` — a complete independent brute-force correctness validator:
  - `Enumeration` class: chunked exhaustive enumeration of all plans (up to 3.5M for small, 7M for f4 variant) using mixed-radix decoding
  - Independent `evaluate_independent()` — manually computes all four objectives and all feasibility flags entirely outside of `src/problem.py`, serving as a ground-truth oracle
  - `pareto_front()` — independent non-dominated set extraction
  - Cross-checks `NBAProblem.evaluate()` against the oracle across 1,000 random plans + supplemental feasible plans (0 disagreements for both configs)
  - Validates HiGHS MILP `solve_single_objective()` against the brute-force optimal for: min f2, min f3, max f1 (at B=15/20/30/40), and max f4 (at B=15/20/30/40) — all PASS

---

#### 5. Documentation Update & PR Merges
> Commits [`ac3737b`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/ac3737bd9d05988b7a5ddfb92000e0d8cb4a8375), [`5a9c0e8`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/5a9c0e8435c03f84099ab85671a13e41e8595fcc), [`879ca3a`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/879ca3afbc9e2713a4cc998901109b7e1f2769d0)

- Updated `README.md` with A12 criterion note (proposal criterion assignment vs. SAR references 2.2.4/2.2.5), dataset disclaimer, and corrected layout
- Merged PR [#1](https://github.com/kavish310107/NBA-Accreditation-Improvement/pull/1) and PR [#2](https://github.com/kavish310107/NBA-Accreditation-Improvement/pull/2) from `dhruvlad1/partial-week3` into `main`

---

## [@kavish310107](https://github.com/kavish310107)

> **Role:** Streamlit Frontend Developer / Decision Support System  
> **Total Commits:** 3 

### Contributions

#### 1. Streamlit Interactive Frontend (Prototype)
> Commit [`36badde`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/36badde6de0ed1aec4b354d8d983b2352b292c1d) — *prototype frontend code*

- Created `app.py` — initial Streamlit web application providing an interactive Decision Support System (DSS) for the NBA accreditation optimization framework, featuring:
  - Executive Overview tab with KPI cards and Plotly criteria performance charts
  - Action Catalog & Schedule tab with filterable action inventory and earliest-start Gantt chart
  - Optimization Studio tab with live algorithm runner (NSGA-II, SPEA2, MOEA/D, Hybrid-Memetic, HiGHS, Greedy, Random)
  - Pareto Frontiers tab with interactive 3D scatter and 2D trade-off projections
  - Implementation Roadmap tab with radar chart, monthly workload chart, Gantt timeline and BOQ export
  - What-If Scenario Simulator tab with per-action sliders and real-time constraint evaluation
- Added `streamlit` and `plotly` to the project dependency stack

---

#### 2. Frontend Bug Fixes
> Commit [`4749c99`](https://github.com/kavish310107/NBA-Accreditation-Improvement/commit/4749c99299622a01ea947bead37942080b52e70d) — *fixed function/file loading errors in frontend*

- Fixed `KeyError: 'subcriteria_file'` crash when switching to small configuration profiles by using `dict.get()` with safe fallback paths
- Added `subcriteria_file` key to `experiments/config_small.yaml` and `experiments/config_small_f4.yaml`
- Isolated results cache per configuration profile to prevent dimension-mismatch errors when switching between 45-action and 20-action configs
- Fixed What-If widget desynchronization — added nonce-based dynamic widget keys so **✨ Auto-Repair Plan** and **🔄 Reset to Baseline** immediately update all `st.number_input` controls
- Added HiGHS + Greedy automatic fallback in Tab 4 to ensure Pareto charts are always populated even before running any optimizer
- Added prerequisite dependency inspector to Tab 2

---

#### 3. Project Contributions Documentation
> *Created CONTRIBUTIONS.md* — *docs: add contributions file with per-author commit history*

- Authored `CONTRIBUTIONS.md` — a detailed record of every contributor's work across all 17 commits, sourced directly from the repository's `git log`
- Documents contributions for [@Tirth-1612](https://github.com/Tirth-1612), [@RishitModi](https://github.com/RishitModi) and [@kavish310107](https://github.com/kavish310107) with per-commit breakdowns, files created and technical descriptions
- Includes a full chronological commit timeline table

---

## Commit Timeline

```
2026-10-28  kavish310107   —        docs: add contributions file (CONTRIBUTIONS.md)
2026-10-28  kavish310107   4749c99  fixed function/file loading errors in frontend
2026-10-28  kavish310107   36badde  prototype frontend code
2026-10-24  RishitModi     879ca3a  Merge pull request #2 from dhruvlad1/partial-week3
2026-10-24  RishitModi     60d223d  test: add small instance brute force validator
2026-10-24  RishitModi     5a9c0e8  Merge pull request #1 from dhruvlad1/partial-week3
2026-10-24  RishitModi     ac3737b  docs: update README
2026-10-24  RishitModi     c375a8a  data: add small and small-f4 validation instances for brute-force checks
2026-10-24  RishitModi     9d465bd  model: capped scoring, feasibility checker, HiGHS fixes; repair WIP
2026-10-24  RishitModi     534b011  data: replace synthetic action list with seeded criteria, profile and actions tables
2026-10-23  Tirth-1612     d0c7c42  chore: stop tracking Python cache files
2026-10-23  Tirth-1612     1668bbf  chore: update .gitignore to exclude caches, build artifacts and generated outputs
2026-10-23  Tirth-1612     61c32ee  feat: add visualization module and main benchmark pipeline
2026-10-23  Tirth-1612     530147d  test/docs: add unit tests, README and project report
2026-10-23  Tirth-1612     7eba55e  feat: implement NSGA-II, SPEA2, MOEA/D, memetic hybrid and baselines
2026-10-23  Tirth-1612     41bd28a  feat: add NBA problem formulation, metrics and multi-objective utilities
2026-10-23  Tirth-1612     fe11021  feat: add synthetic NBA actions dataset, generator script and experiment config
2026-10-23  Tirth-1612     fdc8676  chore: add .gitignore, requirements and pytest configuration
```

---

*Generated from `git log` output of [`kavish310107/NBA-Accreditation-Improvement`](https://github.com/kavish310107/NBA-Accreditation-Improvement). Last updated: 2026-10-28.*

