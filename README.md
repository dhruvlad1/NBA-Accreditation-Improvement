# NBA Accreditation Improvement Planning: Multi-Objective Optimization

A university research and engineering optimization framework for strategic **National Board of Accreditation (NBA)** improvement planning. Formulated as a constrained, discrete multi-objective combinatorial optimization problem with scheduling and prerequisite dependencies.

> **DISCLAIMER:** All datasets, institutional parameters, actions, and scores in this repository are **SYNTHETIC** and designed strictly for academic, research, and algorithmic benchmarking purposes.

---

## 1. Problem Overview & Mathematical Formulation

Higher education institutions undergoing NBA accreditation (Tier-I / Washington Accord) must strategically allocate limited financial budgets and faculty time to improve scores across **10 accreditation criteria** (totaling 1,000 marks).

### 1.1 Decision Variables
Let $x_a \in \{0, 1, \dots, u_a\}$ represent the integer units chosen for improvement action $a \in \{1, \dots, N\}$ (where $N = 45$).
Each action $a$ is characterized by:
- Criterion association: $\text{crit}(a) \in \{1, \dots, 10\}$
- Marginal score gain per unit: $g_a > 0$
- Direct financial cost: $q_a$ (in Lakh INR)
- Required faculty effort: $h_a$ (in person-hours)
- Duration: $d_a$ (in months)
- Prerequisite actions: $\text{prereq}(a)$
- Maximum allowable units: $u_a \in \mathbb{Z}^+$
- Earliest start & finish months: $S_a, F_a$

### 1.2 Criteria Score Functions
For each of the 10 criteria $c \in \{1, \dots, 10\}$:
$$\text{Score}_c(x) = \min\left(M_c, \; s0_c + \sum_{a \in \text{crit } c} g_a \cdot x_a\right)$$
where $M_c$ is the maximum possible score for criterion $c$, and $s0_c$ is the baseline score before intervention.

### 1.3 Objectives
The problem balances four conflicting objectives:
1. **Maximize Total NBA Score ($f_1$):**
   $$\max f_1(x) = \sum_{c=1}^{10} \text{Score}_c(x) \quad [\text{Marks out of 1000}]$$
2. **Minimize Total Implementation Cost ($f_2$):**
   $$\min f_2(x) = \sum_{a=1}^{N} q_a \cdot x_a \quad [\text{Lakh INR}]$$
3. **Minimize Total Faculty Effort ($f_3$):**
   $$\min f_3(x) = \sum_{a=1}^{N} h_a \cdot x_a \quad [\text{Hours}]$$
4. **Maximize Balance / Worst-Off Criterion Performance ($f_4$):**
   $$\max f_4(x) = \min_{c \in \{1,\dots,10\}} \frac{\text{Score}_c(x)}{M_c} \quad [\text{Satisfaction Ratio}]$$

### 1.4 Constraints
All parameters are loaded dynamically from configuration (`experiments/config.yaml`):
1. **Budget Limitation:**
   $$f_2(x) \le B \quad (B = 30.0 \text{ Lakh INR})$$
2. **Earliest-Start Monthly Faculty Capacity:**
   Action durations and prerequisites determine a critical path schedule:
   $$F_a = d_a + \max_{b \in \text{prereq}(a)} F_b \quad (\text{with } \max = 0 \text{ if } \text{prereq}(a) = \emptyset)$$
   $$S_a = F_a - d_a$$
   Active units spread effort evenly over months $S_a < m \le F_a$:
   $$H_m(x) = \sum_{a: S_a < m \le F_a} \frac{h_a}{d_a} x_a \le K_m \quad \forall m \in \{1, \dots, T\} \quad (K_m = 300.0 \text{ hrs/mo})$$
3. **Planning Horizon / Deadlines:**
   $$x_a = 0 \quad \text{if } F_a > T \quad (T = 12 \text{ months})$$
4. **Prerequisite Dependencies & Evidence Packages:**
   Each criterion has a foundational Evidence Package $E_1 \dots E_{10}$ required before subsequent actions:
   $$x_a \le u_a \cdot x_b \quad \forall b \in \text{prereq}(a)$$
5. **Criterion Minimum Quality Floor:**
   $$\text{Score}_c(x) \ge L_c \quad \forall c \in \{1, \dots, 10\}$$
6. **Total Accreditation Threshold:**
   $$f_1(x) \ge R \quad (R = 750.0 \text{ Marks, required for 5-6 year accreditation})$$

---

## 2. Repository Layout

```
NBA-Accreditation-Improvement/
├── README.md                           # Comprehensive documentation and guide
├── requirements.txt                    # Python dependencies
├── pytest.ini                          # Unit testing configuration
├── main.py                             # CLI runner for experiments & visualizations
├── data/
│   ├── generate_dataset.py             # Synthetic dataset generator script
│   └── synthetic_nba_actions.csv       # 45 actions with costs, hours, gains, prereqs
├── experiments/
│   └── config.yaml                     # Master configuration (B=30, T=12, K=300, R=750, criteria)
├── src/
│   ├── __init__.py
│   ├── problem.py                      # Domain formulation, schedule calculation, repair
│   ├── metrics.py                      # Pymoo HV, Spacing, Spread, IGD, Coverage
│   ├── visualization.py                # 2D/3D Pareto, Radar, Gantt, convergence plots
│   ├── multiobjective/
│   │   ├── __init__.py
│   │   ├── dominance.py                # Pareto dominance, Deb's constrained dominance, sorting
│   │   ├── crowding.py                 # Crowding distance assignment
│   │   ├── weights.py                  # Das-Dennis simplex lattice & weight vectors
│   │   └── archive.py                  # Bounded external non-dominated archive
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── operators.py                # Integer SBX crossover, polynomial & unit mutation
│   │   ├── nsga2.py                    # Custom NSGA-II written from scratch
│   │   ├── spea2.py                    # Custom SPEA2 written from scratch
│   │   └── moead.py                    # Custom MOEA/D with Tchebycheff decomposition
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── milp_solver.py              # Exact SciPy HiGHS single & epsilon-constraint MILP
│   │   ├── greedy.py                   # Multi-attribute greedy efficiency baseline
│   │   └── random_search.py            # Monte Carlo random baseline with repair
│   └── hybrid/
│       ├── __init__.py
│       └── memetic_nsga2.py            # Hybrid Memetic NSGA-II with MILP seeding & local search
├── tests/
│   ├── test_problem.py                 # Constraint and schedule tests
│   ├── test_multiobjective.py          # Dominance, sorting, and crowding tests
│   ├── test_algorithms.py              # Optimizer execution tests
│   └── test_metrics.py                 # Quality indicator validation
├── results/                            # CSV logs and summary tables
├── figures/                            # Publication-grade figures
└── report/
    └── report.md                       # Comprehensive university project report
```

---

## 3. Technology Stack

- **Python 3.11+**
- **NumPy & Pandas**: Matrix representations, vectorised constraint checks, dataframes
- **SciPy (`scipy.optimize.milp` with HiGHS)**: Exact integer linear programming solver
- **Matplotlib**: High-DPI publication plots and radar charts
- **Pymoo**: Used strictly for the Hypervolume indicator (`from pymoo.indicators.hv import HV`), as all evolutionary algorithms are written from scratch.
- **PyYAML**: Dynamic configuration loading.

---

## 4. Algorithms Implemented From Scratch

1. **NSGA-II (Deb et al., 2002):** Fast non-dominated sorting, Deb's constrained dominance, crowding distance assignment, integer SBX crossover, unit-flip mutation.
2. **SPEA2 (Zitzler et al., 2001):** Dominance strength assignment, raw fitness, $k$-nearest neighbor density estimation in normalized objective space, iterative archive truncation.
3. **MOEA/D (Zhang & Li, 2007):** Tchebycheff decomposition with penalty constraint handling, Das-Dennis 4D simplex weight lattice, neighborhood reproduction and replacement.
4. **Hybrid Memetic NSGA-II:** Combines NSGA-II global search with exact HiGHS MILP anchor initialization and domain-aware local refinement (surplus point trimming and bottleneck criterion boost).
5. **Exact HiGHS MILP:** Exact mathematical programming baseline using `scipy.optimize.milp` solving single-objective models and $\epsilon$-constraint Pareto sweeps.
6. **Greedy Heuristic Baseline:** Multi-criteria efficiency sorting considering marginal gain, cost, hours, and prerequisite dependencies.
7. **Random Search Baseline:** Monte Carlo sampling with constraint repair.

---

## 5. Quick Start & Execution

### Installation
```bash
python3 -m pip install -r requirements.txt
```

### Run Unit Tests
```bash
pytest -v
```

### Run Full Benchmark Suite
```bash
# Runs multi-seed experiments across all algorithms, computes metrics, and generates all figures
python3 main.py --seeds 42 101 2024
```

### Run a Single Algorithm
```bash
python3 main.py --algorithm Hybrid-Memetic --seeds 42
python3 main.py --algorithm NSGA-II --seeds 42
```

---

## 6. Generated Figures

- `figures/pareto_2d_projections.png`: 2x2 grid of 2D trade-off curves ($f_1$ vs $f_2$, $f_1$ vs $f_3$, $f_2$ vs $f_4$, $f_3$ vs $f_4$).
- `figures/pareto_3d_surface.png`: 3D Pareto frontier relating Cost, Hours, and Marks.
- `figures/criteria_radar_comparison.png`: Radar chart of all 10 NBA criteria comparing baseline, threshold floor, and plans.
- `figures/schedule_gantt_and_workload.png`: Earliest-start Gantt chart with monthly faculty workload vs 300 hrs capacity.
- `figures/convergence_history.png`: Convergence of total score ($f_1$) and archive size over generations.
- `figures/algorithm_comparison_bars.png`: Comparative bar charts for Hypervolume, Spacing, Spread, and Runtime.
