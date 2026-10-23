"""
Script to generate synthetic NBA accreditation criteria and improvement actions dataset.
Everything generated is explicitly marked synthetic for research and educational purposes.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml

DATA_DIR = Path("/Users/tirth_gujarati/NBA-Accreditation-Improvement/data")
EXP_DIR = Path("/Users/tirth_gujarati/NBA-Accreditation-Improvement/experiments")
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

# 10 NBA Criteria (Tier-I Engineering Undergraduate)
CRITERIA_SPECS = [
    {"id": 1, "code": "C1", "name": "Vision, Mission and PEOs", "max_score": 50.0, "baseline_score": 30.0, "min_threshold": 25.0},
    {"id": 2, "code": "C2", "name": "Program Curriculum and Teaching-Learning Processes", "max_score": 100.0, "baseline_score": 62.0, "min_threshold": 55.0},
    {"id": 3, "code": "C3", "name": "Course Outcomes and Program Outcomes (CO-PO)", "max_score": 175.0, "baseline_score": 105.0, "min_threshold": 95.0},
    {"id": 4, "code": "C4", "name": "Students' Performance", "max_score": 100.0, "baseline_score": 60.0, "min_threshold": 55.0},
    {"id": 5, "code": "C5", "name": "Faculty Information and Contributions", "max_score": 200.0, "baseline_score": 125.0, "min_threshold": 110.0},
    {"id": 6, "code": "C6", "name": "Facilities and Technical Support", "max_score": 80.0, "baseline_score": 52.0, "min_threshold": 44.0},
    {"id": 7, "code": "C7", "name": "Continuous Improvement", "max_score": 75.0, "baseline_score": 42.0, "min_threshold": 40.0},
    {"id": 8, "code": "C8", "name": "First Year Academics", "max_score": 50.0, "baseline_score": 32.0, "min_threshold": 28.0},
    {"id": 9, "code": "C9", "name": "Student Support Systems", "max_score": 50.0, "baseline_score": 31.0, "min_threshold": 28.0},
    {"id": 10, "code": "C10", "name": "Governance, Institutional Support and Financial Resources", "max_score": 120.0, "baseline_score": 76.0, "min_threshold": 65.0},
]

# Total marks: 1000, Total baseline: 515. Minimum accreditation: 750.

# 45 Actions definitions:
# Each criterion has an Evidence Package (E1..E10).
# All other actions in criterion c require Ec (and some have secondary prereqs).
# 1 or 2 actions will have F_a > 12 to demonstrate constraint (3) handling.
ACTIONS_SPECS = [
    # Criterion 1 (4 actions)
    {"action_id": "ACT_01", "crit_id": 1, "name": "SYNTHETIC E1: Stakeholder Consultation & PEO Review Dossier", "is_evidence": True, "cost": 0.4, "hours": 45, "duration": 2, "prereqs": [], "max_u": 1, "gain": 6.0},
    {"action_id": "ACT_02", "crit_id": 1, "name": "SYNTHETIC: Alumni Survey Automation & Employer Feedback Cycle", "is_evidence": False, "cost": 0.6, "hours": 30, "duration": 2, "prereqs": ["ACT_01"], "max_u": 2, "gain": 4.5},
    {"action_id": "ACT_03", "crit_id": 1, "name": "SYNTHETIC: Departmental Advisory Board (DAB) Governance Review", "is_evidence": False, "cost": 0.5, "hours": 24, "duration": 2, "prereqs": ["ACT_01"], "max_u": 1, "gain": 4.0},
    {"action_id": "ACT_04", "crit_id": 1, "name": "SYNTHETIC: PEO Dissemination & Internal Stakeholder Workshops", "is_evidence": False, "cost": 0.3, "hours": 20, "duration": 1, "prereqs": ["ACT_01"], "max_u": 2, "gain": 3.0},

    # Criterion 2 (5 actions)
    {"action_id": "ACT_05", "crit_id": 2, "name": "SYNTHETIC E2: Curriculum Benchmarking & Gap Audit Package", "is_evidence": True, "cost": 0.8, "hours": 60, "duration": 2, "prereqs": [], "max_u": 1, "gain": 8.0},
    {"action_id": "ACT_06", "crit_id": 2, "name": "SYNTHETIC: Industry-Aligned Electives Development & Labs", "is_evidence": False, "cost": 1.5, "hours": 75, "duration": 3, "prereqs": ["ACT_05"], "max_u": 2, "gain": 7.0},
    {"action_id": "ACT_07", "crit_id": 2, "name": "SYNTHETIC: ICT-Enabled Pedagogical Training for Faculty", "is_evidence": False, "cost": 0.9, "hours": 40, "duration": 2, "prereqs": ["ACT_05"], "max_u": 2, "gain": 5.0},
    {"action_id": "ACT_08", "crit_id": 2, "name": "SYNTHETIC: Internship & Field Project Evaluation Framework", "is_evidence": False, "cost": 0.7, "hours": 36, "duration": 2, "prereqs": ["ACT_05"], "max_u": 2, "gain": 4.5},
    {"action_id": "ACT_09", "crit_id": 2, "name": "SYNTHETIC: Digital Course Files & LMS Content Accreditation Audit", "is_evidence": False, "cost": 0.6, "hours": 50, "duration": 2, "prereqs": ["ACT_05"], "max_u": 1, "gain": 6.5},

    # Criterion 3 (5 actions)
    {"action_id": "ACT_10", "crit_id": 3, "name": "SYNTHETIC E3: OBE Direct/Indirect Assessment Framework & Rubrics", "is_evidence": True, "cost": 1.0, "hours": 80, "duration": 3, "prereqs": [], "max_u": 1, "gain": 14.0},
    {"action_id": "ACT_11", "crit_id": 3, "name": "SYNTHETIC: Automated CO-PO Attainment Computation Software", "is_evidence": False, "cost": 1.8, "hours": 45, "duration": 2, "prereqs": ["ACT_10"], "max_u": 1, "gain": 12.0},
    {"action_id": "ACT_12", "crit_id": 3, "name": "SYNTHETIC: Bloom's Taxonomy Exam Question Paper Auditing", "is_evidence": False, "cost": 0.5, "hours": 60, "duration": 2, "prereqs": ["ACT_10"], "max_u": 2, "gain": 9.0},
    {"action_id": "ACT_13", "crit_id": 3, "name": "SYNTHETIC: Capstone Project PO-Mapping & Rubric Standardisation", "is_evidence": False, "cost": 0.7, "hours": 40, "duration": 2, "prereqs": ["ACT_10"], "max_u": 2, "gain": 8.0},
    {"action_id": "ACT_14", "crit_id": 3, "name": "SYNTHETIC: Remedial Diagnostic Assessment for Slow Learners", "is_evidence": False, "cost": 0.6, "hours": 50, "duration": 3, "prereqs": ["ACT_10"], "max_u": 2, "gain": 7.5},

    # Criterion 4 (5 actions)
    {"action_id": "ACT_15", "crit_id": 4, "name": "SYNTHETIC E4: Student Graduation & Placement Verification Dossier", "is_evidence": True, "cost": 0.5, "hours": 40, "duration": 2, "prereqs": [], "max_u": 1, "gain": 8.0},
    {"action_id": "ACT_16", "crit_id": 4, "name": "SYNTHETIC: Campus Placement Preparation & Mock Technical Interviews", "is_evidence": False, "cost": 1.4, "hours": 70, "duration": 3, "prereqs": ["ACT_15"], "max_u": 2, "gain": 9.0},
    {"action_id": "ACT_17", "crit_id": 4, "name": "SYNTHETIC: GATE / Competitive Exam Coaching & Portal Access", "is_evidence": False, "cost": 1.0, "hours": 35, "duration": 3, "prereqs": ["ACT_15"], "max_u": 2, "gain": 6.5},
    {"action_id": "ACT_18", "crit_id": 4, "name": "SYNTHETIC: Professional Societies Student Chapter Activities", "is_evidence": False, "cost": 0.8, "hours": 30, "duration": 2, "prereqs": ["ACT_15"], "max_u": 2, "gain": 5.0},
    {"action_id": "ACT_19", "crit_id": 4, "name": "SYNTHETIC: Student Technical Symposium & Hackathon Sponsorship", "is_evidence": False, "cost": 1.2, "hours": 40, "duration": 2, "prereqs": ["ACT_15"], "max_u": 1, "gain": 6.0},

    # Criterion 5 (6 actions)
    {"action_id": "ACT_20", "crit_id": 5, "name": "SYNTHETIC E5: Faculty Cadre Ratio & Qualification Verification Dossier", "is_evidence": True, "cost": 0.6, "hours": 50, "duration": 2, "prereqs": [], "max_u": 1, "gain": 12.0},
    {"action_id": "ACT_21", "crit_id": 5, "name": "SYNTHETIC: Faculty Research Publication Incentives (SCI/Scopus)", "is_evidence": False, "cost": 2.5, "hours": 40, "duration": 4, "prereqs": ["ACT_20"], "max_u": 2, "gain": 15.0},
    {"action_id": "ACT_22", "crit_id": 5, "name": "SYNTHETIC: National/International Conference Travel Grants", "is_evidence": False, "cost": 1.5, "hours": 20, "duration": 3, "prereqs": ["ACT_20"], "max_u": 2, "gain": 9.0},
    {"action_id": "ACT_23", "crit_id": 5, "name": "SYNTHETIC: Faculty Development Programs (FDP / NPTEL Certifications)", "is_evidence": False, "cost": 1.0, "hours": 45, "duration": 2, "prereqs": ["ACT_20"], "max_u": 2, "gain": 10.0},
    {"action_id": "ACT_24", "crit_id": 5, "name": "SYNTHETIC: Sponsored Research & Industry Consultancy Drive", "is_evidence": False, "cost": 2.0, "hours": 60, "duration": 3, "prereqs": ["ACT_20"], "max_u": 1, "gain": 14.0},
    {"action_id": "ACT_25", "crit_id": 5, "name": "SYNTHETIC: Patent Filing & Intellectual Property Workshop", "is_evidence": False, "cost": 1.2, "hours": 30, "duration": 2, "prereqs": ["ACT_20"], "max_u": 2, "gain": 8.0},

    # Criterion 6 (5 actions)
    {"action_id": "ACT_26", "crit_id": 6, "name": "SYNTHETIC E6: Lab Equipment Modernisation & Safety Audit Dossier", "is_evidence": True, "cost": 0.5, "hours": 40, "duration": 2, "prereqs": [], "max_u": 1, "gain": 6.0},
    {"action_id": "ACT_27", "crit_id": 6, "name": "SYNTHETIC: Core Computing & Specialized Hardware Lab Upgrade", "is_evidence": False, "cost": 3.0, "hours": 40, "duration": 3, "prereqs": ["ACT_26"], "max_u": 1, "gain": 10.0},
    {"action_id": "ACT_28", "crit_id": 6, "name": "SYNTHETIC: Engineering Software Licenses (MATLAB/EDA/Simulators)", "is_evidence": False, "cost": 2.2, "hours": 25, "duration": 2, "prereqs": ["ACT_26"], "max_u": 1, "gain": 7.0},
    {"action_id": "ACT_29", "crit_id": 6, "name": "SYNTHETIC: Lab Technical Staff Skill Enhancement Workshop", "is_evidence": False, "cost": 0.4, "hours": 30, "duration": 2, "prereqs": ["ACT_26"], "max_u": 1, "gain": 3.5},
    {"action_id": "ACT_30", "crit_id": 6, "name": "SYNTHETIC: Project Laboratory & Prototyping Space Setup", "is_evidence": False, "cost": 1.8, "hours": 50, "duration": 3, "prereqs": ["ACT_26"], "max_u": 1, "gain": 6.5},

    # Criterion 7 (4 actions)
    {"action_id": "ACT_31", "crit_id": 7, "name": "SYNTHETIC E7: Academic Audit & Continuous Improvement Cycle Dossier", "is_evidence": True, "cost": 0.4, "hours": 35, "duration": 2, "prereqs": [], "max_u": 1, "gain": 7.0},
    {"action_id": "ACT_32", "crit_id": 7, "name": "SYNTHETIC: Closing the Loop PO/PSO Attainment Actions Plan", "is_evidence": False, "cost": 0.8, "hours": 50, "duration": 3, "prereqs": ["ACT_31"], "max_u": 2, "gain": 9.0},
    {"action_id": "ACT_33", "crit_id": 7, "name": "SYNTHETIC: External Peer Review & Accreditation Mock Inspection", "is_evidence": False, "cost": 1.2, "hours": 45, "duration": 2, "prereqs": ["ACT_31", "ACT_32"], "max_u": 1, "gain": 10.0},
    {"action_id": "ACT_34", "crit_id": 7, "name": "SYNTHETIC: Longitudinal Multi-Year Expansion & Renovation Plan", "is_evidence": False, "cost": 4.0, "hours": 70, "duration": 8, "prereqs": ["ACT_31", "ACT_33"], "max_u": 1, "gain": 8.0},  # Notice duration 8 after ACT_33: F will be 2+3+2+8 = 15 > 12! Constraint 3 test!

    # Criterion 8 (4 actions)
    {"action_id": "ACT_35", "crit_id": 8, "name": "SYNTHETIC E8: First-Year Student-Faculty Ratio & Mentoring Dossier", "is_evidence": True, "cost": 0.3, "hours": 30, "duration": 2, "prereqs": [], "max_u": 1, "gain": 4.5},
    {"action_id": "ACT_36", "crit_id": 8, "name": "SYNTHETIC: Bridge Courses in Mathematics & Programming", "is_evidence": False, "cost": 0.6, "hours": 40, "duration": 2, "prereqs": ["ACT_35"], "max_u": 2, "gain": 5.5},
    {"action_id": "ACT_37", "crit_id": 8, "name": "SYNTHETIC: Basic Sciences Lab Modernisation & Safety Protocols", "is_evidence": False, "cost": 1.0, "hours": 30, "duration": 2, "prereqs": ["ACT_35"], "max_u": 1, "gain": 5.0},
    {"action_id": "ACT_38", "crit_id": 8, "name": "SYNTHETIC: First-Year Induction Program (AICTE Guidelines)", "is_evidence": False, "cost": 0.5, "hours": 25, "duration": 1, "prereqs": ["ACT_35"], "max_u": 1, "gain": 4.0},

    # Criterion 9 (3 actions)
    {"action_id": "ACT_39", "crit_id": 9, "name": "SYNTHETIC E9: Mentoring & Grievance Redressal Mechanism Dossier", "is_evidence": True, "cost": 0.3, "hours": 35, "duration": 2, "prereqs": [], "max_u": 1, "gain": 5.0},
    {"action_id": "ACT_40", "crit_id": 9, "name": "SYNTHETIC: Professional Career Counseling & Mental Health Support", "is_evidence": False, "cost": 0.8, "hours": 30, "duration": 2, "prereqs": ["ACT_39"], "max_u": 2, "gain": 6.5},
    {"action_id": "ACT_41", "crit_id": 9, "name": "SYNTHETIC: Entrepreneurship Cell & Incubation Facility Grants", "is_evidence": False, "cost": 1.4, "hours": 35, "duration": 3, "prereqs": ["ACT_39"], "max_u": 1, "gain": 8.0},

    # Criterion 10 (4 actions)
    {"action_id": "ACT_42", "crit_id": 10, "name": "SYNTHETIC E10: Governance, Budget Allocation & Utilization Dossier", "is_evidence": True, "cost": 0.5, "hours": 40, "duration": 2, "prereqs": [], "max_u": 1, "gain": 10.0},
    {"action_id": "ACT_43", "crit_id": 10, "name": "SYNTHETIC: Library Digital Subscriptions (IEEE/Springer) & Automation", "is_evidence": False, "cost": 2.0, "hours": 20, "duration": 2, "prereqs": ["ACT_42"], "max_u": 1, "gain": 12.0},
    {"action_id": "ACT_44", "crit_id": 10, "name": "SYNTHETIC: ERP & Academic Management Information System Deployment", "is_evidence": False, "cost": 2.8, "hours": 50, "duration": 3, "prereqs": ["ACT_42"], "max_u": 1, "gain": 13.0},
    {"action_id": "ACT_45", "crit_id": 10, "name": "SYNTHETIC: Green Campus, Solar Energy & Waste Management Audit", "is_evidence": False, "cost": 1.2, "hours": 25, "duration": 2, "prereqs": ["ACT_42"], "max_u": 1, "gain": 9.0},
]


def generate_files() -> None:
    # 1. Write CSV dataset
    csv_path = DATA_DIR / "synthetic_nba_actions.csv"
    fieldnames = [
        "action_id", "crit_id", "name", "is_evidence", "cost_lakh",
        "faculty_hours", "duration_months", "prerequisites", "max_units", "score_gain"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in ACTIONS_SPECS:
            row = {
                "action_id": a["action_id"],
                "crit_id": a["crit_id"],
                "name": a["name"],
                "is_evidence": a["is_evidence"],
                "cost_lakh": a["cost"],
                "faculty_hours": a["hours"],
                "duration_months": a["duration"],
                "prerequisites": ";".join(a["prereqs"]),
                "max_units": a["max_u"],
                "score_gain": a["gain"],
            }
            writer.writerow(row)

    print(f"Wrote {len(ACTIONS_SPECS)} actions to {csv_path}")

    # 2. Write YAML config
    config_data: Dict[str, Any] = {
        "dataset_metadata": {
            "name": "Synthetic NBA Tier-I Engineering Accreditation Dataset",
            "is_synthetic": True,
            "actions_file": "data/synthetic_nba_actions.csv",
            "total_actions": len(ACTIONS_SPECS),
            "total_criteria": len(CRITERIA_SPECS),
            "description": "Synthetic multi-objective benchmark dataset for university NBA improvement planning.",
        },
        "planning_parameters": {
            "budget_lakh": 30.0,             # B = 30 Lakh INR
            "planning_horizon_months": 12,    # T = 12 months
            "monthly_faculty_capacity": 300.0,# K_m = 300 hours/month
            "min_accreditation_score": 750.0, # R = 750 marks out of 1000
        },
        "criteria": CRITERIA_SPECS,
        "algorithm_parameters": {
            "random_seed": 42,
            "population_size": 100,
            "max_generations": 120,
            "crossover_probability": 0.9,
            "mutation_probability": 0.15,
            "tournament_size": 2,
            "archive_size": 100,
            "moead_neighbors": 15,
            "spea2_archive_size": 100,
            "local_search_iterations": 20,
        },
        "experiments": {
            "seeds": [42, 101, 2024, 777, 999],
            "algorithms": ["NSGA-II", "SPEA2", "MOEA/D", "Hybrid-Memetic", "Greedy-Baseline", "Random-Baseline", "MILP-HiGHS"],
            "metrics": ["hypervolume", "spacing", "spread", "coverage", "feasibility_rate"],
        }
    }

    yaml_path = EXP_DIR / "config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote config to {yaml_path}")


if __name__ == "__main__":
    generate_files()
