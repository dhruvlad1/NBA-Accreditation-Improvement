"""Generate the synthetic NBA planning data files.

The criteria and profile are based on the NBA SAR UG Engineering Tier-I
first-time accreditation format.  The programme profile and action catalogue
are synthetic and intended for this optimisation course project.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CONFIG_PATH = ROOT_DIR / "experiments" / "config.yaml"

CRITERIA = [
    (1, "C1", "Vision, mission and PEOs", 50),
    (2, "C2", "Curriculum and teaching-learning", 100),
    (3, "C3", "Course outcomes and programme outcomes", 175),
    (4, "C4", "Students' performance", 100),
    (5, "C5", "Faculty information and contributions", 200),
    (6, "C6", "Facilities and technical support", 80),
    (7, "C7", "Continuous improvement", 75),
    (8, "C8", "First-year academics", 50),
    (9, "C9", "Student support systems", 50),
    (10, "C10", "Governance, institutional support and financial resources", 120),
]

SUBCRITERIA: List[tuple[str, int, int]] = [
    ("1.1", 1, 5), ("1.2", 1, 5), ("1.3", 1, 15), ("1.4", 1, 15), ("1.5", 1, 10),
    ("2.1.1", 2, 10), ("2.1.2", 2, 5), ("2.1.3", 2, 5), ("2.1.4", 2, 10),
    ("2.2.1", 2, 15), ("2.2.2", 2, 15), ("2.2.3", 2, 20), ("2.2.4", 2, 10), ("2.2.5", 2, 10),
    ("3.1", 3, 25), ("3.2.1", 3, 10), ("3.2.2", 3, 65), ("3.3.1", 3, 10), ("3.3.2", 3, 65),
    ("4.1", 4, 20), ("4.2.1", 4, 15), ("4.2.2", 4, 5), ("4.3", 4, 10), ("4.4", 4, 30),
    ("4.5.1", 4, 5), ("4.5.2", 4, 5), ("4.5.3", 4, 10),
    ("5.1", 5, 20), ("5.2", 5, 20), ("5.3", 5, 20), ("5.4", 5, 10), ("5.5", 5, 10),
    ("5.6", 5, 10), ("5.7", 5, 15), ("5.8.1", 5, 20), ("5.8.2", 5, 20),
    ("5.8.3", 5, 15), ("5.8.4", 5, 20), ("5.9", 5, 10), ("5.10", 5, 10),
    ("6.1", 6, 40), ("6.2", 6, 10), ("6.3", 6, 10), ("6.4", 6, 20),
    ("7.1", 7, 30), ("7.2", 7, 15), ("7.3", 7, 10), ("7.4", 7, 20),
    ("8.1", 8, 5), ("8.2", 8, 5), ("8.3", 8, 10), ("8.4.1", 8, 5), ("8.4.2", 8, 5),
    ("8.5.1", 8, 10), ("8.5.2", 8, 10),
    ("9.1", 9, 5), ("9.2", 9, 10), ("9.3", 9, 5), ("9.4", 9, 5), ("9.5", 9, 10),
    ("9.6", 9, 5), ("9.7", 9, 10),
    ("10.1.1", 10, 5), ("10.1.2", 10, 25), ("10.1.3", 10, 10), ("10.1.4", 10, 5),
    ("10.1.5", 10, 5), ("10.1.6", 10, 5), ("10.2.1", 10, 5), ("10.2.2", 10, 5),
    ("10.2.3", 10, 5), ("10.3.1", 10, 10), ("10.3.2", 10, 20), ("10.4.1", 10, 10),
    ("10.4.2", 10, 10),
]

CURRENT_MARKS = [38, 70, 110, 70, 100, 58, 38, 36, 34, 96]
MIN_LEVELS = [25, 50, 87.5, 60, 120, 48, 37.5, 25, 25, 60]

ACTION_FIELDS = [
    "action_id", "crit_id", "name", "is_evidence", "cost_lakh", "faculty_hours",
    "duration_months", "prerequisites", "max_units", "marks_per_unit", "sar_ref", "origin",
]


def proposal_rows() -> List[Dict[str, object]]:
    def row(action_id: str, crit_id: int, name: str, cost: float, hours: int, duration: int,
            prereqs: str, marks: int, refs: str, max_units: int = 1) -> Dict[str, object]:
        return {
            "action_id": action_id, "crit_id": crit_id, "name": name, "is_evidence": False,
            "cost_lakh": cost, "faculty_hours": hours, "duration_months": duration,
            "prerequisites": prereqs, "max_units": max_units, "marks_per_unit": marks,
            "sar_ref": refs, "origin": "proposal",
        }

    return [
        row("A02", 1, "Run a PEO attainment review", 0.3, 60, 2, "E1", 6, "1.4"),
        row("A05", 2, "Refresh curriculum using industry feedback", 0.5, 120, 4, "A02;E2", 8, "2.1.1"),
        row("A09", 3, "Implement direct and indirect CO assessment", 0.1, 90, 2, "E3", 10, "3.2.1"),
        row("A10", 3, "Standardise CO-PO attainment analysis", 0.1, 150, 3, "A09", 12, "3.2.2"),
        row("A12", 4, "Create student performance improvement plan", 0.3, 100, 4, "E4", 9, "2.2.4;2.2.5"),
        row("A13", 4, "Strengthen placement and progression support", 1.2, 150, 4, "A12", 12, "4.4"),
        row("A14", 5, "Launch faculty mentoring and development", 0.6, 100, 3, "E5", 6, "5.7"),
        row("A15", 5, "Recruit a faculty member with a doctorate", 12.0, 200, 6, "E5", 12, "5.1;5.2;5.3", 2),
        row("A16", 5, "Establish faculty publication support", 0.8, 250, 6, "E5", 10, "5.8.1"),
        row("A18", 6, "Modernise core laboratory equipment", 8.0, 120, 5, "E6", 14, "6.1"),
        row("A19", 6, "Expand technical software support", 0.2, 160, 3, "E6", 6, "6.2"),
        row("A22", 7, "Close the loop on improvement actions", 0.1, 100, 3, "A09;E7", 8, "7.1"),
        row("A24", 9, "Improve mentoring and grievance support", 0.1, 80, 2, "E9", 6, "9.1;9.2"),
        row("A27", 10, "Strengthen budget monitoring controls", 0.2, 70, 2, "E10", 5, "10.2.2"),
    ]


SYNTHETIC_TEMPLATES = [
    ("A01", 1, "Stakeholder consultation evidence cycle", "1.1", "E1"),
    ("A03", 1, "Employer feedback and PEO dissemination", "1.3", "A02;E1"),
    ("A04", 2, "ICT-enabled teaching improvement", "2.2.1", "E2"),
    ("A06", 2, "Industry-aligned elective redesign", "2.2.2", "A05;E2"),
    ("A07", 3, "Question-paper quality audit", "3.1", "E3"),
    ("A08", 3, "Capstone PO mapping workshop", "3.3.1", "A09;E3"),
    ("A11", 4, "Placement preparation clinics", "4.1", "E4"),
    ("A17", 5, "Faculty workload and mentoring review", "5.8.2", "E5"),
    ("A20", 6, "Laboratory safety and calibration plan", "6.3", "E6"),
    ("A21", 7, "External academic peer review", "7.2", "A23;A22;E7"),
    ("A23", 7, "Continuous improvement dashboard", "7.3", "A20;E7"),
    ("A25", 8, "First-year bridge courses", "8.1", "E8"),
    ("A26", 8, "First-year laboratory orientation", "8.3", "E8"),
    ("A28", 8, "First-year mentoring induction", "8.5.1", "A25;E8"),
    ("A29", 9, "Career counselling programme", "9.3", "E9"),
    ("A30", 9, "Student wellness and inclusion support", "9.5", "A24;E9"),
    ("A31", 10, "Digital library resource expansion", "10.1.1", "E10"),
    ("A32", 10, "ERP reporting improvement", "10.3.1", "A27;E10"),
    ("A33", 10, "Green campus resource audit", "10.4.1", "E10"),
    ("A34", 5, "Research collaboration seed grants", "5.8.2", "A16;E5"),
    ("A35", 6, "Prototype and project laboratory setup", "6.4", "A18;E6"),
]


def evidence_rows() -> List[Dict[str, object]]:
    return [{
        "action_id": f"E{criterion_id}", "crit_id": criterion_id,
        "name": f"Evidence package for criterion {criterion_id}", "is_evidence": True,
        "cost_lakh": 0.1, "faculty_hours": 30 + (criterion_id % 5) * 5,
        "duration_months": 1 if criterion_id % 2 else 2, "prerequisites": "",
        "max_units": 1, "marks_per_unit": 0, "sar_ref": "", "origin": "synthetic",
    } for criterion_id, _, _, _ in CRITERIA]


def synthetic_rows(seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    ref_marks = {code: marks for code, _, marks in SUBCRITERIA}
    rows = []
    for action_id, crit_id, name, ref, prereqs in SYNTHETIC_TEMPLATES:
        capacity = ref_marks[ref]
        max_units = 2 if action_id in {"A03", "A06"} else 1
        marks = min(12, capacity // max_units)
        generated = {
            "action_id": action_id, "crit_id": crit_id, "name": f"SYNTHETIC: {name}",
            "is_evidence": False, "cost_lakh": round(rng.uniform(0.8, 3.0), 1),
            "faculty_hours": rng.randint(40, 250), "duration_months": rng.randint(1, 4),
            "prerequisites": prereqs, "max_units": max_units, "marks_per_unit": marks,
            "sar_ref": ref, "origin": "synthetic",
        }
        if action_id in {"A17", "A20", "A21", "A23"}:
            generated["cost_lakh"] = 3.0
            if action_id == "A17":
                generated["marks_per_unit"] = 3
        elif action_id == "A34":
            generated["cost_lakh"] = 3.0
            generated["marks_per_unit"] = 3
        rows.append(generated)
    return rows


def earliest_finishes(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    by_id = {str(row["action_id"]): row for row in rows}
    cache: Dict[str, int] = {}

    def finish(action_id: str) -> int:
        if action_id in cache:
            return cache[action_id]
        row = by_id[action_id]
        prereqs = [p for p in str(row["prerequisites"]).split(";") if p]
        cache[action_id] = int(row["duration_months"]) + max((finish(p) for p in prereqs), default=0)
        return cache[action_id]

    return {action_id: finish(action_id) for action_id in by_id}


def generate_files() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = int(config.get("data", {}).get("seed", 42))
    rows = evidence_rows() + proposal_rows() + synthetic_rows(seed)
    finishes = earliest_finishes(rows)
    assert len(rows) == 45
    assert max(finishes.values()) <= int(config["planning_parameters"]["planning_horizon_months"])

    with (DATA_DIR / "nba_criteria.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["crit_id", "code", "name", "max_marks"])
        writer.writerows(CRITERIA)
    with (DATA_DIR / "nba_subcriteria.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["code", "crit_id", "max_marks"])
        writer.writerows(SUBCRITERIA)
    with (DATA_DIR / "profile.csv").open("w", newline="", encoding="utf-8") as file:
        file.write("# SYNTHETIC programme profile; current marks total 650\n")
        writer = csv.writer(file)
        writer.writerow(["crit_id", "current_marks", "min_level", "is_mandatory"])
        writer.writerows((idx, CURRENT_MARKS[idx - 1], MIN_LEVELS[idx - 1], idx in {4, 5, 6})
                         for idx in range(1, 11))
    with (DATA_DIR / "actions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ACTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total_cost = sum(float(row["cost_lakh"]) * int(row["max_units"]) for row in rows)
    marks_by_criterion = {criterion_id: sum(
        float(row["marks_per_unit"]) * int(row["max_units"])
        for row in rows if int(row["crit_id"]) == criterion_id
    ) for criterion_id, _, _, _ in CRITERIA}
    print(f"Wrote {len(rows)} actions, {len(CRITERIA)} criteria, {len(SUBCRITERIA)} sub-criteria")
    print(f"Total cost if all action units are taken: {total_cost:.1f} lakh")
    print("Total marks available by criterion: " + ", ".join(
        f"C{criterion_id}={marks_by_criterion[criterion_id]:.0f}" for criterion_id, _, _, _ in CRITERIA
    ))


if __name__ == "__main__":
    generate_files()
