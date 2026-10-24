"""Integrity tests for the NBA SAR-derived CSV data and generator."""

from pathlib import Path

import pandas as pd
import pytest

from data.generate_dataset import generate_files

DATA_DIR = Path("data")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, comment="#")


def test_criteria_and_subcriteria_marks() -> None:
    criteria = read_csv("nba_criteria.csv")
    subcriteria = read_csv("nba_subcriteria.csv")
    assert criteria["max_marks"].sum() == 1000
    expected = {1: 50, 2: 100, 3: 175, 4: 100, 5: 200, 6: 80, 7: 75, 8: 50, 9: 50, 10: 120}
    assert dict(zip(criteria["crit_id"], criteria["max_marks"])) == expected
    assert subcriteria.groupby("crit_id")["max_marks"].sum().to_dict() == expected


def test_profile_rules() -> None:
    criteria = read_csv("nba_criteria.csv").set_index("crit_id")
    profile = read_csv("profile.csv").set_index("crit_id")
    assert profile["current_marks"].sum() == 650
    assert (profile["current_marks"] <= criteria["max_marks"]).all()
    assert profile.loc[5, "current_marks"] < profile.loc[5, "min_level"]
    assert all(profile.loc[c, "current_marks"] >= profile.loc[c, "min_level"] for c in profile.index if c != 5)
    assert set(profile.index[profile["is_mandatory"]]) == {4, 5, 6}
    assert 750 - profile["current_marks"].sum() == 100


def test_actions_and_prerequisites() -> None:
    actions = read_csv("actions.csv")
    assert len(actions) == 45
    assert actions["action_id"].is_unique
    evidence = actions[actions["is_evidence"]]
    assert len(evidence) == 10
    assert set(evidence["crit_id"]) == set(range(1, 11))
    action_ids = set(actions["action_id"])
    graph = {action_id: [p for p in str(prereqs).split(";") if p and p != "nan"]
             for action_id, prereqs in zip(actions["action_id"], actions["prerequisites"])}
    assert all(parent in action_ids for parents in graph.values() for parent in parents)

    visiting = set()
    visited = set()

    def visit(action_id: str) -> None:
        if action_id in visiting:
            pytest.fail("action prerequisite graph contains a cycle")
        if action_id in visited:
            return
        visiting.add(action_id)
        for parent in graph[action_id]:
            visit(parent)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in graph:
        visit(action_id)


def test_schedule_and_sar_references() -> None:
    actions = read_csv("actions.csv")
    refs = read_csv("nba_subcriteria.csv").set_index("code")["max_marks"].to_dict()
    finishes = {}
    by_id = actions.set_index("action_id")

    def finish(action_id: str) -> int:
        if action_id not in finishes:
            parents = [p for p in str(by_id.loc[action_id, "prerequisites"]).split(";") if p and p != "nan"]
            finishes[action_id] = int(by_id.loc[action_id, "duration_months"]) + max(
                (finish(parent) for parent in parents), default=0
            )
        return finishes[action_id]

    assert max(finish(action_id) for action_id in by_id.index) <= 12
    for _, action in actions.iterrows():
        codes = [code for code in str(action["sar_ref"]).split(";") if code and code != "nan"]
        assert all(code in refs for code in codes)
        available = sum(refs[code] for code in codes)
        units = float(action["marks_per_unit"]) * int(action["max_units"])
        if not (action["origin"] == "proposal" and units > available):
            assert units <= available


def test_proposal_rows_match_specification() -> None:
    actions = read_csv("actions.csv").set_index("action_id")
    expected = {
        "A02": (1, 0.3, 60, 2, "E1", 1, 6, "1.4"),
        "A05": (2, 0.5, 120, 4, "A02;E2", 1, 8, "2.1.1"),
        "A09": (3, 0.1, 90, 2, "E3", 1, 10, "3.2.1"),
        "A10": (3, 0.1, 150, 3, "A09", 1, 12, "3.2.2"),
        "A12": (4, 0.3, 100, 4, "E4", 1, 9, "2.2.4;2.2.5"),
        "A13": (4, 1.2, 150, 4, "A12", 1, 12, "4.4"),
        "A14": (5, 0.6, 100, 3, "E5", 1, 6, "5.7"),
        "A15": (5, 12.0, 200, 6, "E5", 2, 12, "5.1;5.2;5.3"),
        "A16": (5, 0.8, 250, 6, "E5", 1, 10, "5.8.1"),
        "A18": (6, 8.0, 120, 5, "E6", 1, 14, "6.1"),
        "A19": (6, 0.2, 160, 3, "E6", 1, 6, "6.2"),
        "A22": (7, 0.1, 100, 3, "A09;E7", 1, 8, "7.1"),
        "A24": (9, 0.1, 80, 2, "E9", 1, 6, "9.1;9.2"),
        "A27": (10, 0.2, 70, 2, "E10", 1, 5, "10.2.2"),
    }
    for action_id, values in expected.items():
        row = actions.loc[action_id]
        assert tuple(row[field] for field in
                     ["crit_id", "cost_lakh", "faculty_hours", "duration_months",
                      "prerequisites", "max_units", "marks_per_unit", "sar_ref"]) == values
        assert row["origin"] == "proposal"


def test_generator_is_deterministic(tmp_path: Path) -> None:
    original = {name: (DATA_DIR / name).read_bytes() for name in
                ["nba_criteria.csv", "nba_subcriteria.csv", "profile.csv", "actions.csv"]}
    generate_files()
    first = {name: (DATA_DIR / name).read_bytes() for name in original}
    generate_files()
    second = {name: (DATA_DIR / name).read_bytes() for name in original}
    assert first == second == original
