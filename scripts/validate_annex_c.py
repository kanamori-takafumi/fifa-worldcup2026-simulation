#!/usr/bin/env python3
"""Validate data/annex_c_third_place_assignments.csv.

Checks:
- 495 rows, options 1..495;
- one row for every 8-of-12 third-place group combination;
- each row assigns exactly those eight groups once;
- each assignment satisfies the allowed third-place groups in round32_slots.csv.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

ANNEX_C_COLUMNS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]
ANNEX_C_SLOT_TO_MATCH = {
    "1A": 79,
    "1B": 85,
    "1D": 81,
    "1E": 74,
    "1G": 82,
    "1I": 77,
    "1K": 87,
    "1L": 80,
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()

    annex = read_csv(args.data_dir / "annex_c_third_place_assignments.csv")
    r32 = read_csv(args.data_dir / "round32_slots.csv")

    allowed_by_match = {}
    for r in r32:
        if r["side2"] == "3rd":
            allowed_by_match[int(r["match_no"])] = set(r["side2_allowed_third_groups"].split("/"))

    expected_sets = {tuple(c) for c in combinations("ABCDEFGHIJKL", 8)}
    found_sets = set()
    options = set()

    for r in annex:
        option = int(r["option"])
        if option in options:
            raise ValueError(f"duplicate option {option}")
        options.add(option)

        third_groups = tuple(sorted(g for g in r["third_groups"].split("/") if g))
        values = []
        for col in ANNEX_C_COLUMNS:
            cell = r[col].strip().upper()
            if len(cell) != 2 or not cell.startswith("3") or cell[1] not in "ABCDEFGHIJKL":
                raise ValueError(f"invalid cell option={option}, column={col}: {cell!r}")
            group = cell[1]
            values.append(group)
            match_no = ANNEX_C_SLOT_TO_MATCH[col]
            if group not in allowed_by_match[match_no]:
                raise ValueError(
                    f"not allowed: option={option}, column={col}, match={match_no}, group={group}"
                )

        if tuple(sorted(values)) != third_groups:
            raise ValueError(f"third_groups mismatch at option {option}: {third_groups} vs {values}")
        if len(set(values)) != 8:
            raise ValueError(f"repeated third-place group at option {option}: {values}")
        found_sets.add(third_groups)

    if options != set(range(1, 496)):
        raise ValueError(f"options must be exactly 1..495; missing={sorted(set(range(1, 496)) - options)[:10]}")
    if found_sets != expected_sets:
        raise ValueError(
            f"third-place combinations incomplete: missing={len(expected_sets - found_sets)}, "
            f"extra={len(found_sets - expected_sets)}"
        )

    print("Annex C validation OK: 495 options, 495 unique 8-of-12 combinations, all slot constraints satisfied.")


if __name__ == "__main__":
    main()
