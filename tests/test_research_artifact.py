"""Integrity gates for the committed LIBERO-Spatial exploratory pilot."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "libero-spatial-pilot-2026-07-02"
)


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (BUNDLE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_pilot_bundle_checksums() -> None:
    for line in (BUNDLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest()
        assert actual == expected, name


def test_pilot_episode_and_summary_invariants() -> None:
    episodes = _csv_rows("episodes.csv")
    assert len(episodes) == 200

    keys = {
        (
            row["policy_type"],
            row["suite"],
            int(row["task_id"]),
            int(row["init_state_id"]),
            int(row["seed"]),
        )
        for row in episodes
    }
    assert len(keys) == 200

    by_spec: dict[tuple[str, int, int, int], set[str]] = defaultdict(set)
    successes: Counter[str] = Counter()
    for row in episodes:
        spec = (
            row["suite"],
            int(row["task_id"]),
            int(row["init_state_id"]),
            int(row["seed"]),
        )
        by_spec[spec].add(row["policy_type"])
        successes[row["policy_type"]] += row["success"].lower() == "true"

    assert len(by_spec) == 100
    assert all(policies == {"openvla", "vla_jepa"} for policies in by_spec.values())
    assert successes == {"openvla": 84, "vla_jepa": 99}

    summary = json.loads((BUNDLE / "summary.json").read_text(encoding="utf-8"))
    for policy, count in successes.items():
        assert summary["overall"][policy]["successes"] == count
        assert summary["overall"][policy]["episodes"] == 100
    paired = summary["paired_outcomes"]["outcomes"]
    assert paired["both_success"]["count"] == 84
    assert paired["vla_jepa_only_success"]["count"] == 15
    assert paired["openvla_only_success"]["count"] == 0
    assert paired["both_failure"]["count"] == 1


def test_pilot_step_and_signature_invariants() -> None:
    table = pq.read_table(
        BUNDLE / "steps.parquet",
        columns=["policy_type", "episode_id", "step_idx"],
    )
    assert table.num_rows == 23_283
    data = table.to_pydict()
    step_keys = set(zip(data["policy_type"], data["episode_id"], data["step_idx"], strict=True))
    assert len(step_keys) == table.num_rows

    signatures = _csv_rows("failure_signatures.csv")
    assert len(signatures) == 17
    assert {row["source_terminal_failure"] for row in signatures} == {"unlabeled"}
    assert {row["heuristic_label"] for row in signatures} <= {
        "no_hold_signal_candidate",
        "repeated_close_candidate",
    }
