"""Smoke test: the rollout schema round-trips through parquet.

Small and dependency-light (pyarrow + numpy, both core deps). Exercises the load-bearing
contract — that ``Episode.to_step_rows`` -> ``writer.write_rows`` produces a parquet file
whose schema matches ``ROLLOUT_SCHEMA`` exactly and reads back losslessly. This is the
gate that keeps ingest, the rollout writer, and the notebook reader in agreement.
"""

from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from harness.ingest.base import Episode, Step
from harness.schema import (
    ACTION_DIM,
    COLUMNS,
    ROLLOUT_SCHEMA,
    SCHEMA_VERSION,
    empty_step_row,
    validate_rows,
)
from harness.writer import assert_emits_schema, write_rows


def _toy_episode() -> Episode:
    steps = tuple(
        Step(
            timestep=t,
            state=np.zeros(8, dtype=np.float32),
            action=np.full(ACTION_DIM, 0.1 * t, dtype=np.float32),
            reward=float(t == 2),
            done=(t == 2),
            is_terminal=(t == 2),
            eef_pos=np.array([0.1, 0.2, 0.3 + 0.01 * t], dtype=np.float32),
            gripper_state=0.04 - 0.01 * t,
            object_poses={"akita_black_bowl": [0.0, 0.1, 0.2, 0, 0, 0, 1]},
        )
        for t in range(3)
    )
    return Episode(
        episode_id="libero_goal/0/7/0",
        source="libero",
        instruction="put the bowl on the plate",
        steps=steps,
        success=False,
        terminal_failure="re_grasp",
        model="openvla/openvla-7b-finetuned-libero-goal",
        policy_type="openvla",
        suite="libero_goal",
        task_id=0,
        task_name="put_the_bowl_on_the_plate",
        metadata={"control_mode": "relative"},
    )


def test_empty_row_has_all_columns():
    row = empty_step_row()
    assert set(row) == set(COLUMNS)
    assert all(v is None for v in row.values())


def test_to_step_rows_matches_schema_columns():
    rows = _toy_episode().to_step_rows(run_id="test")
    assert len(rows) == 3
    for row in rows:
        assert set(row) == set(COLUMNS)
        assert row["schema_version"] == SCHEMA_VERSION
    # validate_rows enforces the arrow types; would raise on a mismatch.
    table = validate_rows(rows)
    assert table.schema.equals(ROLLOUT_SCHEMA, check_metadata=False)


def test_roundtrip_through_parquet(tmp_path):
    rows = _toy_episode().to_step_rows(run_id="test")
    out = write_rows(rows, tmp_path / "ep.parquet")

    assert out.exists()
    assert_emits_schema(out)  # written schema == ROLLOUT_SCHEMA

    table = pq.read_table(out)
    assert table.num_rows == 3
    assert table.column("episode_id")[0].as_py() == "libero_goal/0/7/0"
    assert table.column("terminal_failure")[0].as_py() == "re_grasp"
    assert table.column("success")[0].as_py() is False
    # per-step granularity preserved: gripper_action is action[-1].
    # float32 round-trip is lossy (0.1 -> 0.10000000149...), so compare approximately.
    np.testing.assert_allclose(table.column("action")[1].as_py(), [0.1] * ACTION_DIM, rtol=1e-6)
    np.testing.assert_allclose(table.column("gripper_action")[1].as_py(), 0.1, rtol=1e-6)
    assert table.column("step_idx").to_pylist() == [0, 1, 2]


def test_failure_filter_query(tmp_path):
    # The wedge query: select failures, group by terminal_failure. Done here in arrow.
    out = write_rows(_toy_episode().to_step_rows(run_id="test"), tmp_path / "ep.parquet")
    table = pq.read_table(out)
    failures = table.filter(pc.invert(table.column("success")))
    assert failures.num_rows == 3
    assert set(failures.column("terminal_failure").to_pylist()) == {"re_grasp"}
