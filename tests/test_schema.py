"""Smoke test: the Daft rollout schema round-trips through Parquet."""

from __future__ import annotations

import daft
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from physical_ai_evals.core.episode import Episode, Step
from physical_ai_evals.core.schema import (
    ACTION_DIM,
    COLUMNS,
    ROLLOUT_SCHEMA,
    SCHEMA_VERSION,
    empty_step_row,
    validate_rows,
)
from physical_ai_evals.core.writer import assert_emits_schema, write_rows


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
    assert isinstance(rows[0]["action"], np.ndarray)
    frame = validate_rows(rows)
    assert frame.schema() == ROLLOUT_SCHEMA


def test_tensor_shape_is_enforced():
    rows = _toy_episode().to_step_rows(run_id="test")
    rows[0]["action"] = np.zeros(ACTION_DIM - 1, dtype=np.float32)

    with pytest.raises(Exception, match="shapes different"):
        validate_rows(rows)


def test_roundtrip_through_parquet(tmp_path):
    rows = _toy_episode().to_step_rows(run_id="test")
    out = write_rows(rows, tmp_path / "ep.parquet")

    assert out.exists()
    assert_emits_schema(out)  # written schema == ROLLOUT_SCHEMA

    frame = daft.read_parquet(str(out))
    assert frame.schema() == ROLLOUT_SCHEMA
    data = frame.to_pydict()
    assert len(data["episode_id"]) == 3
    assert data["episode_id"][0] == "libero_goal/0/7/0"
    assert data["terminal_failure"][0] == "re_grasp"
    assert data["success"][0] is False
    assert isinstance(data["action"][1], np.ndarray)
    assert data["action"][1].shape == (ACTION_DIM,)
    np.testing.assert_allclose(data["action"][1], [0.1] * ACTION_DIM, rtol=1e-6)
    np.testing.assert_allclose(data["gripper_action"][1], 0.1, rtol=1e-6)
    assert data["embedding"] == [None, None, None]
    assert data["step_idx"] == [0, 1, 2]

    arrow_schema = pq.ParquetFile(out).schema_arrow
    action_field = arrow_schema.field("action")
    assert pa.types.is_fixed_size_list(action_field.type)
    assert action_field.type.list_size == ACTION_DIM
    assert action_field.metadata[b"ARROW:extension:name"] == b"daft.super_extension"
    embedding_field = arrow_schema.field("embedding")
    assert pa.types.is_fixed_size_list(embedding_field.type)
    assert embedding_field.type.list_size == 1024
    assert embedding_field.metadata[b"ARROW:extension:name"] == b"daft.super_extension"

    # Scalar projections remain readable from PyArrow even when nullable tensor columns
    # require Daft for a full logical-type reconstruction.
    projected = pq.read_table(out, columns=["episode_id", "success"])
    assert projected.column("episode_id")[0].as_py() == "libero_goal/0/7/0"


def test_failure_filter_query(tmp_path):
    # The wedge query: select failures and inspect their terminal labels.
    out = write_rows(_toy_episode().to_step_rows(run_id="test"), tmp_path / "ep.parquet")
    failures = (
        daft.read_parquet(str(out))
        .where(daft.col("success") == False)
        .select("terminal_failure")
        .to_pydict()
    )
    assert len(failures["terminal_failure"]) == 3
    assert set(failures["terminal_failure"]) == {"re_grasp"}
