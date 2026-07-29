from __future__ import annotations

import daft
from daft import DataType

ACTION_DIM = 7
STATE_DIM = 8
EEF_POS_DIM = 3
EMBEDDING_DIM = 1024
SCHEMA_VERSION = "rollout-v2"

TERMINAL_FAILURE_LABELS = (
    "re_grasp",
    "no_grasp",
    "drop_no_recover",
    "wrong_object",
    "missed_target",
    "timeout",
    "collision",
    "unlabeled",
)


def rollout_schema() -> daft.Schema:
    """One row per step; episode fields denormalized onto each step."""
    f32 = DataType.float32()
    return daft.Schema.from_field_name_and_types(
        [
            ("schema_version", DataType.string()),
            ("episode_id", DataType.string()),
            ("run_id", DataType.string()),
            ("model", DataType.string()),
            ("policy_type", DataType.string()),
            ("source", DataType.string()),
            ("suite", DataType.string()),
            ("task_id", DataType.int32()),
            ("task_name", DataType.string()),
            ("instruction", DataType.string()),
            ("bddl_file", DataType.string()),
            ("init_state_id", DataType.int32()),
            ("seed", DataType.int64()),
            ("control_mode", DataType.string()),
            ("success", DataType.bool()),
            ("terminal_failure", DataType.string()),
            ("num_steps", DataType.int32()),
            ("step_idx", DataType.int32()),
            ("action", DataType.tensor(f32, shape=(ACTION_DIM,))),
            ("reward", f32),
            ("done", DataType.bool()),
            ("state", DataType.tensor(f32, shape=(STATE_DIM,))),
            ("eef_pos", DataType.tensor(f32, shape=(EEF_POS_DIM,))),
            ("gripper_state", f32),
            ("gripper_action", f32),
            ("object_poses", DataType.string()),
            ("frame_path", DataType.string()),
            ("wrist_path", DataType.string()),
            ("video_path", DataType.string()),
            ("embedding", DataType.embedding(f32, EMBEDDING_DIM)),
        ]
    )


ROLLOUT_SCHEMA: daft.Schema = rollout_schema()
COLUMNS: tuple[str, ...] = tuple(ROLLOUT_SCHEMA.column_names())


def empty_step_row() -> dict[str, object]:
    """Return all schema columns initialized to ``None``."""
    return {name: None for name in COLUMNS}


def validate_rows(rows: list[dict[str, object]]) -> daft.DataFrame:
    """Build and materialize a DataFrame under ``ROLLOUT_SCHEMA``."""
    frame = daft.from_pylist(rows)
    return frame.select(
        *(daft.col(field.name).cast(field.dtype) for field in ROLLOUT_SCHEMA)
    ).collect()
