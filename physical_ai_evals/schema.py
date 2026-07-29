"""Normalized episode and step schemas for evaluation output."""

from __future__ import annotations

import daft
from daft import DataType

ACTION_DIM = 7
STATE_DIM = 8
EEF_POS_DIM = 3
SCHEMA_VERSION = "eval-v1"

# Evaluation output deliberately mirrors LeRobot's two-level model: one
# metadata/outcome row per episode and one transition row per frame. The files
# are an evaluation trace, not a training dataset, so simulator provenance and
# outcome fields live in the episode table instead of LeRobot's stats metadata.
EPISODE_SCHEMA = daft.Schema.from_field_name_and_types(
    [
        ("episode_index", DataType.int64()),
        ("episode_key", DataType.string()),
        ("episode_id", DataType.string()),
        ("suite", DataType.string()),
        ("suite_variant", DataType.string()),
        ("perturbation", DataType.string()),
        ("task_id", DataType.int64()),
        ("task_key", DataType.string()),
        ("task_name", DataType.string()),
        ("instruction", DataType.string()),
        ("bddl_path", DataType.string()),
        ("init_path", DataType.string()),
        ("init_state_id", DataType.int64()),
        ("seed", DataType.int64()),
        ("success", DataType.bool()),
        ("length", DataType.int64()),
        ("reward", DataType.float32()),
        ("terminal_failure", DataType.string()),
        ("primary_video_path", DataType.string()),
        ("wrist_video_path", DataType.string()),
    ]
)

STEP_SCHEMA = daft.Schema.from_field_name_and_types(
    [
        ("episode_index", DataType.int64()),
        ("episode_key", DataType.string()),
        ("episode_id", DataType.string()),
        ("task_key", DataType.string()),
        ("frame_index", DataType.int64()),
        ("timestamp", DataType.float32()),
        ("action", DataType.tensor(DataType.float32(), shape=(ACTION_DIM,))),
        ("reward", DataType.float32()),
        ("next.done", DataType.bool()),
        (
            "observation.state",
            DataType.tensor(DataType.float32(), shape=(STATE_DIM,)),
        ),
        (
            "observation.eef_position",
            DataType.tensor(DataType.float32(), shape=(EEF_POS_DIM,)),
        ),
        ("observation.gripper", DataType.float32()),
    ]
)

__all__ = [
    "ACTION_DIM",
    "EEF_POS_DIM",
    "EPISODE_SCHEMA",
    "SCHEMA_VERSION",
    "STATE_DIM",
    "STEP_SCHEMA",
]
