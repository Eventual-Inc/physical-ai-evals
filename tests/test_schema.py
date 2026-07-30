"""The normalized trace keeps episode metadata out of transition rows."""

from physical_ai_evals.schema import EPISODE_SCHEMA, STEP_SCHEMA


def test_episode_and_step_schema_are_normalized_and_lerobot_shaped():
    episodes = [field.name for field in EPISODE_SCHEMA]
    steps = [field.name for field in STEP_SCHEMA]

    assert {"episode_index", "instruction", "success", "length"} <= set(episodes)
    assert {"episode_index", "frame_index", "timestamp", "action"} <= set(steps)
    assert "evaluation_id" not in episodes
    assert "schema_version" not in episodes
    assert "success" not in steps
    assert "primary_video_path" not in steps
    assert "wrist_video_path" not in steps
    assert "primary_video_path" in episodes
    assert "wrist_video_path" in episodes
