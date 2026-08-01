"""Contract tests for the persistent daft-cuTile VLA-JEPA adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from physical_ai_evals.cutile_vla_jepa import (
    CUTILE_DAFT_REVISION,
    CUTILE_VLA_ACTION_HORIZON,
    CUTILE_VLA_STATIC_BATCH_SIZE,
    CutileVLAJEPAPolicy,
    _normal_noise,
)
from physical_ai_evals.policy import vla_jepa_cutile
from physical_ai_evals.rollout import _policy_observation


class _Engine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.resets = 0
        self.closes = 0

    def reset(self) -> None:
        self.resets += 1

    def predict_action_host(
        self,
        images: np.ndarray,
        telemetry: np.ndarray,
        instructions: list[str],
        *,
        noise: np.ndarray,
        active_rows: int,
    ) -> np.ndarray:
        self.calls.append(
            {
                "images": images.copy(),
                "telemetry": telemetry.copy(),
                "instructions": list(instructions),
                "noise": noise.copy(),
                "active_rows": active_rows,
            }
        )
        output = np.empty(
            (active_rows, CUTILE_VLA_ACTION_HORIZON, 7),
            dtype=np.float32,
        )
        for row in range(active_rows):
            for step in range(CUTILE_VLA_ACTION_HORIZON):
                output[row, step] = 100 * len(self.calls) + 10 * row + step
        return output

    def close(self) -> None:
        self.closes += 1


def _observation(row: int, *, episode_key: str | None = None) -> dict[str, Any]:
    return {
        "image": np.full((224, 224, 3), 10 + row, dtype=np.uint8),
        "wrist_image": np.full((224, 224, 3), 20 + row, dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32) + row,
        "instruction": f"observation instruction {row}",
        "episode_key": episode_key or f"episode-{row}",
        "seed": 7,
    }


def test_cutile_policy_prewarms_b4_and_reuses_all_seven_actions() -> None:
    engine = _Engine()
    policy = CutileVLAJEPAPolicy("/policy", "/qwen", _engine=engine)
    observations = [_observation(0), _observation(1)]
    policy.reset_batch(["first", "second"])

    policy.prepare_batch(observations)
    assert len(engine.calls) == 2
    assert all(call["active_rows"] == 2 for call in engine.calls)
    assert all(np.count_nonzero(call["noise"]) == 0 for call in engine.calls)

    actions = [policy.act_batch(observations) for _ in range(7)]
    assert len(engine.calls) == 3
    for step, action in enumerate(actions):
        np.testing.assert_array_equal(action[0], np.full(7, 300 + step, np.float32))
        np.testing.assert_array_equal(action[1], np.full(7, 310 + step, np.float32))

    policy.act_batch(observations)
    assert len(engine.calls) == 4
    assert policy.batch_profile() == {
        "cutile_prewarm_calls": 2,
        "cutile_action_chunk_calls": 2,
        "cutile_actions_reused_per_chunk": 7,
    }

    staged = engine.calls[2]
    assert staged["images"].shape == (4, 2, 224, 224, 3)
    assert staged["telemetry"].shape == (4, 8)
    assert staged["noise"].shape == (4, 7, 7)
    assert staged["instructions"] == [
        "observation instruction 0",
        "observation instruction 1",
        "",
        "",
    ]
    assert staged["images"][0, 0, 0, 0, 0] == 10
    assert staged["images"][0, 1, 0, 0, 0] == 20
    assert not staged["images"][2:].any()
    assert not staged["telemetry"][2:].any()
    assert not staged["noise"][2:].any()

    policy.close()
    assert engine.closes == 1


def test_cutile_noise_is_episode_stable_and_cohort_order_independent() -> None:
    expected = _normal_noise(episode_key="episode-a", seed=11, chunk_index=3)
    np.testing.assert_array_equal(
        expected,
        _normal_noise(episode_key="episode-a", seed=11, chunk_index=3),
    )
    assert not np.array_equal(
        expected,
        _normal_noise(episode_key="episode-a", seed=11, chunk_index=4),
    )
    assert not np.array_equal(
        expected,
        _normal_noise(episode_key="episode-b", seed=11, chunk_index=3),
    )

    engine = _Engine()
    policy = CutileVLAJEPAPolicy("/policy", "/qwen", _engine=engine)
    policy.reset_batch(["a", "b"])
    first = policy._inputs(
        [_observation(0, episode_key="episode-a"), _observation(1, episode_key="episode-b")],
        chunk_index=3,
    )[3]
    policy.reset_batch(["b", "a"])
    second = policy._inputs(
        [_observation(1, episode_key="episode-b"), _observation(0, episode_key="episode-a")],
        chunk_index=3,
    )[3]
    np.testing.assert_array_equal(first[0], second[1])
    np.testing.assert_array_equal(first[1], second[0])


def test_cutile_policy_fails_closed_on_invalid_fixed_inputs() -> None:
    policy = CutileVLAJEPAPolicy("/policy", "/qwen", _engine=_Engine())
    with pytest.raises(ValueError, match="1 to 4"):
        policy.reset_batch(["task"] * 5)

    policy.reset_batch(["task"])
    observation = _observation(0)
    observation.pop("wrist_image")
    with pytest.raises(ValueError, match="wrist camera"):
        policy.act(observation)

    observation = _observation(0)
    observation["state"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        policy.act(observation)

    observation = _observation(0)
    observation["image"] = np.zeros((224, 224, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype uint8"):
        policy.act(observation)

    observation = _observation(0)
    observation.pop("episode_key")
    with pytest.raises(ValueError, match="episode_key"):
        policy.act(observation)


def test_cutile_policy_spec_pins_engine_and_omits_vjepa2() -> None:
    spec = vla_jepa_cutile()
    assert spec.metadata["engine"] == "daft_cutile"
    assert spec.metadata["engine_revision"] == CUTILE_DAFT_REVISION
    assert spec.metadata["static_batch_size"] == CUTILE_VLA_STATIC_BATCH_SIZE
    assert "qwen3_vl" in spec.metadata
    assert "vjepa2" not in spec.metadata


def test_cutile_policy_fails_closed_on_transfer_counter_regression() -> None:
    policy = CutileVLAJEPAPolicy("/policy", "/qwen", _engine=_Engine())
    policy.reset_batch(["a", "b"])
    policy._prewarm_calls = 2
    policy._action_chunk_calls = 11
    calls = 13
    input_bytes = calls * 2 * (2 * 224 * 224 * 3 + 4_908)
    output_bytes = calls * 2 * 7 * 7 * 4

    def counters(sync_count: int) -> Any:
        payload = {
            "counters_enabled": True,
            "vla_action_input_h2d_bytes": input_bytes,
            "vla_action_input_d2d_bytes": 0,
            "vla_action_output_d2h_bytes": output_bytes,
            "vla_action_host_visible_sync_count": sync_count,
        }
        return SimpleNamespace(**payload, as_dict=lambda: payload)

    policy._drain_counters = lambda: counters(calls)
    assert policy.batch_profile()["cutile_device_resident"] is True
    policy._drain_counters = lambda: counters(calls + 1)
    with pytest.raises(RuntimeError, match="device-residency counters changed"):
        policy.batch_profile()


def test_rollout_observation_carries_stable_noise_identity() -> None:
    source = {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    normalized = _policy_observation(
        source,
        "task",
        {"episode_key": "stable-key", "seed": 23},
    )
    assert normalized["episode_key"] == "stable-key"
    assert normalized["seed"] == 23
