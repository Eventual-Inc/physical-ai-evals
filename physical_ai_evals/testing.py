"""Deterministic CPU benchmark for full-pipeline conformance tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import daft
import numpy as np

from physical_ai_evals.libero import _episode_specs
from physical_ai_evals.policy import Observation, PolicySpec
from physical_ai_evals.rollout import Benchmark
from physical_ai_evals.schema import ACTION_DIM

MOCK_REVISION = "mock-v1"


class MockEnvironment:
    def __init__(self, task_id: int, seed: int, height: int, width: int) -> None:
        self.task_id = task_id
        self.seed_value = seed
        self.height = height
        self.width = width
        self.time = 0
        self.init_state_id = 0

    def _observation(self) -> dict[str, np.ndarray]:
        pixel = (self.time * 7 + self.task_id + self.init_state_id) % 256
        return {
            "agentview_image": np.full((self.height, self.width, 3), pixel, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full(
                (self.height, self.width, 3), 255 - pixel, dtype=np.uint8
            ),
            "robot0_eef_pos": np.array([0.1 * self.time, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.04 - 0.001 * self.time, -0.04], dtype=np.float32),
        }

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def reset(self) -> dict[str, np.ndarray]:
        self.time = 0
        return self._observation()

    def set_init_state(self, init_state: Any) -> dict[str, np.ndarray]:
        self.time = 0
        self.init_state_id = int(np.asarray(init_state).ravel()[0])
        return self._observation()

    def step(self, _action: Any):
        self.time += 1
        done = self.time >= 5 + (self.task_id % 3)
        reward = float(done and self.task_id % 2 == 0)
        return self._observation(), reward, done, {}

    def check_success(self) -> bool:
        return self.time >= 5 and self.task_id % 2 == 0

    def close(self) -> None:
        return None


@dataclass
class MockRuntime:
    camera_height: int = 16
    camera_width: int = 16

    def __post_init__(self) -> None:
        self.environment: MockEnvironment | None = None
        self.key: tuple[int, int] | None = None

    def open(self, spec: Mapping[str, Any]):
        task_id = int(spec["task_id"])
        seed = int(spec["seed"])
        key = (task_id, seed)
        if self.key != key:
            self.close()
            self.environment = MockEnvironment(task_id, seed, self.camera_height, self.camera_width)
            self.key = key
        return (
            self.environment,
            f"perform mock task {task_id}",
            np.array([spec["init_state_id"]], dtype=np.float32),
            f"mock_task_{task_id}",
        )

    def close(self) -> None:
        if self.environment is not None:
            self.environment.close()
        self.environment = None
        self.key = None


class MockPolicy:
    action_dim = ACTION_DIM
    control_mode = "relative"

    def __init__(self, gain: float = 0.03) -> None:
        self.gain = gain
        self.time = 0
        self.instruction_bias = 0.0

    def reset(self, instruction: str) -> None:
        self.time = 0
        self.instruction_bias = float(len(instruction) % 7)

    def act(self, observation: Observation) -> np.ndarray:
        del observation
        self.time += 1
        action = np.full(
            ACTION_DIM,
            self.gain * self.time + 0.01 * self.instruction_bias,
            dtype=np.float32,
        )
        action[-1] = 1.0 if self.time % 2 else -1.0
        return action

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class _MockPolicyFactory:
    gain: float
    counter_path: str | None

    def __call__(self) -> MockPolicy:
        if self.counter_path is not None:
            with Path(self.counter_path).open("a", encoding="utf-8") as stream:
                stream.write("initialized\n")
        return MockPolicy(self.gain)


def mock_policy(
    *,
    gain: float = 0.03,
    counter_path: str | Path | None = None,
) -> PolicySpec:
    return PolicySpec(
        factory=_MockPolicyFactory(gain, None if counter_path is None else str(counter_path)),
        policy_id="physical-ai-evals/mock-policy",
        revision=MOCK_REVISION,
        camera_height=16,
        camera_width=16,
        num_steps_wait=0,
        metadata={"gain": gain},
    )


def mock_benchmark(
    *,
    task_ids: Sequence[int] = (0, 1, 2),
    episodes: int = 2,
    seed: int = 7,
) -> Benchmark:
    ids = list(task_ids)
    tasks = daft.from_pydict(
        {
            "suite": ["mock"] * len(ids),
            "suite_variant": [None] * len(ids),
            "perturbation": [None] * len(ids),
            "task_id": ids,
            "task_key": [str(task_id) for task_id in ids],
            "task_name": [f"mock_task_{task_id}" for task_id in ids],
            "instruction": [f"perform mock task {task_id}" for task_id in ids],
            "bddl_path": [None] * len(ids),
            "init_path": [None] * len(ids),
        }
    )
    return Benchmark(
        name="mock",
        revision=MOCK_REVISION,
        specs=_episode_specs(
            tasks,
            benchmark="mock",
            revision=MOCK_REVISION,
            episodes=episodes,
            seed=seed,
            max_steps=20,
        ),
        runtime_factory=MockRuntime,
        metadata={"tasks": ids},
    )
