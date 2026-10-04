"""Tests for examples/vla_jepa_libero_pro_daft.py.

The example imports LIBERO at module load, so these tests run only in the
example's own environment:

    uv sync --script examples/vla_jepa_libero_pro_daft.py
    PY=$(uv python find --script examples/vla_jepa_libero_pro_daft.py)
    uv pip install --python "$PY" pytest
    "$PY" -m pytest tests/test_vla_jepa_libero_pro_daft.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("libero", reason="the example's uv script environment provides LIBERO")


def _example() -> ModuleType:
    name = "vla_jepa_libero_pro_daft_example"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parents[1] / "examples" / "vla_jepa_libero_pro_daft.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _observation(identity: int) -> dict[str, np.ndarray]:
    return {
        "image": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "state": np.full(8, identity, dtype=np.float32),
    }


def test_variable_length_rollouts_only_infer_for_ready_lanes() -> None:
    example = _example()

    class FakePolicy:
        device = "cpu"

        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def predict(self, images, telemetry, instructions):
            identities = [int(state[0]) for state in telemetry]
            self.calls.append(identities)
            chunks = np.zeros(
                (len(identities), example.ACTION_HORIZON, example.ACTION_DIM),
                dtype=np.float32,
            )
            for index, identity in enumerate(identities):
                chunks[index, :, 0] = identity
            return chunks

    class FakeEpisode:
        suite_variant = "fake_suite"
        perturbation = "fake"
        task = "fake_task"

        def __init__(self, identity: int, length: int) -> None:
            self.identity = identity
            self.length = length
            self.initial_state_id = identity
            self.instruction = f"instruction-{identity}"
            self.steps = 0

        def reset(self):
            return _observation(self.identity)

        def step(self, action):
            assert action[0] == self.identity
            self.steps += 1
            return _observation(self.identity), self.steps >= self.length

    policy = FakePolicy()
    environments = [FakeEpisode(1, 2), FakeEpisode(2, 9), FakeEpisode(3, 3)]
    specs = [
        {
            "episode_key": f"episode-{environment.identity}",
            "initial_state_id": environment.identity,
            "max_steps": 20,
            "video_path": None,
        }
        for environment in environments
    ]

    results = example._run_rollouts(policy, environments, specs)

    assert [result.control_steps for result in results] == [2, 9, 3]
    assert all(result.success for result in results)
    assert policy.calls == [[1, 2, 3], [2]]


def test_rollout_is_one_lazy_daft_operation() -> None:
    example = _example()
    from daft import Expression, col
    from daft.functions import to_struct

    orchestrator = example.RolloutOrchestrator("model", "rev", "qwen", "rev", "jepa", "rev")
    frame = example.daft.from_pylist(
        [
            {
                "episode_key": "episode-0",
                "repo_id": "repo",
                "repo_revision": "rev",
                "suite_variant": "suite",
                "perturbation": "none",
                "task": "task",
                "initial_state_id": 0,
                "max_steps": 1,
                "environment_seed": 7,
                "video_path": None,
            }
        ]
    )

    result = orchestrator.rollout(to_struct(*(col(name) for name in example._EPISODE_SPEC_COLUMNS)))

    assert isinstance(result, Expression)
    assert frame.with_column("result", result).column_names[-1] == "result"
