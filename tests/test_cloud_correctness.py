"""CPU-only regression tests for cloud rollout identity, seeding, and resume safety."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import get_args

import numpy as np
import pytest
from daft import Series

import physical_ai_evals.cloud.rollout_udf as rollout_udf
from physical_ai_evals.cloud.sweep import (
    OPENVLA_REVISIONS,
    enumerate_specs,
    evaluation_fingerprint,
    immutable_model_id,
    resolve_openvla_config,
    write_evaluation_manifest,
)
from physical_ai_evals.core.episode import Episode, Step
from physical_ai_evals.core.writer import write_episode, write_rows


def _episode(*, model: str) -> Episode:
    return Episode(
        episode_id="libero_goal/0/0/7",
        source="libero",
        instruction="put the bowl on the plate",
        steps=(Step(timestep=0, action=np.zeros(7, np.float32), done=True),),
        success=True,
        model=model,
        policy_type="openvla",
        suite="libero_goal",
        task_id=0,
        metadata={"init_state_id": 0, "seed": 7, "control_mode": "relative"},
    )


def test_openvla_resolution_is_single_suite_pinned_and_fail_closed():
    model_id, revision, unnorm_key = resolve_openvla_config(["libero_goal"])
    assert model_id == "openvla/openvla-7b-finetuned-libero-goal"
    assert revision == OPENVLA_REVISIONS[model_id]
    assert unnorm_key == "libero_goal"

    with pytest.raises(ValueError, match="exactly one suite"):
        resolve_openvla_config(["libero_goal", "libero_spatial"])
    with pytest.raises(ValueError, match="registered for 'libero_spatial'"):
        resolve_openvla_config(
            ["libero_goal"], "openvla/openvla-7b-finetuned-libero-spatial"
        )
    with pytest.raises(ValueError, match="compatibility declaration"):
        resolve_openvla_config(
            ["libero_goal"], "research/custom-openvla", model_revision="a" * 40
        )
    with pytest.raises(ValueError, match="40-character commit SHA"):
        resolve_openvla_config(
            ["libero_goal"],
            "research/custom-openvla",
            unnorm_key="libero_goal",
            model_revision="main",
        )
    with pytest.raises(ValueError, match="immutable model_revision"):
        rollout_udf._build_policy("openvla", model_id, "", "cpu", "libero_goal", "sdpa")


def test_evaluation_fingerprint_is_canonical_and_config_sensitive():
    first = evaluation_fingerprint(
        {"model": "repo@abc", "camera": {"width": 256, "height": 256}}
    )
    reordered = evaluation_fingerprint(
        {"camera": {"height": 256, "width": 256}, "model": "repo@abc"}
    )
    changed = evaluation_fingerprint(
        {"camera": {"height": 224, "width": 224}, "model": "repo@abc"}
    )
    assert first == reordered
    assert first != changed


def test_evaluation_manifest_binds_namespace_to_config(tmp_path):
    config = {"model": "repo@" + "a" * 40, "seed": 7}
    evaluation_id = evaluation_fingerprint(config)
    manifest = write_evaluation_manifest(tmp_path, evaluation_id, config)
    payload = json.loads(manifest.read_text())
    assert payload["evaluation_id"] == evaluation_id
    assert payload["config"] == config
    with pytest.raises(ValueError, match="does not match"):
        write_evaluation_manifest(tmp_path, "wrong-id", config)


def test_resume_requires_valid_matching_parquet(tmp_path):
    model = immutable_model_id(
        "openvla/openvla-7b-finetuned-libero-goal",
        OPENVLA_REVISIONS["openvla/openvla-7b-finetuned-libero-goal"],
    )
    part = write_episode(_episode(model=model), tmp_path, run_id="evaluation-deadbeef")

    def specs_for(model_id: str) -> dict:
        return (
            enumerate_specs(
                ["libero_goal"],
                [0],
                1,
                7,
                out_dir=str(tmp_path),
                policy_type="openvla",
                model_id=model_id,
            )
            .select("suite", "task_id", "init_state_id", "seed")
            .to_pydict()
        )

    pending = {"suite": ["libero_goal"], "task_id": [0], "init_state_id": [0], "seed": [7]}
    empty = {"suite": [], "task_id": [], "init_state_id": [], "seed": []}

    # a valid, matching part is skipped
    assert specs_for(model) == empty
    # a part written by a different model is not resumable
    assert specs_for("different/model@" + "b" * 40) == pending

    part.write_bytes(b"truncated parquet")
    # an unreadable part is not resumable
    assert specs_for(model) == pending

    # the first run of an evaluation has no parts on the volume at all
    part.unlink()
    assert specs_for(model) == pending
    assert (
        enumerate_specs(
            ["libero_goal"], [0], 1, 7,
            out_dir=str(tmp_path / "never-written"),
            policy_type="openvla", model_id=model,
        )
        .select("suite", "task_id", "init_state_id", "seed")
        .to_pydict()
        == pending
    )


def test_write_rows_failure_preserves_previous_part_and_cleans_temp(tmp_path, monkeypatch):
    episode = _episode(model="model@" + "a" * 40)
    rows = episode.to_step_rows(run_id="evaluation-deadbeef")
    target = write_rows(rows, tmp_path / "episode.parquet")
    previous = target.read_bytes()

    def fail_after_partial_write(_frame, path, **_kwargs):
        path.write_bytes(b"partial")
        raise RuntimeError("simulated interrupted parquet write")

    monkeypatch.setattr(
        "physical_ai_evals.core.writer._write_parquet_frame",
        fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        write_rows(rows, target)

    assert target.read_bytes() == previous
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_rollout_row_seed_controls_env_and_cache_key(tmp_path, monkeypatch):
    # Daft retains the undecorated worker class as the generic type parameter.
    worker_class = get_args(rollout_udf.LiberoRollout.__orig_bases__[0])[0]
    monkeypatch.setattr(rollout_udf, "_build_policy", lambda *_args, **_kwargs: object())
    process_seeds: list[int] = []
    monkeypatch.setattr(rollout_udf, "_seed_process", process_seeds.append)

    made_env_seeds: list[int] = []
    reseeded_envs: list[int] = []
    episode_seeds: list[tuple[int, int, str]] = []

    class FakeEnv:
        def __init__(self, seed):
            self.creation_seed = seed
            self.closed = False

        def seed(self, value):
            reseeded_envs.append(value)

        def close(self):
            self.closed = True

    def fake_make_env(_suite, _task_id, *, camera_height, camera_width, seed):
        assert (camera_height, camera_width) == (256, 256)
        made_env_seeds.append(seed)
        return FakeEnv(seed), SimpleNamespace(language="instruction", name="task", bddl_file="x")

    def fake_run_episode(env, _policy, **kwargs):
        episode_seeds.append((env.creation_seed, kwargs["seed"], kwargs["model"]))
        return SimpleNamespace(
            episode_id=kwargs["episode_id"],
            success=True,
            num_steps=1,
            reward=1.0,
            terminal_failure=None,
            parquet_path="unused.parquet",
        )

    monkeypatch.setattr("physical_ai_evals.bench.libero.make_env", fake_make_env)
    monkeypatch.setattr(
        "physical_ai_evals.bench.libero.libero_init_states", lambda _suite, _task_id: [np.zeros(1)]
    )
    monkeypatch.setattr("physical_ai_evals.bench.libero.run_episode", fake_run_episode)

    worker = worker_class(
        policy_type="openvla",
        out_dir=str(tmp_path),
        model_id="research/model",
        model_revision="a" * 40,
        unnorm_key="libero_goal",
        write_frames=False,
        write_video=False,
    )
    result = worker.rollout(
        Series.from_pylist(["libero_goal", "libero_goal", "libero_goal"]),
        Series.from_pylist([0, 0, 0]),
        Series.from_pylist([0, 0, 0]),
        Series.from_pylist([7, 7, 8]),
    )

    assert len(result) == 3
    assert made_env_seeds == [7, 8]
    assert reseeded_envs == [7, 7, 8]
    assert [row[:2] for row in episode_seeds] == [(7, 7), (7, 7), (8, 8)]
    assert {row[2] for row in episode_seeds} == {"research/model@" + "a" * 40}
    assert process_seeds.count(7) >= 2
    assert process_seeds.count(8) >= 1
