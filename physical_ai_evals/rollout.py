"""Daft-native batch evaluation: specs -> rollout -> episodes + steps."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

import daft
import numpy as np
from daft import DataFrame, DataType, col, lit
from daft.functions import explode, unnest

from physical_ai_evals.policy import Policy, PolicySpec
from physical_ai_evals.provenance import evaluation_manifest, write_manifest
from physical_ai_evals.schema import (
    ACTION_DIM,
    EEF_POS_DIM,
    EPISODE_SCHEMA,
    STATE_DIM,
    STEP_SCHEMA,
)


class Runtime(Protocol):
    """Stateful benchmark runtime owned by one rollout actor."""

    def open(self, spec: Mapping[str, Any]) -> tuple[Any, str, Any, str | None]: ...

    def open_batch(
        self,
        specs: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, list[str], list[Any], list[str | None]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Benchmark:
    """A lazy episode-spec frame plus the runtime factory that executes it."""

    name: str
    revision: str
    specs: DataFrame
    runtime_factory: Callable[..., Runtime]
    metadata: Mapping[str, Any] | None = None


_SPEC_COLUMNS = (
    "episode_index",
    "episode_key",
    "episode_id",
    "benchmark",
    "benchmark_revision",
    "suite",
    "suite_variant",
    "perturbation",
    "task_id",
    "task_key",
    "task_name",
    "instruction",
    "bddl_path",
    "init_path",
    "init_state_id",
    "seed",
    "max_steps",
)

_ROLLOUT_DTYPE = DataType.struct(
    {
        "resolved_task_name": DataType.string(),
        "resolved_instruction": DataType.string(),
        "success": DataType.bool(),
        "num_steps": DataType.int64(),
        "reward": DataType.float32(),
        "terminal_failure": DataType.string(),
        "primary_video_path": DataType.string(),
        "wrist_video_path": DataType.string(),
        "steps": DataType.list(
            DataType.struct(
                {
                    "frame_index": DataType.int64(),
                    "timestamp": DataType.float32(),
                    "action": DataType.tensor(DataType.float32(), shape=(ACTION_DIM,)),
                    "reward": DataType.float32(),
                    "next.done": DataType.bool(),
                    "observation.state": DataType.tensor(DataType.float32(), shape=(STATE_DIM,)),
                    "observation.eef_position": DataType.tensor(
                        DataType.float32(), shape=(EEF_POS_DIM,)
                    ),
                    "observation.gripper": DataType.float32(),
                }
            )
        ),
    }
)


def _seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _derotate(image: Any) -> np.ndarray:
    return np.asarray(image)[::-1, ::-1]


def _eef_position(observation: Mapping[str, Any]) -> np.ndarray | None:
    value = observation.get("robot0_eef_pos")
    if value is None:
        return None
    return np.asarray(value, np.float32).ravel()[:EEF_POS_DIM]


def _gripper(observation: Mapping[str, Any]) -> float | None:
    value = observation.get("robot0_gripper_qpos")
    if value is None:
        return None
    qpos = np.asarray(value, np.float32).ravel()
    return float(qpos[0] - qpos[1]) if qpos.size >= 2 else float(qpos[0])


def _proprio(observation: Mapping[str, Any]) -> np.ndarray | None:
    eef = observation.get("robot0_eef_pos")
    gripper = observation.get("robot0_gripper_qpos")
    if eef is None or gripper is None:
        return None

    from physical_ai_evals.geometry import quat_xyzw_to_axis_angle

    parts = [np.asarray(eef, np.float32).ravel()[:3]]
    quaternion = observation.get("robot0_eef_quat")
    if quaternion is not None:
        parts.append(quat_xyzw_to_axis_angle(np.asarray(quaternion).reshape(1, 4))[0])
    gripper_array = np.asarray(gripper, np.float32).ravel()
    parts.append(gripper_array[:2] if gripper_array.size >= 2 else gripper_array)
    state = np.concatenate(parts).astype(np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"LIBERO proprioception must have shape {(STATE_DIM,)}, got {state.shape}")
    return state


class _EpisodeVideos:
    """Stream cameras to atomic, deterministic episode video paths."""

    def __init__(self, root: str | Path, episode_key: str, fps: int) -> None:
        self.directory = Path(root) / episode_key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writers: dict[str, Any] = {}
        self._temporary: dict[str, Path] = {}

    def append(self, role: str, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        if role not in self._writers:
            import imageio.v2 as imageio

            fd, name = tempfile.mkstemp(dir=self.directory, prefix=f".{role}.", suffix=".mp4")
            os.close(fd)
            path = Path(name)
            self._temporary[role] = path
            self._writers[role] = imageio.get_writer(path, fps=self.fps, codec="libx264")
        self._writers[role].append_data(np.asarray(frame, dtype=np.uint8))

    def finish(self) -> tuple[str | None, str | None]:
        paths: dict[str, str] = {}
        for role, writer in self._writers.items():
            writer.close()
            temporary = self._temporary[role]
            final = self.directory / f"{role}.mp4"
            os.replace(temporary, final)
            paths[role] = str(Path("videos") / self.directory.name / final.name)
        self._writers.clear()
        self._temporary.clear()
        return paths.get("primary"), paths.get("wrist")

    def abort(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for path in self._temporary.values():
            path.unlink(missing_ok=True)
        self._writers.clear()
        self._temporary.clear()


def _run_episode(
    runtime: Runtime,
    policy: Policy,
    spec: Mapping[str, Any],
    *,
    media_dir: str | None,
    num_steps_wait: int,
    frames_per_second: int,
) -> dict[str, Any]:
    seed = int(spec["seed"])
    _seed_process(seed)
    environment, instruction, init_state, task_name = runtime.open(spec)
    seed_environment = getattr(environment, "seed", None)
    if callable(seed_environment):
        seed_environment(seed)

    environment.reset()
    observation = environment.set_init_state(init_state)
    dummy_action = [0.0] * int(getattr(policy, "action_dim", ACTION_DIM))
    for _ in range(num_steps_wait):
        observation = environment.step(dummy_action)[0]

    policy.reset(instruction)
    videos = (
        _EpisodeVideos(media_dir, str(spec["episode_key"]), frames_per_second)
        if media_dir is not None
        else None
    )
    steps: list[dict[str, Any]] = []
    final_reward = 0.0
    try:
        for frame_index in range(int(spec["max_steps"])):
            primary = _derotate(observation["agentview_image"])
            wrist = (
                _derotate(observation["robot0_eye_in_hand_image"])
                if "robot0_eye_in_hand_image" in observation
                else None
            )
            state = _proprio(observation)
            action = policy.act(
                {
                    "image": primary,
                    "wrist_image": wrist,
                    "state": state,
                    "instruction": instruction,
                }
            )
            action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
            if action.shape != (ACTION_DIM,):
                raise ValueError(
                    f"policy action must have shape {(ACTION_DIM,)}, got {action.shape}"
                )
            if videos is not None:
                videos.append("primary", primary)
                videos.append("wrist", wrist)

            next_observation, final_reward, done, _info = environment.step(action)
            steps.append(
                {
                    "frame_index": frame_index,
                    "timestamp": float(frame_index / frames_per_second),
                    "action": action,
                    "reward": float(final_reward),
                    "next.done": bool(done),
                    "observation.state": state,
                    "observation.eef_position": _eef_position(next_observation),
                    "observation.gripper": _gripper(next_observation),
                }
            )
            observation = next_observation
            if done:
                break

        success = bool(environment.check_success())
        primary_path, wrist_path = videos.finish() if videos is not None else (None, None)
        return {
            "resolved_task_name": task_name or spec.get("task_name"),
            "resolved_instruction": instruction,
            "success": success,
            "num_steps": len(steps),
            "reward": float(final_reward),
            "terminal_failure": None if success else "unlabeled",
            "primary_video_path": primary_path,
            "wrist_video_path": wrist_path,
            "steps": steps,
        }
    except Exception:
        if videos is not None:
            videos.abort()
        raise


def _observation_rows(observations: Any, count: int) -> list[Mapping[str, Any]]:
    """Normalize LIBERO's object array (or a dict-of-batches) to row dictionaries."""
    if isinstance(observations, np.ndarray) and observations.dtype == object:
        rows = observations.tolist()
        if len(rows) == count and all(isinstance(row, Mapping) for row in rows):
            return rows
    if isinstance(observations, Sequence) and not isinstance(observations, (str, bytes, Mapping)):
        rows = list(observations)
        if len(rows) == count and all(isinstance(row, Mapping) for row in rows):
            return cast(list[Mapping[str, Any]], rows)
    if isinstance(observations, Mapping):

        def take(value: Any, index: int) -> Any:
            if isinstance(value, Mapping):
                return {name: take(child, index) for name, child in value.items()}
            array = np.asarray(value)
            if array.ndim > 0 and array.shape[0] == count:
                return array[index]
            return value

        return [
            {name: take(value, index) for name, value in observations.items()}
            for index in range(count)
        ]
    raise TypeError(
        "vector environment observations must be an object array, row sequence, "
        "or dictionary of batched arrays"
    )


class _GpuSampler:
    """Low-rate nvidia-smi sampling for rollout-level utilization evidence."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                first = completed.stdout.splitlines()[0]
                utilization, memory, power = (
                    float(value.strip()) for value in first.split(",")[:3]
                )
                self.samples.append((utilization, memory, power))
            except (IndexError, OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if not self.samples:
            return {
                "gpu_samples": 0,
                "gpu_utilization_mean": None,
                "gpu_utilization_p95": None,
                "gpu_utilization_max": None,
                "gpu_memory_mib_max": None,
                "gpu_power_watts_mean": None,
            }
        values = np.asarray(self.samples, dtype=np.float64)
        return {
            "gpu_samples": len(self.samples),
            "gpu_utilization_mean": float(values[:, 0].mean()),
            "gpu_utilization_p95": float(np.percentile(values[:, 0], 95)),
            "gpu_utilization_max": float(values[:, 0].max()),
            "gpu_memory_mib_max": float(values[:, 1].max()),
            "gpu_power_watts_mean": float(values[:, 2].mean()),
        }


def _policy_observation(
    observation: Mapping[str, Any],
    instruction: str,
) -> dict[str, Any]:
    return {
        "image": _derotate(observation["agentview_image"]),
        "wrist_image": (
            _derotate(observation["robot0_eye_in_hand_image"])
            if "robot0_eye_in_hand_image" in observation
            else None
        ),
        "state": _proprio(observation),
        "instruction": instruction,
    }


def _run_batch(
    runtime: Runtime,
    policy: Policy,
    specs: Sequence[Mapping[str, Any]],
    *,
    media_dir: str | None,
    num_steps_wait: int,
    frames_per_second: int,
    profile: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one fixed policy batch over LIBERO's CPU subprocess environments."""
    if not specs:
        return [], {}
    reset_batch = getattr(policy, "reset_batch", None)
    act_batch = getattr(policy, "act_batch", None)
    if not callable(reset_batch) or not callable(act_batch):
        raise TypeError("policy does not implement reset_batch() and act_batch()")

    seeds = {int(spec["seed"]) for spec in specs}
    if len(seeds) != 1:
        raise ValueError("one rollout batch must use one process seed")
    _seed_process(next(iter(seeds)))

    sampler = _GpuSampler()
    started = time.perf_counter()
    reset_seconds = 0.0
    settle_seconds = 0.0
    policy_seconds = 0.0
    environment_seconds = 0.0
    policy_calls = 0
    transitions = 0

    environment, instructions, init_states, task_names = runtime.open_batch(specs)
    reset_started = time.perf_counter()
    environment.reset()
    observations = _observation_rows(environment.set_init_state(init_states), len(specs))
    reset_seconds += time.perf_counter() - reset_started

    dummy = np.zeros((len(specs), int(getattr(policy, "action_dim", ACTION_DIM))), np.float32)
    settle_started = time.perf_counter()
    for _ in range(num_steps_wait):
        stepped = environment.step(dummy)
        observations = _observation_rows(stepped[0], len(specs))
    settle_seconds += time.perf_counter() - settle_started

    reset_batch(instructions)
    if profile:
        # Sample steady-state control, separately from environment construction
        # and initial-state settling where an idle inference GPU is expected.
        sampler.start()
    videos = [
        (
            _EpisodeVideos(media_dir, str(spec["episode_key"]), frames_per_second)
            if media_dir is not None
            else None
        )
        for spec in specs
    ]
    steps: list[list[dict[str, Any]]] = [[] for _ in specs]
    final_rewards = [0.0 for _ in specs]
    finished = [False for _ in specs]
    try:
        max_frames = max(int(spec["max_steps"]) for spec in specs)
        for frame_index in range(max_frames):
            active = [
                index
                for index, spec in enumerate(specs)
                if not finished[index] and frame_index < int(spec["max_steps"])
            ]
            if not active:
                break

            normalized = [
                _policy_observation(observation, instruction)
                for observation, instruction in zip(observations, instructions, strict=True)
            ]
            policy_started = time.perf_counter()
            actions = np.asarray(act_batch(normalized), dtype=np.float32)
            policy_seconds += time.perf_counter() - policy_started
            policy_calls += 1
            if actions.shape != (len(specs), ACTION_DIM):
                raise ValueError(
                    "policy action batch must have shape "
                    f"{(len(specs), ACTION_DIM)}, got {actions.shape}"
                )
            actions = np.clip(actions, -1.0, 1.0)

            for index in active:
                video = videos[index]
                if video is not None:
                    video.append("primary", normalized[index]["image"])
                    video.append("wrist", normalized[index]["wrist_image"])

            environment_started = time.perf_counter()
            next_batch, rewards, dones, _infos = environment.step(
                actions[active],
                id=active,
            )
            environment_seconds += time.perf_counter() - environment_started
            transitions += len(active)
            next_rows = _observation_rows(next_batch, len(active))
            for offset, index in enumerate(active):
                next_observation = next_rows[offset]
                reward = float(rewards[offset])
                done = bool(dones[offset])
                steps[index].append(
                    {
                        "frame_index": frame_index,
                        "timestamp": float(frame_index / frames_per_second),
                        "action": actions[index],
                        "reward": reward,
                        "next.done": done,
                        "observation.state": normalized[index]["state"],
                        "observation.eef_position": _eef_position(next_observation),
                        "observation.gripper": _gripper(next_observation),
                    }
                )
                observations[index] = next_observation
                final_rewards[index] = reward
                finished[index] = done

        environment_started = time.perf_counter()
        successes = [bool(value) for value in environment.check_success()]
        environment_seconds += time.perf_counter() - environment_started
        results: list[dict[str, Any]] = []
        for index, spec in enumerate(specs):
            video = videos[index]
            primary_path, wrist_path = video.finish() if video is not None else (None, None)
            results.append(
                {
                    "resolved_task_name": task_names[index] or spec.get("task_name"),
                    "resolved_instruction": instructions[index],
                    "success": successes[index],
                    "num_steps": len(steps[index]),
                    "reward": final_rewards[index],
                    "terminal_failure": None if successes[index] else "unlabeled",
                    "primary_video_path": primary_path,
                    "wrist_video_path": wrist_path,
                    "steps": steps[index],
                }
            )
    except Exception:
        for video in videos:
            if video is not None:
                video.abort()
        raise
    finally:
        gpu = sampler.finish()

    elapsed = time.perf_counter() - started
    return results, {
        "batch_size": len(specs),
        "episode_keys": [str(spec["episode_key"]) for spec in specs],
        "wall_seconds": elapsed,
        "reset_seconds": reset_seconds,
        "settle_seconds": settle_seconds,
        "policy_seconds": policy_seconds,
        "environment_seconds": environment_seconds,
        "other_seconds": max(
            0.0,
            elapsed - reset_seconds - settle_seconds - policy_seconds - environment_seconds,
        ),
        "policy_calls": policy_calls,
        "transitions": transitions,
        "transitions_per_second": transitions / elapsed if elapsed else None,
        **gpu,
    }


@daft.cls(use_process=False, max_concurrency=1, name_override="RolloutActor")
class RolloutActor:
    """One loaded policy plus one reusable benchmark runtime.

    ``evaluate`` invokes this actor in checkpointed cohorts. The decorated
    batch method is also a normal Daft expression when passed a struct column.
    """

    def __init__(
        self,
        policy: PolicySpec,
        runtime_factory: Callable[..., Runtime],
        *,
        media_dir: str | None,
        profile: bool,
    ) -> None:
        self.policy_spec = policy
        self.runtime = partial(
            runtime_factory,
            camera_height=policy.camera_height,
            camera_width=policy.camera_width,
        )()
        prepare_runtime = getattr(self.runtime, "prepare", None)
        if callable(prepare_runtime):
            prepare_runtime()
        self.policy: Policy | None = policy.factory()
        if int(self.policy.action_dim) != ACTION_DIM:
            raise ValueError(
                f"policy action_dim must be {ACTION_DIM}, got {self.policy.action_dim}"
            )
        if str(self.policy.control_mode) != policy.control_mode:
            raise ValueError(
                "policy factory control_mode does not match PolicySpec: "
                f"{self.policy.control_mode!r} != {policy.control_mode!r}"
            )
        self.media_dir = media_dir
        self.profile_enabled = profile
        self._profiles: list[dict[str, Any]] = []

    @daft.method.batch(batch_size=8, return_dtype=_ROLLOUT_DTYPE)
    def rollout(
        self,
        specs: daft.Series | Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = specs.to_pylist() if isinstance(specs, daft.Series) else list(specs)
        if self.policy is None:
            raise RuntimeError("rollout actor is closed")
        reset_batch = getattr(self.policy, "reset_batch", None)
        act_batch = getattr(self.policy, "act_batch", None)
        if len(rows) > 1 and (not callable(reset_batch) or not callable(act_batch)):
            results = []
            for spec in rows:
                started = time.perf_counter()
                result = _run_episode(
                    self.runtime,
                    self.policy,
                    spec,
                    media_dir=self.media_dir,
                    num_steps_wait=self.policy_spec.num_steps_wait,
                    frames_per_second=self.policy_spec.frames_per_second,
                )
                results.append(result)
                if self.profile_enabled:
                    self._profiles.append(
                        {
                            "batch_size": 1,
                            "episode_keys": [str(spec["episode_key"])],
                            "wall_seconds": time.perf_counter() - started,
                            "batch_fallback": "policy_has_no_batch_api",
                        }
                    )
            return results
        if not callable(reset_batch) or not callable(act_batch):
            started = time.perf_counter()
            result = _run_episode(
                self.runtime,
                self.policy,
                rows[0],
                media_dir=self.media_dir,
                num_steps_wait=self.policy_spec.num_steps_wait,
                frames_per_second=self.policy_spec.frames_per_second,
            )
            if self.profile_enabled:
                self._profiles.append(
                    {
                        "batch_size": 1,
                        "episode_keys": [str(rows[0]["episode_key"])],
                        "wall_seconds": time.perf_counter() - started,
                        "batch_fallback": "policy_has_no_batch_api",
                    }
                )
            return [result]

        results, batch_profile = _run_batch(
            self.runtime,
            self.policy,
            rows,
            media_dir=self.media_dir,
            num_steps_wait=self.policy_spec.num_steps_wait,
            frames_per_second=self.policy_spec.frames_per_second,
            profile=self.profile_enabled,
        )
        if self.profile_enabled:
            self._profiles.append(batch_profile)
        return results

    @daft.method(return_dtype=DataType.python())
    def take_profiles(self) -> list[dict[str, Any]]:
        profiles = self._profiles
        self._profiles = []
        return profiles

    @daft.method(return_dtype=DataType.bool())
    def close(self) -> bool:
        try:
            close_policy = getattr(self.policy, "close", None)
            if callable(close_policy):
                close_policy()
        finally:
            self.policy = None
            self.runtime.close()
        return True


@dataclass(frozen=True)
class Evaluation:
    """Lazy Daft views over one completed or partially completed evaluation."""

    path: Path
    evaluation_id: str
    episodes: DataFrame
    steps: DataFrame

    def metrics(self, *group_by: str) -> DataFrame:
        """Return a lazy success/count table, optionally grouped."""
        expressions = [
            col("episode_key").count().alias("episodes"),
            col("success").cast(DataType.int64()).sum().alias("successes"),
            col("length").mean().alias("mean_steps"),
        ]
        if group_by:
            return self.episodes.groupby(*group_by).agg(*expressions)
        return self.episodes.agg(*expressions)

    def success_rate(self) -> float:
        counts = self.metrics().to_pydict()
        episodes = int(counts["episodes"][0])
        successes = int(counts["successes"][0] or 0)
        return successes / episodes if episodes else float("nan")


def _empty_frame(schema: daft.Schema) -> DataFrame:
    frame = daft.from_pydict({field.name: [None] for field in schema})
    return frame.select(*(col(field.name).cast(field.dtype) for field in schema)).where(lit(False))


def _parquet_files(root: Path) -> list[str]:
    return sorted(str(path) for path in root.rglob("*.parquet"))


def _read_table(root: Path, schema: daft.Schema) -> DataFrame:
    files = _parquet_files(root)
    if not files:
        return _empty_frame(schema)
    frame = daft.read_parquet(
        files,
        infer_schema=False,
        schema={field.name: field.dtype for field in schema},
        hive_partitioning=True,
        ignore_corrupt_files=True,
        file_path_column="_source_path",
    )
    return (
        frame.where(col("_source_path").contains("/episode_key="))
        .select(*(col(field.name).cast(field.dtype) for field in schema))
        .where(col("episode_id").not_null())
    )


def read_evaluation(path: str | Path) -> Evaluation:
    """Open episodes and steps as lazy Daft frames."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"evaluation manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation_id = str(manifest["evaluation_id"])
    return Evaluation(
        path=root,
        evaluation_id=evaluation_id,
        episodes=_read_table(root / "episodes", EPISODE_SCHEMA),
        steps=_read_table(root / "steps", STEP_SCHEMA),
    )


def _completed_episode_keys(root: Path) -> DataFrame:
    episodes = _read_table(root / "episodes", EPISODE_SCHEMA)
    steps = _read_table(root / "steps", STEP_SCHEMA)
    if not _parquet_files(root / "episodes") or not _parquet_files(root / "steps"):
        return _empty_frame(
            daft.Schema.from_field_name_and_types([("episode_key", DataType.string())])
        )
    complete_steps = (
        steps.groupby("episode_key")
        .agg(
            col("frame_index").count().alias("_rows"),
            col("frame_index").min().alias("_first"),
            col("frame_index").max().alias("_last"),
            col("frame_index").count_distinct().alias("_distinct"),
        )
        .where(
            (col("_first") == lit(0))
            & (col("_last") == col("_rows") - lit(1))
            & (col("_distinct") == col("_rows"))
        )
    )
    return (
        episodes.join(complete_steps, on="episode_key", how="inner")
        .where(col("length") == col("_rows"))
        .select("episode_key")
        .distinct()
    )


def _canonical_specs(benchmark: Benchmark) -> tuple[DataFrame, str]:
    missing = sorted(set(_SPEC_COLUMNS) - set(benchmark.specs.column_names))
    # episode_index is assigned only after a stable sort/materialization.
    if missing != ["episode_index"]:
        raise ValueError(f"benchmark specs are missing required columns: {missing!r}")
    data = benchmark.specs.select(
        *(name for name in _SPEC_COLUMNS if name != "episode_index")
    ).to_pydict()
    count = len(data["episode_key"])
    if count == 0:
        raise ValueError("benchmark selection produced no episodes")
    required = ("episode_key", "episode_id", "suite", "task_key", "init_state_id", "seed")
    for name in required:
        if any(value is None for value in data[name]):
            raise ValueError(f"benchmark specs contain null {name!r} values")
    order = sorted(
        range(count),
        key=lambda index: (
            data["suite"][index],
            data["task_key"][index],
            data["init_state_id"][index],
            data["seed"][index],
        ),
    )
    data = {name: [values[index] for index in order] for name, values in data.items()}
    if len(set(data["episode_key"])) != count:
        raise ValueError("benchmark specs contain duplicate episode keys")
    if len(set(data["episode_id"])) != count:
        raise ValueError("benchmark specs contain duplicate episode IDs")
    data["episode_index"] = list(range(count))
    ordered: dict[str, Any] = {name: data[name] for name in _SPEC_COLUMNS}
    canonical = json.dumps(
        [
            {name: ordered[name][index] for name in _SPEC_COLUMNS}
            for index in range(count)
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return daft.from_pydict(ordered), hashlib.sha256(canonical).hexdigest()


def _episode_frame(rollout: DataFrame) -> DataFrame:
    return rollout.select(
        "episode_index",
        "episode_key",
        "episode_id",
        "suite",
        "suite_variant",
        "perturbation",
        "task_id",
        "task_key",
        col("resolved_task_name").alias("task_name"),
        col("resolved_instruction").alias("instruction"),
        "bddl_path",
        "init_path",
        "init_state_id",
        "seed",
        "success",
        col("num_steps").alias("length"),
        "reward",
        "terminal_failure",
        "primary_video_path",
        "wrist_video_path",
    ).select(*(col(field.name).cast(field.dtype) for field in EPISODE_SCHEMA))


def _step_frame(rollout: DataFrame) -> DataFrame:
    return (
        rollout.select(
            "episode_index",
            "episode_key",
            "episode_id",
            "task_key",
            explode(col("steps")).alias("_step"),
        )
        .select(
            "episode_index",
            "episode_key",
            "episode_id",
            "task_key",
            unnest(col("_step")),
        )
        .select(*(col(field.name).cast(field.dtype) for field in STEP_SCHEMA))
    )


def _write_partition(frame: DataFrame, root: Path) -> None:
    """Write one episode partition; factored for failure-injection tests."""
    frame.write_parquet(
        root,
        partition_cols=["episode_key"],
        write_mode="overwrite-partitions",
        write_success_file=False,
    )


def _pending_batches(
    pending: DataFrame,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    specs = [
        {name: row[name] for name in _SPEC_COLUMNS}
        for row in pending.sort("episode_index").iter_rows()
    ]
    return [specs[offset : offset + batch_size] for offset in range(0, len(specs), batch_size)]


def _append_profiles(path: Path, profiles: Sequence[Mapping[str, Any]]) -> None:
    if not profiles:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for profile in profiles:
            stream.write(json.dumps(dict(profile), sort_keys=True, allow_nan=False))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(
    policy: PolicySpec,
    benchmark: Benchmark,
    *,
    out: str | Path,
    write_video: bool = True,
    checkpoint: Callable[[], None] | None = None,
    env_batch_size: int = 1,
    profile: bool = False,
) -> Evaluation:
    """Run pending specs and return lazy episode/step frames.

    The policy and simulator are stateful and stay in this Python process. Daft
    owns spec planning, resume anti-joins, schema casts, partition writes, reads,
    and metrics. A crash loses at most the active environment cohort.
    """
    if env_batch_size < 1:
        raise ValueError("env_batch_size must be positive")
    specs, specs_hash = _canonical_specs(benchmark)
    evaluation_id, config = evaluation_manifest(
        policy={
            "id": policy.policy_id,
            "revision": policy.revision,
            "control_mode": policy.control_mode,
            "camera_height": policy.camera_height,
            "camera_width": policy.camera_width,
            "num_steps_wait": policy.num_steps_wait,
            "frames_per_second": policy.frames_per_second,
            "metadata": dict(policy.metadata or {}),
        },
        benchmark={
            "name": benchmark.name,
            "revision": benchmark.revision,
            "metadata": dict(benchmark.metadata or {}),
            "execution": {
                "env_batch_size": env_batch_size,
                "profile": profile,
                "write_video": write_video,
            },
        },
        specs_sha256=specs_hash,
    )
    root = Path(out) / evaluation_id
    write_manifest(root, evaluation_id, config)

    completed = _completed_episode_keys(root)
    pending = specs.join(completed, on="episode_key", how="anti").collect()
    if pending.count_rows() == 0:
        return read_evaluation(root)
    media_dir = str(root / "videos") if write_video else None
    actor = RolloutActor(
        policy,
        benchmark.runtime_factory,
        media_dir=media_dir,
        profile=profile,
    )
    try:
        for batch in _pending_batches(pending, env_batch_size):
            results = actor.rollout(batch)
            for spec, result in zip(batch, results, strict=True):
                rollout = daft.from_pylist([{**spec, **result}])
                steps = _step_frame(rollout)
                episodes = _episode_frame(rollout)
                # The episode row is the completion marker. If a process dies
                # after steps land, resume overwrites the incomplete partition.
                _write_partition(steps, root / "steps")
                _write_partition(episodes, root / "episodes")
            _append_profiles(root / "profiles.jsonl", actor.take_profiles())
            if checkpoint is not None:
                checkpoint()
    finally:
        actor.close()

    return read_evaluation(root)


def canonical_signature(
    frame: DataFrame,
    *,
    sort_by: Sequence[str],
    columns: Sequence[str] | None = None,
) -> str:
    """Stable content signature for conformance tests, not Parquet bytes."""
    selected = frame.select(*(columns or frame.column_names)).sort(list(sort_by))
    data = selected.to_pydict()

    def normalize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return value.hex()
        return value

    rows = [
        {name: normalize(data[name][index]) for name in selected.column_names}
        for index in range(len(next(iter(data.values()), [])))
    ]
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
