"""Daft-native batch evaluation: planned rollouts -> episodes + steps."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypedDict

import daft
import numpy as np
from daft import DataFrame, DataType, col, lit
from daft.functions import explode, unnest

from physical_ai_evals.policy import Observation, Policy, PolicySpec
from physical_ai_evals.provenance import evaluation_manifest, write_manifest
from physical_ai_evals.schema import (
    ACTION_DIM,
    EEF_POS_DIM,
    EPISODE_SCHEMA,
    STATE_DIM,
    STEP_SCHEMA,
)


class RuntimeObservation(TypedDict):
    """Benchmark observation normalized for policies and evaluation traces."""

    image: np.ndarray
    wrist_image: np.ndarray | None
    state: np.ndarray | None
    eef_position: np.ndarray | None
    gripper: float | None


class Runtime(Protocol):
    """Stateful benchmark runtime owned by one rollout actor."""

    def open(self, rollout: Mapping[str, Any]) -> tuple[Any, str, Any, str | None]: ...

    def open_batch(
        self,
        rollouts: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, list[str], list[Any], list[str | None]]: ...

    def normalize_observation(
        self,
        observation: Mapping[str, Any],
    ) -> RuntimeObservation: ...

    def normalize_observations(
        self,
        observations: Any,
        count: int,
    ) -> list[RuntimeObservation]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Benchmark:
    """A lazy rollout table plus the runtime factory that executes it."""

    name: str
    revision: str
    rollouts: DataFrame
    runtime_factory: Callable[..., Runtime]
    metadata: Mapping[str, Any] | None = None


_ROLLOUT_COLUMNS = (
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
    rollout: Mapping[str, Any],
    *,
    media_dir: str | None,
    num_steps_wait: int,
    frames_per_second: int,
) -> dict[str, Any]:
    seed = int(rollout["seed"])
    _seed_process(seed)
    environment, instruction, init_state, task_name = runtime.open(rollout)
    seed_environment = getattr(environment, "seed", None)
    if callable(seed_environment):
        seed_environment(seed)

    environment.reset()
    observation = environment.set_init_state(init_state)
    dummy_action = [0.0] * int(getattr(policy, "action_dim", ACTION_DIM))
    for _ in range(num_steps_wait):
        observation = environment.step(dummy_action)[0]

    policy.reset(instruction)
    normalized = runtime.normalize_observation(observation)
    videos = (
        _EpisodeVideos(media_dir, str(rollout["episode_key"]), frames_per_second)
        if media_dir is not None
        else None
    )
    steps: list[dict[str, Any]] = []
    final_reward = 0.0
    try:
        for frame_index in range(int(rollout["max_steps"])):
            policy_observation = _policy_observation(normalized, instruction)
            action = policy.act(policy_observation)
            action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
            if action.shape != (ACTION_DIM,):
                raise ValueError(
                    f"policy action must have shape {(ACTION_DIM,)}, got {action.shape}"
                )
            if videos is not None:
                videos.append("primary", normalized["image"])
                videos.append("wrist", normalized["wrist_image"])

            next_observation, final_reward, done, _info = environment.step(action)
            next_normalized = runtime.normalize_observation(next_observation)
            steps.append(
                {
                    "frame_index": frame_index,
                    "timestamp": float(frame_index / frames_per_second),
                    "action": action,
                    "reward": float(final_reward),
                    "next.done": bool(done),
                    "observation.state": normalized["state"],
                    "observation.eef_position": next_normalized["eef_position"],
                    "observation.gripper": next_normalized["gripper"],
                }
            )
            normalized = next_normalized
            if done:
                break

        success = bool(environment.check_success())
        primary_path, wrist_path = videos.finish() if videos is not None else (None, None)
        return {
            "resolved_task_name": task_name or rollout.get("task_name"),
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


def _policy_observation(
    observation: RuntimeObservation,
    instruction: str,
) -> Observation:
    return {
        "image": observation["image"],
        "wrist_image": observation["wrist_image"],
        "state": observation["state"],
        "instruction": instruction,
    }


def _run_batch(
    runtime: Runtime,
    policy: Policy,
    rollouts: Sequence[Mapping[str, Any]],
    *,
    media_dir: str | None,
    num_steps_wait: int,
    frames_per_second: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one fixed policy batch over LIBERO's CPU subprocess environments."""
    if not rollouts:
        return [], {}
    reset_batch = getattr(policy, "reset_batch", None)
    act_batch = getattr(policy, "act_batch", None)
    if not callable(reset_batch) or not callable(act_batch):
        raise TypeError("policy does not implement reset_batch() and act_batch()")

    seeds = {int(rollout["seed"]) for rollout in rollouts}
    if len(seeds) != 1:
        raise ValueError("one rollout batch must use one process seed")
    _seed_process(next(iter(seeds)))

    started = time.perf_counter()
    reset_seconds = 0.0
    settle_seconds = 0.0
    policy_seconds = 0.0
    environment_seconds = 0.0
    policy_calls = 0
    transitions = 0

    environment, instructions, init_states, task_names = runtime.open_batch(rollouts)
    reset_started = time.perf_counter()
    environment.reset()
    normalized = runtime.normalize_observations(
        environment.set_init_state(init_states),
        len(rollouts),
    )
    reset_seconds += time.perf_counter() - reset_started

    dummy = np.zeros(
        (len(rollouts), int(getattr(policy, "action_dim", ACTION_DIM))),
        np.float32,
    )
    settle_started = time.perf_counter()
    for _ in range(num_steps_wait):
        stepped = environment.step(dummy)
        normalized = runtime.normalize_observations(stepped[0], len(rollouts))
    settle_seconds += time.perf_counter() - settle_started

    reset_batch(instructions)
    videos = [
        (
            _EpisodeVideos(media_dir, str(rollout["episode_key"]), frames_per_second)
            if media_dir is not None
            else None
        )
        for rollout in rollouts
    ]
    steps: list[list[dict[str, Any]]] = [[] for _ in rollouts]
    final_rewards = [0.0 for _ in rollouts]
    finished = [False for _ in rollouts]
    try:
        max_frames = max(int(rollout["max_steps"]) for rollout in rollouts)
        for frame_index in range(max_frames):
            active = [
                index
                for index, rollout in enumerate(rollouts)
                if not finished[index] and frame_index < int(rollout["max_steps"])
            ]
            if not active:
                break

            policy_observations = [
                _policy_observation(observation, instruction)
                for observation, instruction in zip(normalized, instructions, strict=True)
            ]
            policy_started = time.perf_counter()
            actions = np.asarray(act_batch(policy_observations), dtype=np.float32)
            policy_seconds += time.perf_counter() - policy_started
            policy_calls += 1
            if actions.shape != (len(rollouts), ACTION_DIM):
                raise ValueError(
                    "policy action batch must have shape "
                    f"{(len(rollouts), ACTION_DIM)}, got {actions.shape}"
                )
            actions = np.clip(actions, -1.0, 1.0)

            for index in active:
                video = videos[index]
                if video is not None:
                    video.append("primary", policy_observations[index]["image"])
                    video.append("wrist", policy_observations[index]["wrist_image"])

            environment_started = time.perf_counter()
            next_batch, rewards, dones, _infos = environment.step(
                actions[active],
                id=active,
            )
            environment_seconds += time.perf_counter() - environment_started
            transitions += len(active)
            next_normalized = runtime.normalize_observations(next_batch, len(active))
            for offset, index in enumerate(active):
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
                        "observation.eef_position": next_normalized[offset]["eef_position"],
                        "observation.gripper": next_normalized[offset]["gripper"],
                    }
                )
                normalized[index] = next_normalized[offset]
                final_rewards[index] = reward
                finished[index] = done

        environment_started = time.perf_counter()
        successes = [bool(value) for value in environment.check_success()]
        environment_seconds += time.perf_counter() - environment_started
        results: list[dict[str, Any]] = []
        for index, rollout in enumerate(rollouts):
            video = videos[index]
            primary_path, wrist_path = video.finish() if video is not None else (None, None)
            results.append(
                {
                    "resolved_task_name": task_names[index] or rollout.get("task_name"),
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

    elapsed = time.perf_counter() - started
    return results, {
        "batch_size": len(rollouts),
        "episode_keys": [str(rollout["episode_key"]) for rollout in rollouts],
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
        self._timings: list[dict[str, Any]] = []

    @daft.method.batch(batch_size=8, return_dtype=_ROLLOUT_DTYPE)
    def rollout(
        self,
        rollouts: daft.Series | Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = rollouts.to_pylist() if isinstance(rollouts, daft.Series) else list(rollouts)
        if self.policy is None:
            raise RuntimeError("rollout actor is closed")
        reset_batch = getattr(self.policy, "reset_batch", None)
        act_batch = getattr(self.policy, "act_batch", None)
        if len(rows) > 1 and (not callable(reset_batch) or not callable(act_batch)):
            results = []
            for rollout in rows:
                started = time.perf_counter()
                result = _run_episode(
                    self.runtime,
                    self.policy,
                    rollout,
                    media_dir=self.media_dir,
                    num_steps_wait=self.policy_spec.num_steps_wait,
                    frames_per_second=self.policy_spec.frames_per_second,
                )
                results.append(result)
                elapsed = time.perf_counter() - started
                transitions = int(result["num_steps"])
                self._timings.append(
                    {
                        "batch_size": 1,
                        "episode_keys": [str(rollout["episode_key"])],
                        "wall_seconds": elapsed,
                        "transitions": transitions,
                        "transitions_per_second": (
                            transitions / elapsed if elapsed else None
                        ),
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
            elapsed = time.perf_counter() - started
            transitions = int(result["num_steps"])
            self._timings.append(
                {
                    "batch_size": 1,
                    "episode_keys": [str(rows[0]["episode_key"])],
                    "wall_seconds": elapsed,
                    "transitions": transitions,
                    "transitions_per_second": transitions / elapsed if elapsed else None,
                    "batch_fallback": "policy_has_no_batch_api",
                }
            )
            return [result]

        results, batch_timing = _run_batch(
            self.runtime,
            self.policy,
            rows,
            media_dir=self.media_dir,
            num_steps_wait=self.policy_spec.num_steps_wait,
            frames_per_second=self.policy_spec.frames_per_second,
        )
        self._timings.append(batch_timing)
        return results

    @daft.method(return_dtype=DataType.python())
    def take_timings(self) -> list[dict[str, Any]]:
        timings = self._timings
        self._timings = []
        return timings

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


def _canonical_rollouts(benchmark: Benchmark) -> tuple[DataFrame, str]:
    missing = sorted(set(_ROLLOUT_COLUMNS) - set(benchmark.rollouts.column_names))
    # episode_index is assigned only after a stable sort/materialization.
    if missing != ["episode_index"]:
        raise ValueError(f"benchmark rollouts are missing required columns: {missing!r}")
    data = benchmark.rollouts.select(
        *(name for name in _ROLLOUT_COLUMNS if name != "episode_index")
    ).to_pydict()
    count = len(data["episode_key"])
    if count == 0:
        raise ValueError("benchmark selection produced no episodes")
    required = ("episode_key", "episode_id", "suite", "task_key", "init_state_id", "seed")
    for name in required:
        if any(value is None for value in data[name]):
            raise ValueError(f"benchmark rollouts contain null {name!r} values")
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
        raise ValueError("benchmark rollouts contain duplicate episode keys")
    if len(set(data["episode_id"])) != count:
        raise ValueError("benchmark rollouts contain duplicate episode IDs")
    data["episode_index"] = list(range(count))
    ordered: dict[str, Any] = {name: data[name] for name in _ROLLOUT_COLUMNS}
    canonical = json.dumps(
        [
            {name: ordered[name][index] for name in _ROLLOUT_COLUMNS}
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
    rollouts = [
        {name: row[name] for name in _ROLLOUT_COLUMNS}
        for row in pending.sort("episode_index").iter_rows()
    ]
    return [
        rollouts[offset : offset + batch_size]
        for offset in range(0, len(rollouts), batch_size)
    ]


def _append_timings(path: Path, timings: Sequence[Mapping[str, Any]]) -> None:
    if not timings:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for timing in timings:
            stream.write(json.dumps(dict(timing), sort_keys=True, allow_nan=False))
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
) -> Evaluation:
    """Run pending rollouts and return lazy episode/step frames.

    The policy and simulator are stateful and stay in this Python process. Daft
    owns rollout planning, resume anti-joins, schema casts, partition writes,
    reads, and metrics. A crash loses at most the active environment cohort.
    """
    if env_batch_size < 1:
        raise ValueError("env_batch_size must be positive")
    rollouts, rollouts_hash = _canonical_rollouts(benchmark)
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
                "write_video": write_video,
            },
        },
        rollouts_sha256=rollouts_hash,
    )
    root = Path(out) / evaluation_id
    write_manifest(root, evaluation_id, config)

    completed = _completed_episode_keys(root)
    pending = rollouts.join(completed, on="episode_key", how="anti").collect()
    if pending.count_rows() == 0:
        return read_evaluation(root)
    media_dir = str(root / "videos") if write_video else None
    actor = RolloutActor(
        policy,
        benchmark.runtime_factory,
        media_dir=media_dir,
    )
    try:
        for batch in _pending_batches(pending, env_batch_size):
            results = actor.rollout(batch)
            for planned_rollout, result in zip(batch, results, strict=True):
                completed_rollout = daft.from_pylist([{**planned_rollout, **result}])
                steps = _step_frame(completed_rollout)
                episodes = _episode_frame(completed_rollout)
                # The episode row is the completion marker. If a process dies
                # after steps land, resume overwrites the incomplete partition.
                _write_partition(steps, root / "steps")
                _write_partition(episodes, root / "episodes")
            _append_timings(root / "timings.jsonl", actor.take_timings())
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
