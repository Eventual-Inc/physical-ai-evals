"""Daft-native batch evaluation: specs -> rollout -> episodes + steps."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

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
    data = {
        name: [values[index] for index in order]
        for name, values in data.items()
    }
    if len(set(data["episode_key"])) != count:
        raise ValueError("benchmark specs contain duplicate episode keys")
    if len(set(data["episode_id"])) != count:
        raise ValueError("benchmark specs contain duplicate episode IDs")
    data["episode_index"] = list(range(count))
    ordered: dict[str, Any] = {name: data[name] for name in _SPEC_COLUMNS}
    canonical = json.dumps(
        [
            {
                key: ordered[key][index]
                for key in (
                    "episode_key",
                    "episode_id",
                    "benchmark",
                    "benchmark_revision",
                    "suite",
                    "task_key",
                    "init_state_id",
                    "seed",
                    "max_steps",
                )
            }
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


def evaluate(
    policy: PolicySpec,
    benchmark: Benchmark,
    *,
    out: str | Path,
    write_video: bool = True,
    checkpoint: Callable[[], None] | None = None,
) -> Evaluation:
    """Run pending specs and return lazy episode/step frames.

    The policy and simulator are stateful and stay in this Python process. Daft
    owns spec planning, resume anti-joins, schema casts, partition writes, reads,
    and metrics. A crash loses at most the active episode.
    """
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
    runtime = partial(
        benchmark.runtime_factory,
        camera_height=policy.camera_height,
        camera_width=policy.camera_width,
    )()
    policy_instance: Policy | None = None
    try:
        policy_instance = policy.factory()
        if int(policy_instance.action_dim) != ACTION_DIM:
            raise ValueError(
                f"policy action_dim must be {ACTION_DIM}, got {policy_instance.action_dim}"
            )
        if str(policy_instance.control_mode) != policy.control_mode:
            raise ValueError(
                "policy factory control_mode does not match PolicySpec: "
                f"{policy_instance.control_mode!r} != {policy.control_mode!r}"
            )
        for row in pending.sort("episode_key").iter_rows():
            spec = {name: row[name] for name in _SPEC_COLUMNS}
            result = _run_episode(
                runtime,
                policy_instance,
                spec,
                media_dir=media_dir,
                num_steps_wait=policy.num_steps_wait,
                frames_per_second=policy.frames_per_second,
            )
            rollout = daft.from_pylist([{**spec, **result}])
            steps = _step_frame(rollout)
            episodes = _episode_frame(rollout)
            # The episode row is the completion marker. If a process dies after
            # steps land, resume sees no valid completion and overwrites them.
            _write_partition(steps, root / "steps")
            _write_partition(episodes, root / "episodes")
            if checkpoint is not None:
                checkpoint()
    finally:
        try:
            close_policy = getattr(policy_instance, "close", None)
            if callable(close_policy):
                close_policy()
        finally:
            runtime.close()

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
