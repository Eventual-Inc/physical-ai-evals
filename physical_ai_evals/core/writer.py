from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from physical_ai_evals.core.episode import Episode, Step
from physical_ai_evals.core.schema import ROLLOUT_SCHEMA, validate_rows


def write_rows(rows: list[dict], out_path: str | Path, *, compression: str = "snappy") -> Path:
    """Validate rows and atomically replace one Parquet part file."""
    table: pa.Table = validate_rows(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=out_path.parent,
        prefix=f".{out_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path, compression=compression)
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, out_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return out_path


def write_episode(
    episode: Episode,
    out_dir: str | Path,
    *,
    run_id: str = "ingest",
    video_path: str | None = None,
    frame_path_for=None,
    wrist_path_for=None,
    compression: str = "snappy",
) -> Path:
    """Write one episode parquet to ``{out_dir}/{episode_id-slug}.parquet``."""
    rows = episode.to_step_rows(
        run_id=run_id,
        video_path=video_path,
        frame_path_for=frame_path_for,
        wrist_path_for=wrist_path_for,
    )
    slug = episode.episode_id.replace("/", "__")
    return write_rows(rows, Path(out_dir) / f"{slug}.parquet", compression=compression)


class RolloutWriter:
    """Streaming writer: begin_episode -> append_step -> end_episode (one parquet part)."""

    def __init__(
        self,
        out_dir: str | Path,
        frames_dir: str | Path | None = None,
        videos_dir: str | Path | None = None,
        run_id: str = "rollout",
        *,
        write_frames: bool = True,
        write_video: bool = True,
        compression: str = "snappy",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.frames_dir = Path(frames_dir) if frames_dir else self.out_dir.parent / "frames"
        self.videos_dir = Path(videos_dir) if videos_dir else self.out_dir.parent / "videos"
        self.run_id = run_id
        self.write_frames = write_frames
        self.write_video = write_video
        self.compression = compression
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.last_video_path: str | None = None
        self._reset()

    def _reset(self) -> None:
        self._meta: dict | None = None
        self._steps: list[Step] = []
        self._frame_paths: list[tuple[str | None, str | None]] = []
        self._video_frames: list[np.ndarray] = []

    def begin_episode(
        self,
        episode_id: str,
        *,
        suite: str | None = None,
        task_id: int | None = None,
        task_name: str | None = None,
        instruction: str = "",
        model: str = "",
        policy_type: str = "",
        init_state_id: int | None = None,
        seed: int | None = None,
        bddl_file: str | None = None,
        control_mode: str = "relative",
    ) -> None:
        self._reset()
        self._meta = dict(
            episode_id=episode_id, suite=suite, task_id=task_id, task_name=task_name,
            instruction=instruction, model=model, policy_type=policy_type,
            init_state_id=init_state_id, seed=seed, bddl_file=bddl_file, control_mode=control_mode,
        )

    def append_step(
        self,
        step_idx: int,
        *,
        action=None,
        reward: float | None = None,
        done: bool = False,
        state=None,
        eef_pos=None,
        gripper_state: float | None = None,
        object_poses: dict | None = None,
        primary_frame=None,
        wrist_frame=None,
    ) -> None:
        slug = self._slug()
        fp = self._write_png(primary_frame, slug, step_idx, "primary") if (self.write_frames and primary_frame is not None) else None
        wp = self._write_png(wrist_frame, slug, step_idx, "wrist") if (self.write_frames and wrist_frame is not None) else None
        self._frame_paths.append((fp, wp))
        if self.write_video and primary_frame is not None:
            self._video_frames.append(np.asarray(primary_frame, dtype=np.uint8))
        self._steps.append(
            Step(
                timestep=step_idx,
                action=None if action is None else np.asarray(action, np.float32),
                reward=None if reward is None else float(reward),
                done=bool(done),
                is_terminal=bool(done),
                state=None if state is None else np.asarray(state, np.float32),
                eef_pos=None if eef_pos is None else np.asarray(eef_pos, np.float32),
                gripper_state=None if gripper_state is None else float(gripper_state),
                object_poses=object_poses or {},
            )
        )

    def end_episode(self, success: bool, terminal_failure: str | None = None) -> Path:
        m = self._meta
        if m is None:
            raise RuntimeError("end_episode called before begin_episode")
        video_path = self._write_video(success) if (self.write_video and self._video_frames) else None
        episode = Episode(
            episode_id=m["episode_id"], source="libero", instruction=m["instruction"],
            steps=tuple(self._steps), success=bool(success), terminal_failure=terminal_failure,
            model=m["model"], policy_type=m["policy_type"], suite=m["suite"],
            task_id=m["task_id"], task_name=m["task_name"],
            metadata={
                "control_mode": m["control_mode"], "bddl_file": m["bddl_file"],
                "init_state_id": m["init_state_id"], "seed": m["seed"],
            },
        )
        fmap = {i: self._frame_paths[i][0] for i in range(len(self._frame_paths))}
        wmap = {i: self._frame_paths[i][1] for i in range(len(self._frame_paths))}
        rows = episode.to_step_rows(
            run_id=self.run_id,
            video_path=video_path,
            frame_path_for=lambda _eid, i: fmap.get(i),
            wrist_path_for=lambda _eid, i: wmap.get(i),
        )
        out = write_rows(rows, self.out_dir / f"{self._slug()}.parquet", compression=self.compression)
        self._reset()
        return out

    def _slug(self) -> str:
        if self._meta is None:
            raise RuntimeError("end_episode called before begin_episode")
        return self._meta["episode_id"].replace("/", "__")

    def _write_png(self, frame, slug: str, step_idx: int, role: str) -> str:
        import imageio.v3 as iio
        d = self.frames_dir / slug
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{step_idx:04d}_{role}.png"
        iio.imwrite(path, np.asarray(frame, dtype=np.uint8))
        return str(path)

    def _write_video(self, success: bool) -> str:
        import imageio.v3 as iio
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        path = self.videos_dir / f"{self._slug()}__{'success' if success else 'fail'}.mp4"
        iio.imwrite(path, np.stack(self._video_frames), fps=20, codec="libx264")
        self.last_video_path = str(path)
        return str(path)


def assert_emits_schema(path: str | Path) -> None:
    """Assert a parquet part matches ``ROLLOUT_SCHEMA``."""
    got = pq.read_schema(Path(path))
    if not got.equals(ROLLOUT_SCHEMA, check_metadata=False):
        got_fields = [(f.name, str(f.type)) for f in got]
        want_fields = [(f.name, str(f.type)) for f in ROLLOUT_SCHEMA]
        raise AssertionError(
            f"schema mismatch for {path}\n  got : {got_fields}\n  want: {want_fields}"
        )
