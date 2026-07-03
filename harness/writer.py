"""Rollouts/Episodes -> parquet writer.

Emits EXACTLY ``harness.schema.ROLLOUT_SCHEMA`` (one row per step), as a directory of
part files so ``daft.read_parquet('data/rollouts/*.parquet')`` consumes it cleanly.
``write_rows`` / ``write_episode`` are the real write primitives; ``RolloutWriter`` is the
streaming capture loop the LIBERO runner drives per step.

We write with ``pyarrow.parquet`` (not Daft) so the on-disk fixture stays portable and
Daft-only extension types never leak into the file (see schema.py rationale).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from harness.ingest.base import Episode, Step
from harness.schema import ROLLOUT_SCHEMA, validate_rows


def write_rows(rows: list[dict], out_path: str | Path, *, compression: str = "snappy") -> Path:
    """Validate step-row dicts against the schema and write ONE parquet part file.

    This is the single real write primitive. ``rows`` must be dicts with the schema's
    columns (build them via ``Episode.to_step_rows`` or ``schema.empty_step_row``).
    Returns the written path. Raises on schema violation (cheap correctness gate).
    """
    table: pa.Table = validate_rows(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression=compression)
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
    """Write one normalized ``Episode`` to ``{out_dir}/{episode_id-slug}.parquet``.

    One part file per episode keeps the writer crash-safe (a failed episode never
    corrupts others) and lets Daft glob the directory. ``episode_id`` is slugified
    (``/`` -> ``__``) into the filename.
    """
    rows = episode.to_step_rows(
        run_id=run_id,
        video_path=video_path,
        frame_path_for=frame_path_for,
        wrist_path_for=wrist_path_for,
    )
    slug = episode.episode_id.replace("/", "__")
    return write_rows(rows, Path(out_dir) / f"{slug}.parquet", compression=compression)


class RolloutWriter:
    """Streaming capture writer used by the LIBERO rollout loop.

    The runner drives it per episode::

        writer.begin_episode(episode_id, suite=..., instruction=..., model=..., policy_type=...)
        for t in loop:
            writer.append_step(t, action=..., reward=..., done=..., eef_pos=..., gripper_state=...,
                               primary_frame=img, wrist_frame=wrist)
        path = writer.end_episode(success, terminal_failure=...)   # -> one parquet part

    Per-step frames are written to ``frames_dir`` as PNGs and only their PATHS land in the
    parquet (never inline bytes); ``end_episode`` optionally writes an mp4 and flushes one
    parquet part built through ``Episode.to_step_rows`` — the SAME path the ingestors use, so
    rollout and ingest emit identical columns. The writer is reused across episodes (each
    ``begin_episode`` resets the buffer).
    """

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
        """Start buffering a new episode. Stores the episode-level metadata for the rows."""
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
        """Buffer one step; write its frame PNG(s) to ``frames_dir`` if ``write_frames``.

        ``primary_frame``/``wrist_frame`` are uint8 HWC arrays (already de-rotated by the
        runner). Only their on-disk paths are stored on the row; the arrays feed the mp4.
        """
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
        """Finalize: optionally write the mp4, build rows via ``Episode.to_step_rows``, and
        write one parquet part. Returns the parquet path and clears the buffer."""
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

    # --- helpers ---

    def _slug(self) -> str:
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
    """Read a written part file back and assert its schema matches ``ROLLOUT_SCHEMA``.

    Compares via pyarrow's ``Schema.equals(check_metadata=False)``: parquet round-trips list
    columns with the inner field renamed ``item`` -> ``element`` (semantically identical), which
    breaks naive stringified-type comparison. Schema-level metadata is ignored.
    """
    got = pq.read_schema(Path(path))
    if not got.equals(ROLLOUT_SCHEMA, check_metadata=False):
        got_fields = [(f.name, str(f.type)) for f in got]
        want_fields = [(f.name, str(f.type)) for f in ROLLOUT_SCHEMA]
        raise AssertionError(
            f"schema mismatch for {path}\n  got : {got_fields}\n  want: {want_fields}"
        )
