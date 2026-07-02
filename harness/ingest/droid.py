"""DROID adapter — parses the raw per-episode ``trajectory.h5`` (the (b) fork).

CONCRETE (implemented), TensorFlow-free. Daft-native ``daft.datasets.droid.raw()``
discovers DROID episodes and hands back a ``daft.File`` to each ``trajectory.h5`` plus the
MP4 camera ``VideoFile``s and unnested episode metadata — but it does NOT parse the
numeric trajectory (actions/proprio). That parse is ours, and ``trajectory.h5`` is HDF5,
so it reuses our h5 machinery. DECIDED 2026-06-11: this (b) fork, not RLDS/tfds.

Two entry points, ONE shared parser:
  * ``parse_trajectory_h5(local_path, episode_meta)`` — the unit the Daft-compose path calls
    per row (after ``trajectory`` is downloaded to a local path); ``episode_meta`` is the
    ``droid.raw()`` row (uuid, success, current_task, trajectory_length, ...).
  * ``DroidIngestor.load(dir)`` — offline path over a directory of episode folders, reading
    ``trajectory.h5`` + sibling ``metadata_*.json``. Smoke-testable without Daft/GCS.

``trajectory.h5`` layout (per the DROID/r2d2 raw release; leaf names marked UNVERIFIED were
not confirmed against a real file — the parser PROBES the ambiguous ones)::

    action_dict/  (or top-level action/ — PROBED)
      |-- cartesian_position  (T, 6)  f64   absolute pose target (xyz + euler/axis)
      +-- gripper_position    (T, 1)  f64   commanded gripper
    observation/
      +-- robot_state/
            |-- cartesian_position  (T, 6)  f64   MEASURED pose   (note: obs leaves are
            |-- gripper_position    (T, 1)  f64   MEASURED gripper  PLURAL joint_positionS)
            +-- joint_positions     (T, 7)  f64   (optional)

Conventions: action = concat(cartesian_position(6), gripper_position(1)) -> 7-D, matching the
official RLDS builder. ``control_mode='absolute'`` (DROID stores absolute pose targets, NOT
deltas — do not mislabel relative). No RGB lives in the h5 (frames come from the MP4s via
``video_path``); no per-step reward.

Parked alternative (a) — RLDS/TFDS curated DROID (needs TensorFlow), intentionally NOT taken::

    import tensorflow_datasets as tfds
    ds = tfds.load('droid', data_dir=path, split='train')   # gs://gresearch/robotics, or droid_100
    for ep in ds:                                            # ep['steps'] nested tf.data
        ...                                                   # action/observation/language_instruction
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator

import numpy as np

from harness.ingest.base import Episode, Ingestor, Step


def parse_trajectory_h5(
    local_path: str,
    episode_meta: dict,
    *,
    state_includes_joints: bool = False,
) -> Episode:
    """Parse one raw-DROID ``trajectory.h5`` into a normalized Episode.

    ``episode_meta`` carries episode-level fields (a ``droid.raw()`` row or a parsed
    ``metadata_*.json``): ``uuid``, ``success``, ``current_task``, ``trajectory_length``.
    ``state_includes_joints`` appends ``joint_positions`` to the proprio state (7 -> 14 dim).
    """
    import h5py  # lazy

    f = h5py.File(local_path, "r")
    try:
        # --- commanded action: action_dict (preferred) or top-level action (PROBED) ---
        action_grp = f["action_dict"] if "action_dict" in f else f["action"]
        cart = np.asarray(action_grp["cartesian_position"][:], dtype=np.float32)   # (T,6)
        grip = np.asarray(action_grp["gripper_position"][:], dtype=np.float32).reshape(-1, 1)
        n = len(cart)
        action7 = np.concatenate([cart[:, :6], grip[:n, :1]], axis=1).astype(np.float32)  # (T,7)

        # --- measured proprio: observation/robot_state (obs leaves are PLURAL) ---
        robot_state = f["observation"]["robot_state"]
        meas_cart = np.asarray(robot_state["cartesian_position"][:], dtype=np.float32)      # (T,6)
        meas_grip = np.asarray(robot_state["gripper_position"][:], dtype=np.float32).reshape(n, -1)
        joints = (
            np.asarray(robot_state["joint_positions"][:], dtype=np.float32)
            if "joint_positions" in robot_state else None
        )

        steps = []
        for i in range(n):
            parts = [meas_cart[i, :6], meas_grip[i, :1]]
            if state_includes_joints and joints is not None:
                parts.append(joints[i])
            steps.append(
                Step(
                    timestep=i,
                    state=np.concatenate(parts).astype(np.float32),
                    action=action7[i],
                    reward=None,  # DROID has no per-step reward
                    done=(i == n - 1),
                    is_terminal=(i == n - 1),
                    eef_pos=meas_cart[i, :3],
                    gripper_state=float(meas_grip[i, 0]),  # measured, != commanded action[-1]
                )
            )

        uuid = episode_meta.get("uuid", "unknown")
        return Episode(
            episode_id=f"droid/{uuid}",
            source="droid",
            instruction=(episode_meta.get("current_task") or "").strip(),
            steps=tuple(steps),
            success=bool(episode_meta.get("success", False)),
            terminal_failure=None,
            model="droid",          # human teleop, no policy
            policy_type="droid",
            suite=None,
            task_id=None,
            task_name=None,
            metadata={"control_mode": "absolute"},  # DROID stores absolute pose targets
        )
    finally:
        f.close()


def _local_path(uri: str) -> str:
    """Daft's glob returns ``file://`` URIs for local paths; h5py needs a plain path.

    Returns the local filesystem path for a ``file://`` (or scheme-less) URI; leaves a remote
    URI (``gs://``, ``s3://``) untouched — those must be downloaded before h5py can open them.
    """
    if uri.startswith("file://"):
        from urllib.parse import unquote, urlparse
        return unquote(urlparse(uri).path)
    return uri


def _read_droid_meta(episode_dir: str) -> dict:
    """Read the sibling ``metadata_*.json`` for an episode dir (offline path)."""
    metas = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
    if metas:
        with open(metas[0]) as fh:
            return json.load(fh)
    return {}


class DroidIngestor(Ingestor):
    """Adapter for raw DROID: discovery + metadata via the native ``daft.datasets.droid.raw()``
    (Daft >=0.7.16), numeric trajectory
    parsed by ``parse_trajectory_h5`` (the (b) fork). Daft does the episode discovery and
    metadata parse; we parse the per-episode ``trajectory.h5`` it points at.
    """

    source = "droid"

    def load(
        self,
        path: str,
        *,
        limit: int | None = None,
        state_includes_joints: bool = False,
        use_daft: bool = True,
    ) -> Iterator[Episode]:
        """Yield normalized Episodes from a raw-DROID root.

        ``path``: a directory of episode folders (local) or a ``gs://`` root. With ``use_daft``
        (default) episodes are discovered via ``daft.datasets.droid.raw()`` (metadata from each
        ``metadata_*.json``); ``use_daft=False`` falls back to a plain ``trajectory.h5`` glob
        with no episode metadata. Each episode's ``trajectory.h5`` is parsed by
        ``parse_trajectory_h5``.
        """
        gen = self._load_via_daft if use_daft else self._load_via_glob
        yield from gen(path, limit=limit, state_includes_joints=state_includes_joints)

    def _load_via_daft(self, path, *, limit, state_includes_joints):
        import daft.datasets

        cols = ["uuid", "success", "current_task", "trajectory_length", "episode_dir"]
        rows = daft.datasets.droid.raw(path).select(*cols).to_pylist()
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            # raw() points at each episode dir; the trajectory.h5 lives inside it. Local paths
            # parse in place; a gs:// root would need the `trajectory` File downloaded first.
            h5_path = os.path.join(_local_path(row["episode_dir"]), "trajectory.h5")
            yield parse_trajectory_h5(h5_path, row, state_includes_joints=state_includes_joints)

    def _load_via_glob(self, path, *, limit, state_includes_joints):
        h5_paths = sorted(glob.glob(os.path.join(path, "**", "trajectory.h5"), recursive=True))
        if limit is not None:
            h5_paths = h5_paths[:limit]
        for h5_path in h5_paths:
            meta = _read_droid_meta(os.path.dirname(h5_path))
            yield parse_trajectory_h5(h5_path, meta, state_includes_joints=state_includes_joints)
