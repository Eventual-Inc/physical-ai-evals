"""EgoDex HDF5 + MP4 adapter.

EgoDex is egocentric human-manipulation data, not robot teleoperation data. The HDF5
side carries ARKit SE(3) transforms for camera/body/hand joints; the MP4 with the same
stem carries RGB. We expose selected hand transform translations as ``state`` and leave
robot-specific fields (EEF position, gripper state, reward) null.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from harness.ingest.base import Episode, Ingestor, Step

DEFAULT_TRANSFORMS = ("leftHand", "rightHand")


def _attr_str(attrs, name: str) -> str | None:
    if name not in attrs:
        return None
    value = attrs[name]
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return _attr_str({name: value.reshape(-1)[0]}, name)
    return str(value)


def _instruction(attrs) -> str:
    llm_type = _attr_str(attrs, "llm_type")
    if llm_type == "reversible":
        which = _attr_str(attrs, "which_llm_description")
        key = "llm_description" if which == "1" else "llm_description2"
        return _attr_str(attrs, key) or _attr_str(attrs, "llm_description") or ""
    return _attr_str(attrs, "llm_description") or _attr_str(attrs, "instruction") or ""


def _episode_paths(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    return sorted(glob.glob(str(p / "**" / "*.hdf5"), recursive=True))


def parse_egodex_hdf5(
    local_path: str,
    *,
    task_name: str | None = None,
    transforms: tuple[str, ...] = DEFAULT_TRANSFORMS,
) -> Episode:
    """Parse one EgoDex annotation HDF5 into an Episode."""
    import h5py  # lazy

    f = h5py.File(local_path, "r")
    try:
        tf_group = f["transforms"]
        available = [name for name in transforms if name in tf_group]
        if not available:
            available = ["camera"]

        arrays = [np.asarray(tf_group[name][:], dtype=np.float32) for name in available]
        n = min(len(arr) for arr in arrays)
        positions = np.concatenate([arr[:n, :3, 3] for arr in arrays], axis=1).astype(np.float32)

        video_path = str(Path(local_path).with_suffix(".mp4"))
        metadata = {
            "control_mode": "human_pose",
            "egodex_transforms": available,
        }
        if Path(video_path).exists():
            metadata["video_path"] = video_path

        task = task_name or Path(local_path).parent.name
        ep_stem = Path(local_path).stem
        steps = tuple(
            Step(
                timestep=i,
                state=positions[i],
                action=None,
                reward=None,
                done=(i == n - 1),
                is_terminal=(i == n - 1),
            )
            for i in range(n)
        )
        return Episode(
            episode_id=f"egodex/{task}/{ep_stem}",
            source="egodex",
            instruction=_instruction(f.attrs),
            steps=steps,
            success=False,
            terminal_failure="unlabeled",
            model="egodex",
            policy_type="egodex",
            task_name=task,
            metadata=metadata,
        )
    finally:
        f.close()


class EgoDexIngestor(Ingestor):
    """Adapter for EgoDex task folders containing paired ``.hdf5`` and ``.mp4`` files."""

    source = "egodex"

    def load(self, path: str, *, limit: int | None = None) -> Iterator[Episode]:
        paths = _episode_paths(path)
        if limit is not None:
            paths = paths[:limit]
        root = Path(path) if Path(path).is_dir() else Path(path).parent
        for h5_path in paths:
            rel_parent = Path(os.path.relpath(Path(h5_path).parent, root))
            task_name = None if str(rel_parent) == "." else rel_parent.as_posix()
            yield parse_egodex_hdf5(h5_path, task_name=task_name)
