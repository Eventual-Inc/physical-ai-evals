"""ALOHA / Mobile ALOHA HDF5 adapter.

ALOHA episodes are ACT-style HDF5 files, typically one file per episode::

    observations/
      images/{cam_high, cam_low, cam_left_wrist, cam_right_wrist}
      qpos          (T, 7|14) float64
      qvel          (T, 7|14) float64 optional
    action          (T, 7|14) float64
    base_action     (T, 2)    float64 optional, Mobile ALOHA only

This adapter keeps the robot-native joint action space instead of pretending it is a
7-DoF EEF delta action. The canonical schema stores variable-length lists, so bimanual
14-D and mobile 16-D actions both fit without schema changes.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterator

import numpy as np

from harness.ingest.base import Episode, Ingestor, Step


def _attr_text(attrs, *names: str) -> str | None:
    for name in names:
        if name not in attrs:
            continue
        value = attrs[name]
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.ndarray):
            return _attr_text({name: value.reshape(-1)[0]}, name)
        return str(value)
    return None


def _episode_paths(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    patterns = ("episode_*.hdf5", "episode_*.h5", "*.hdf5", "*.h5")
    seen: set[str] = set()
    out: list[str] = []
    for pattern in patterns:
        for item in sorted(glob.glob(str(p / "**" / pattern), recursive=True)):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def parse_aloha_hdf5(local_path: str, *, task_name: str | None = None) -> Episode:
    """Parse one ALOHA HDF5 episode into the common Episode representation."""
    import h5py  # lazy

    f = h5py.File(local_path, "r")
    try:
        obs = f["observations"] if "observations" in f else f
        qpos = np.asarray(obs["qpos"][:], dtype=np.float32)
        action = np.asarray(f["action"][:], dtype=np.float32)
        n = min(len(qpos), len(action))

        if "base_action" in f:
            base_action = np.asarray(f["base_action"][:n], dtype=np.float32)
            action = np.concatenate([action[:n], base_action], axis=1).astype(np.float32)
        else:
            action = action[:n]

        instruction = (
            _attr_text(f.attrs, "language_instruction", "instruction", "task")
            or _attr_text(obs.attrs, "language_instruction", "instruction", "task")
            or task_name
            or Path(local_path).parent.name
        )
        task = (
            _attr_text(f.attrs, "task_name", "task")
            or task_name
            or Path(local_path).parent.name
        )
        ep_stem = Path(local_path).stem

        steps = tuple(
            Step(
                timestep=i,
                state=qpos[i],
                action=action[i],
                reward=None,
                done=(i == n - 1),
                is_terminal=(i == n - 1),
            )
            for i in range(n)
        )
        return Episode(
            episode_id=f"aloha/{task}/{ep_stem}",
            source="aloha",
            instruction=instruction or "",
            steps=steps,
            success=True,
            terminal_failure=None,
            model="aloha",
            policy_type="aloha",
            task_name=task,
            metadata={"control_mode": "joint"},
        )
    finally:
        f.close()


class AlohaIngestor(Ingestor):
    """Adapter for ACT-style ALOHA and Mobile ALOHA HDF5 episodes."""

    source = "aloha"

    def load(self, path: str, *, limit: int | None = None) -> Iterator[Episode]:
        paths = _episode_paths(path)
        if limit is not None:
            paths = paths[:limit]
        root = Path(path) if Path(path).is_dir() else Path(path).parent
        for h5_path in paths:
            rel_parent = Path(os.path.relpath(Path(h5_path).parent, root))
            task_name = None if str(rel_parent) == "." else rel_parent.as_posix()
            yield parse_aloha_hdf5(h5_path, task_name=task_name)
