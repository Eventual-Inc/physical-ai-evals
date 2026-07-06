"""ABC exported-episode adapter.

The ABC release (https://abc.bot/) owns raw MCAP conversion. Its documented exported
training layout is::

    episode_<uuid>/
      states_actions.bin               # float64, first half state, second half action
      combined_camera-images-rgb.mp4
      episode_metadata.json             # task_name, cameras, timing, num_steps

This adapter reads that stable export format. It intentionally does not parse raw MCAPs.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from harness.ingest.base import Episode, Ingestor, Step

STATES_ACTIONS = "states_actions.bin"
METADATA = "episode_metadata.json"
VIDEO = "combined_camera-images-rgb.mp4"


def _episode_dirs(path: str) -> list[Path]:
    p = Path(path)
    if p.is_dir() and (p / STATES_ACTIONS).exists():
        return [p]
    return [Path(x).parent for x in sorted(glob.glob(str(p / "**" / STATES_ACTIONS), recursive=True))]


def parse_abc_episode_dir(episode_dir: str | Path, *, root: str | Path | None = None) -> Episode:
    """Parse one ABC exported episode directory."""
    episode_dir = Path(episode_dir)
    meta_path = episode_dir / METADATA
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    raw = np.fromfile(episode_dir / STATES_ACTIONS, dtype=np.float64)
    num_steps = int(meta.get("num_steps") or 0)
    if num_steps <= 0:
        if raw.size % 28 != 0:
            raise ValueError(f"{episode_dir}: cannot infer num_steps from {STATES_ACTIONS}")
        num_steps = raw.size // 28
    if raw.size % num_steps != 0:
        raise ValueError(f"{episode_dir}: {STATES_ACTIONS} size is not divisible by num_steps")
    width = raw.size // num_steps
    if width % 2 != 0:
        raise ValueError(f"{episode_dir}: expected even state/action width, got {width}")

    sa = raw.reshape(num_steps, width).astype(np.float32)
    split = width // 2
    states, actions = sa[:, :split], sa[:, split:]

    base = Path(root) if root is not None else episode_dir.parent
    rel = Path(os.path.relpath(episode_dir, base)).as_posix()
    task = meta.get("task_name") or episode_dir.parent.name
    video_path = episode_dir / VIDEO
    metadata = {
        "control_mode": "joint",
        "abc_cameras": meta.get("cameras"),
        "abc_alignment": meta.get("alignment"),
    }
    if video_path.exists():
        metadata["video_path"] = str(video_path)

    steps = tuple(
        Step(
            timestep=i,
            state=states[i],
            action=actions[i],
            reward=None,
            done=(i == num_steps - 1),
            is_terminal=(i == num_steps - 1),
        )
        for i in range(num_steps)
    )
    return Episode(
        episode_id=f"abc/{rel}",
        source="abc",
        instruction=task or "",
        steps=steps,
        success=True,
        terminal_failure=None,
        model="abc",
        policy_type="abc",
        task_name=task,
        metadata=metadata,
    )


class ABCIngestor(Ingestor):
    """Adapter for ABC's converted ``episode_<uuid>`` training format."""

    source = "abc"

    def load(self, path: str, *, limit: int | None = None) -> Iterator[Episode]:
        dirs = _episode_dirs(path)
        if limit is not None:
            dirs = dirs[:limit]
        root = Path(path) if Path(path).is_dir() and not (Path(path) / STATES_ACTIONS).exists() else Path(path).parent
        for episode_dir in dirs:
            yield parse_abc_episode_dir(episode_dir, root=root)
