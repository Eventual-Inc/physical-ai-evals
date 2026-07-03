"""LeRobot dataset adapter — delegates to the Daft-native reader.

STRATEGY (see NOTES.md): we do NOT hand-roll a LeRobot reader. We call
``daft.datasets.lerobot.read()`` (native in nightly Daft; PyPI from the release after 0.7.16)
which returns a one-row-per-FRAME DataFrame in native LeRobot column names, then NORMALIZE it
onto the common ``Episode``/``Step`` model so it projects through ``Episode.to_step_rows`` onto
the canonical ``ROLLOUT_SCHEMA`` — exactly like every other source. Going through Episode (vs a
direct in-Daft projection) GUARANTEES schema parity; the lazy/at-scale projection that writes
parquet straight from Daft is parked in BACKLOG.

GAP this does not cover: ``read()`` is **v3.0-only**. For LeRobot **v2.1** (what VLA-JEPA
consumes) the ``lerobot`` library is still required — that fallback recipe lives in
``_load_v21`` below (not wired; install ``lerobot>=0.5.0`` and implement when a v2.1 dataset
actually shows up).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import numpy as np

from harness.ingest.base import Episode, Ingestor, Step


def _np1d(v):
    """A frame cell (list/ndarray) -> float32 1-D ndarray, or None."""
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float32).ravel()
    return arr if arr.size else None


class LeRobotIngestor(Ingestor):
    """Adapter for LeRobot v3 datasets via ``daft.datasets.lerobot`` (PR #7090)."""

    source = "lerobot"

    def load(self, path: str, *, limit: int | None = None) -> Iterator[Episode]:
        """Yield normalized Episodes from a LeRobot v3 dataset.

        ``path``: HF repo id (``org/name``), or a local / ``s3://`` / ``hf://`` root.
        ``limit``: cap episodes (smoke tests). Instruction is joined from ``read_tasks``
        on ``task_index``. ``success`` is unknown for raw LeRobot data, so it's set False /
        ``terminal_failure='unlabeled'`` for the notebook's labeling pass.
        """
        import daft.datasets

        frames = daft.datasets.lerobot.read(path, load_video_frames=False).to_pydict()
        task_map = self._task_map(path)

        n = len(frames.get("episode_index", []))
        groups: dict = defaultdict(list)
        for i in range(n):
            groups[frames["episode_index"][i]].append(i)

        action = frames.get("action")
        state = frames.get("observation.state")
        frame_index = frames.get("frame_index")
        task_index = frames.get("task_index")

        for ep_idx in sorted(groups)[: (limit if limit is not None else None)]:
            rows = sorted(groups[ep_idx], key=lambda i: (frame_index[i] if frame_index else i))
            steps = tuple(
                Step(
                    timestep=int(frame_index[i]) if frame_index else pos,
                    action=_np1d(action[i]) if action else None,
                    state=_np1d(state[i]) if state else None,
                )
                for pos, i in enumerate(rows)
            )
            first = rows[0]
            instruction = task_map.get(task_index[first], "") if task_index else ""
            yield Episode(
                episode_id=f"lerobot/{ep_idx}",
                source="lerobot",
                instruction=instruction or "",
                steps=steps,
                success=False,             # not in LeRobot data; notebook labels it
                terminal_failure="unlabeled",
                model=str(path),           # provenance = the dataset id
                policy_type="lerobot",
                metadata={},               # control_mode unknown for arbitrary LeRobot data
            )

    @staticmethod
    def _task_map(path: str) -> dict:
        """``task_index -> instruction`` from ``read_tasks`` (the natural-language command)."""
        import daft.datasets
        try:
            tasks = daft.datasets.lerobot.read_tasks(path).to_pydict()
        except Exception:
            return {}
        idx = tasks.get("task_index")
        txt = tasks.get("task")
        if not idx or not txt:
            return {}
        return {idx[i]: txt[i] for i in range(len(idx))}

    def _load_v21(self, path, *, limit=None):
        """FALLBACK for LeRobot v2.1 (Daft #7090 is v3-only). NOT wired — implement against
        the lerobot lib when a v2.1 dataset appears:

            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(repo_id=path, return_uint8=True)
            assert ds.meta.info['codebase_version'].startswith('v2')
            # group flat frames by episode_index, order by frame_index, map observation.images.*
            # via self.camera_role_map, hoist `task` to the Episode, same Episode contract.
        """
        raise NotImplementedError(self._load_v21.__doc__)
