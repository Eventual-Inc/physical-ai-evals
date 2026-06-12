"""Normalized Episode/Step representation + the abstract Ingestor contract.

CONCRETE (not a stub). Every dataset adapter — ``lerobot.py``, ``droid.py``,
``hdf5.py`` — yields the SAME ``Episode`` object regardless of on-disk format, and
``Episode.to_step_rows`` emits exactly the columns of ``harness.schema.ROLLOUT_SCHEMA``.
That single normalization point is what makes "ingest DROID/LeRobot/HDF5 + run your
own rollouts" land in one queryable Daft DataFrame.

Conventions fixed here (the three formats disagree on all of these — see NOTES.md):
  * Images: ``uint8`` HWC, keyed by a CANONICAL ROLE ('primary' / 'wrist'), never the
    source-specific camera name. Each adapter maps its keys via ``camera_role_map``.
  * State / action: ``float32`` (DROID stores f64 — cast on the way in).
  * Instruction: hoisted to the Episode (LIBERO HDF5 has no per-frame instruction;
    DROID has per-step; LeRobot has per-frame 'task' — all collapse to one string).
  * success: per-Episode bool. Derivation differs by source (reward.max()==1 /
    dones[-1] / DROID is_terminal / sim predicate) — each adapter sets it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from harness.schema import SCHEMA_VERSION, empty_step_row

#: Canonical camera roles. Adapters map their source keys onto these.
PRIMARY = "primary"
WRIST = "wrist"

#: Default source-camera-key -> canonical-role maps. Adapters use these unless the
#: IngestConfig overrides them. Centralized so the role vocabulary stays uniform.
DEFAULT_CAMERA_ROLE_MAPS: dict[str, dict[str, str]] = {
    "lerobot": {
        "observation.images.image": PRIMARY,
        "observation.images.image2": WRIST,
        "observation.images.wrist_image": WRIST,
        "observation.images.agentview": PRIMARY,
    },
    "droid": {
        "exterior_image_1_left": PRIMARY,
        "wrist_image_left": WRIST,
    },
    "hdf5": {
        "agentview_image": PRIMARY,       # robosuite-canonical (post-processed demos)
        "robot0_eye_in_hand_image": WRIST,
        "agentview_rgb": PRIMARY,          # native-LIBERO alias
        "eye_in_hand_rgb": WRIST,
    },
}


@dataclass(frozen=True)
class Step:
    """One timestep of a trajectory, normalized across formats.

    Attributes
    ----------
    timestep:      0-based index within the episode.
    images:        dict[role -> HWC uint8 ndarray]; roles are 'primary'/'wrist'.
    state:         proprio vector, float32 (None if the source has none).
    action:        action vector, float32 (None for the final no-action frame).
    reward:        sparse reward {0,1} or None.
    done:          env/episode done flag at this step.
    is_terminal:   true terminal (DROID is_terminal / dones[-1]).
    eef_pos:       end-effector xyz (3,) float32, for slip/lift detection (None if absent).
    gripper_state: scalar gripper opening (None if absent).
    object_poses:  {object_name: [x,y,z,qx,qy,qz,qw]} for tracked manipulands (may be empty).
    timestamp:     seconds from episode start, or None.
    """

    timestep: int
    images: dict[str, np.ndarray] = field(default_factory=dict)
    state: np.ndarray | None = None
    action: np.ndarray | None = None
    reward: float | None = None
    done: bool = False
    is_terminal: bool = False
    eef_pos: np.ndarray | None = None
    gripper_state: float | None = None
    object_poses: dict[str, list[float]] = field(default_factory=dict)
    timestamp: float | None = None


@dataclass(frozen=True)
class Episode:
    """An ordered trajectory plus episode-level metadata.

    ``episode_id`` is the join/dedup key. For LIBERO rollouts it is the deterministic
    quadruple ``f"{suite}/{task_id}/{init_state_id}/{seed}"``; for ingest sources it is
    ``f"{source}/{dataset}/{index}"``.
    """

    episode_id: str
    source: str                       # 'lerobot' | 'droid' | 'hdf5' | 'libero'
    instruction: str
    steps: tuple[Step, ...]
    success: bool
    terminal_failure: str | None = None
    model: str = "dataset"            # ingest provenance has no policy; overridden by rollouts
    policy_type: str = "dataset"
    suite: str | None = None
    task_id: int | None = None
    task_name: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def to_step_rows(
        self,
        *,
        run_id: str = "ingest",
        frame_path_for=None,
        wrist_path_for=None,
        video_path: str | None = None,
    ) -> list[dict[str, object]]:
        """Flatten this episode into rows matching ``harness.schema.ROLLOUT_SCHEMA``.

        Parameters
        ----------
        run_id:          groups rows from one ingest/rollout invocation.
        frame_path_for:  optional ``(episode_id, step) -> str`` returning the on-disk
                         path where the primary frame for that step was written. If
                         None, ``frame_path`` is left null (images stay in-memory only).
        wrist_path_for:  same, for the wrist frame.
        video_path:      per-episode mp4 path, denormalized onto every row.

        Returns a list of dicts (one per step), each built from ``empty_step_row()`` so
        no column is ever silently dropped. The writer feeds these straight into
        ``schema.validate_rows`` -> parquet.
        """
        rows: list[dict[str, object]] = []
        for step in self.steps:
            row = empty_step_row()
            row.update(
                schema_version=SCHEMA_VERSION,
                episode_id=self.episode_id,
                run_id=run_id,
                model=self.model,
                policy_type=self.policy_type,
                source=self.source,
                suite=self.suite,
                task_id=self.task_id,
                task_name=self.task_name,
                instruction=self.instruction,
                control_mode=self.metadata.get("control_mode"),
                # reproducibility/provenance columns carried via metadata (LIBERO sets
                # bddl_file; rollouts set init_state_id/seed). Declared in the schema but
                # previously never populated — wire them here so adapters can fill them.
                bddl_file=self.metadata.get("bddl_file"),
                init_state_id=self.metadata.get("init_state_id"),
                seed=self.metadata.get("seed"),
                success=self.success,
                terminal_failure=self.terminal_failure,
                num_steps=self.num_steps,
                step_idx=step.timestep,
                action=None if step.action is None else _as_f32_list(step.action),
                reward=None if step.reward is None else float(step.reward),
                done=bool(step.done),
                state=None if step.state is None else _as_f32_list(step.state),
                eef_pos=None if step.eef_pos is None else _as_f32_list(step.eef_pos),
                gripper_state=step.gripper_state,
                gripper_action=(
                    None if step.action is None else float(np.asarray(step.action).ravel()[-1])
                ),
                object_poses=json.dumps(step.object_poses) if step.object_poses else None,
                frame_path=(frame_path_for(self.episode_id, step.timestep)
                            if frame_path_for else None),
                wrist_path=(wrist_path_for(self.episode_id, step.timestep)
                            if wrist_path_for else None),
                video_path=video_path,
                embedding=None,  # populated by the embedding pass, never at ingest time
            )
            rows.append(row)
        return rows


def _as_f32_list(arr) -> list[float]:
    """Cast any array-like to a flat ``list[float]`` (float32 semantics)."""
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


class Ingestor(ABC):
    """Abstract adapter: one on-disk dataset format -> stream of normalized Episodes.

    Each concrete adapter (lerobot/droid/hdf5) lazy-imports its heavyweight dependency
    inside ``__init__``/``load`` so that merely importing ``harness`` does not require
    lerobot + tensorflow + h5py all at once.
    """

    #: 'lerobot' | 'droid' | 'hdf5' — selects the default camera-role map.
    source: str = "base"

    def __init__(self, camera_role_map: dict[str, str] | None = None) -> None:
        self.camera_role_map = (
            camera_role_map or DEFAULT_CAMERA_ROLE_MAPS.get(self.source, {})
        )

    @abstractmethod
    def load(self, path: str, *, limit: int | None = None) -> Iterator[Episode]:
        """Yield normalized ``Episode`` objects from ``path``.

        ``path`` is a local directory / file, an HF repo_id (lerobot), or a gs:// root
        (droid). ``limit`` caps episodes for smoke tests. Implementations MUST:
          * map source camera keys to canonical roles via ``self.camera_role_map``,
          * cast images to uint8 HWC and state/action to float32,
          * hoist the instruction to the Episode,
          * set ``success`` per the source's success rule.
        """
        raise NotImplementedError(self.load.__doc__)
