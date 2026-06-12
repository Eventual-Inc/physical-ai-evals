"""Run configuration for the rollout harness.

CONCRETE dataclasses (no stubs). These are the knobs the CLI fills in and the
rollout/ingest paths consume. Kept dependency-free (stdlib only) so importing config
never drags in torch/robosuite/daft.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from harness.schema import EMBEDDING_DIM

#: The 4 'core' LIBERO suites used by the standard published VLA eval protocol
#: (10 episodes/task each = 400 episodes). 'libero_10' is LIBERO-Long.
CORE_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

#: OpenPI-style per-suite step caps (short suites ~220, long ~520). The long suite
#: needs the bigger budget; everything else shares the short cap.
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 220,
    "libero_goal": 220,
    "libero_10": 520,
}
DEFAULT_MAX_STEPS = 300


@dataclass
class RolloutConfig:
    """Config for a LIBERO rollout sweep (``harness rollout``).

    The (suite, task_id, init_state_id, seed) quadruple is the deterministic episode
    key — see NOTES.md. ``control_mode`` MUST match the policy's training
    parameterization or rollouts fail *silently* (near-zero success, no exception).
    """

    policy_type: str                       # 'openvla' | 'vla_jepa'
    suites: tuple[str, ...] = CORE_SUITES
    n_episodes_per_task: int = 10          # standard protocol = 10/task
    task_ids: tuple[int, ...] | None = None  # None -> all tasks in each suite
    seed: int = 0
    control_mode: str = "relative"         # 'relative' (delta) | 'absolute'

    # env construction (mirrors OpenPI _get_libero_env)
    camera_height: int = 256
    camera_width: int = 256
    num_steps_wait: int = 10               # settle steps (zero action) before policy control
    max_steps: int | None = None           # None -> per-suite SUITE_MAX_STEPS

    # policy loading (passed through to harness/policies/*)
    model_id: str | None = None            # HF id / lerobot repo id; None -> policy default
    unnorm_key: str | None = None          # OpenVLA only, e.g. 'libero_goal_no_noops'
    device: str = "cuda"                   # 'cuda' | 'cpu' | 'mps'; CPU/MPS drop flash-attn

    # output / capture
    out_dir: Path = Path("data/rollouts")
    frames_dir: Path = Path("data/frames")
    videos_dir: Path = Path("data/videos")
    write_video: bool = True
    write_frames: bool = True              # per-step frames (needed for image embeddings)
    run_id: str | None = None              # None -> auto timestamp at CLI time

    def resolved_max_steps(self, suite: str) -> int:
        """Per-suite step cap, honoring an explicit ``max_steps`` override."""
        if self.max_steps is not None:
            return self.max_steps
        return SUITE_MAX_STEPS.get(suite, DEFAULT_MAX_STEPS)


@dataclass
class IngestConfig:
    """Config for normalizing an existing dataset into rollout parquet (``harness ingest``)."""

    source: str                            # 'lerobot' | 'droid' | 'hdf5'
    input_path: str                        # local path or repo_id (lerobot) / gs:// (droid)
    out_dir: Path = Path("data/rollouts")
    frames_dir: Path = Path("data/frames")
    write_frames: bool = True
    limit_episodes: int | None = None      # cap for smoke tests (e.g. droid_100 subset)
    #: Override the source camera-key -> canonical-role map (see ingest/base.py).
    camera_role_map: dict[str, str] | None = None


@dataclass
class EmbedConfig:
    """Config for the embedding pass that populates the ``embedding`` column.

    Lives here (not in the notebook) so the harness and notebook share one DIM/model.
    GPU is optional — ``device`` falls back to CPU so CI can run a tiny model.
    """

    dim: int = EMBEDDING_DIM
    modality: str = "text"                 # 'text' (instruction) | 'image' (frames)
    provider: str = "sentence_transformers"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = field(default_factory=lambda: os.environ.get("HARNESS_DEVICE", "cpu"))
