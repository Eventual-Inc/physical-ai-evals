from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from physical_ai_evals.core.schema import EMBEDDING_DIM

CORE_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 250,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
DEFAULT_MAX_STEPS = 300


class RolloutConfig(BaseModel):
    """Config for ``physical-ai-evals rollout``."""

    model_config = ConfigDict(frozen=False)

    policy_type: str
    suites: tuple[str, ...] = CORE_SUITES
    n_episodes_per_task: int = 50
    task_ids: tuple[int, ...] | None = None
    seed: int = 7
    control_mode: str = "relative"

    camera_height: int = 256
    camera_width: int = 256
    num_steps_wait: int = 10
    max_steps: int | None = None

    model_id: str | None = None
    unnorm_key: str | None = None
    device: str = "cuda"

    out_dir: Path = Path("data/rollouts")
    frames_dir: Path = Path("data/frames")
    videos_dir: Path = Path("data/videos")
    write_video: bool = True
    write_frames: bool = True
    run_id: str | None = None

    def resolved_max_steps(self, suite: str) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return SUITE_MAX_STEPS.get(suite, DEFAULT_MAX_STEPS)


class IngestConfig(BaseModel):
    """Config for ``physical-ai-evals ingest``."""

    model_config = ConfigDict(frozen=False)

    source: str
    input_path: str
    out_dir: Path = Path("data/rollouts")
    frames_dir: Path = Path("data/frames")
    write_frames: bool = True
    limit_episodes: int | None = None
    camera_role_map: dict[str, str] | None = None


class EmbedConfig(BaseModel):
    """Config for the embedding pass."""

    model_config = ConfigDict(frozen=False)

    dim: int = EMBEDDING_DIM
    modality: str = "text"
    provider: str = "sentence_transformers"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = Field(default_factory=lambda: os.environ.get("HARNESS_DEVICE", "cpu"))
