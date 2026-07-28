"""Policy interface: reset(instruction) then act(observation) -> 7-DoF action."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np

from physical_ai_evals.core.schema import ACTION_DIM


class Observation(TypedDict, total=False):
    """Runner observation; ``image`` required (already de-rotated agentview RGB)."""

    image: np.ndarray
    wrist_image: np.ndarray | None
    state: np.ndarray | None
    instruction: str


class Policy(ABC):
    """VLA policy: observation -> unnormalized 7-DoF action (runner clips to [-1, 1])."""

    action_dim: int = ACTION_DIM
    control_mode: str = "relative"  # must match checkpoint training mode

    @abstractmethod
    def reset(self, instruction: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Observation) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        """Release model resources. Optional for backends."""
