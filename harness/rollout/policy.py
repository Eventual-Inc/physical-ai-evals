"""Abstract Policy contract — the single interface both backends implement.

CONCRETE (not a stub). OpenVLA and VLA-JEPA both reduce to
(RGB image + language instruction) -> 7-DoF EEF delta action, so they fit cleanly
behind ``reset(instruction)`` / ``act(observation) -> action``. The runner
(``rollout/libero_runner.py``) only ever talks to this interface; swapping policies
never touches the runner or the schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np

from harness.schema import ACTION_DIM


class Observation(TypedDict, total=False):
    """The observation dict the runner passes to ``Policy.act``.

    ``image`` is the ONLY required key. The agentview frame MUST already be de-rotated
    (LIBERO renders agentview 180deg rotated vs what released VLA checkpoints expect —
    feeding the raw frame silently tanks success to ~0, no exception; see NOTES.md).
    """

    image: np.ndarray               # HxWx3 uint8, de-rotated agentview RGB  (REQUIRED)
    wrist_image: np.ndarray | None  # HxWx3 uint8, wrist cam — None when the env has no wrist cam
    state: np.ndarray | None        # proprio vector — None when the policy is vision-only
    instruction: str                # the language command (also given via reset; kept for convenience)


class Policy(ABC):
    """A VLA policy mapping observation -> action.

    Action contract: ``act`` returns ``np.float32`` shape ``(ACTION_DIM,)`` = 7-DoF EEF
    delta ``(x, y, z, roll, pitch, yaw, gripper)`` in robot units, already
    UN-NORMALIZED and ready to pass to ``env.step``. The first 6 are continuous
    deltas; index 6 is the gripper command. Values are expected within the env's
    ``Box(-1, 1)``; the runner clips defensively.
    """

    #: 7. Exposed so the runner/writer can validate without importing schema.
    action_dim: int = ACTION_DIM

    #: 'relative' | 'absolute' — MUST match the checkpoint's training parameterization
    #: or rollouts fail silently. Set by the concrete policy at construction.
    control_mode: str = "relative"

    @abstractmethod
    def reset(self, instruction: str) -> None:
        """Begin a new episode with ``instruction``.

        Clears any per-episode state (action-chunk buffer for chunked policies like
        VLA-JEPA, cached prompt for OpenVLA). MUST be called between episodes.
        """
        raise NotImplementedError(self.reset.__doc__)

    @abstractmethod
    def act(self, observation: Observation) -> np.ndarray:
        """Return the next action for ``observation``.

        Returns ``np.ndarray`` shape ``(action_dim,)``, dtype float32. For chunked
        policies, internally buffers a predicted chunk and only re-plans on chunk
        boundaries — the caller still gets exactly one action per call.
        """
        raise NotImplementedError(self.act.__doc__)

    def close(self) -> None:  # noqa: B027  (intentionally a no-op default)
        """Release model resources (GPU memory, file handles). Default no-op."""
