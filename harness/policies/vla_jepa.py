"""VLA-JEPA policy adapter — the comparative HEADLINER.

VLA-JEPA (arXiv 2602.10098, ginwind, early 2026) ships a first-party LeRobot policy
(``policy.type='vla_jepa'``) with checkpoints under ``lerobot/VLA-JEPA-*``
(``lerobot/VLA-JEPA-LIBERO`` is the LIBERO one). At inference it is a Qwen3-VL-2B backbone +
DiT flow-matching action head; the V-JEPA2 latent world model is training-only.

We adapt the STANDARD LeRobot ``PreTrainedPolicy`` interface (``from_pretrained`` + ``reset``
+ ``select_action``). The action CHUNKING (chunk_size=7, n_action_steps=7) lives INSIDE the
LeRobot policy: ``select_action`` buffers a predicted chunk and re-plans only on chunk
boundaries, and ``reset()`` clears that buffer — so this adapter stays thin and just forwards
``reset``/``select_action``. ``reset()`` between episodes is REQUIRED or a new episode replays
the previous chunk.

HONEST CAVEATS (see NOTES.md, do NOT paper over): the exact LeRobot policy CLASS symbol was
not verified verbatim — we load via the policy factory / ``from_pretrained``, NOT a guessed
class import. Do NOT confuse VLA-JEPA (2602.10098) with the unrelated JEPA-VLA (2602.11832).
The modern LeRobot/Qwen3-VL stack conflicts with OpenVLA's old transformers — own env.

Heavy deps (lerobot/torch) are imported lazily; an injected backend (``_policy``) makes the
batch-assembly + chunk-reset plumbing unit-testable without weights or a GPU.
"""

from __future__ import annotations

import numpy as np

from harness.rollout.policy import Observation, Policy

#: First-party LeRobot checkpoints (HF). LIBERO checkpoint uses 2 cameras (agentview + wrist).
#: Reported LIBERO success: spatial 95 / object 100 / goal 98 / long 93 (~96.5%) — UNVERIFIED.
LEROBOT_CHECKPOINTS: dict[str, str] = {
    "libero": "lerobot/VLA-JEPA-LIBERO",
    "pretrain": "lerobot/VLA-JEPA-Pretrain",
    "simpler": "lerobot/VLA-JEPA-SimplerEnv",
}


def _to_numpy(x):
    if hasattr(x, "detach"):       # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _to_batched_tensor(arr, device: str):
    """array-like -> a leading-batch tensor on ``device``. Degrades to a batched ndarray when
    torch is absent (unit tests); the real VLA-JEPA env has torch via lerobot."""
    if arr is None:
        return None
    try:
        import torch
        return torch.as_tensor(np.asarray(arr)).unsqueeze(0).to(device)
    except ImportError:
        return np.asarray(arr)[None, ...]


class VLAJEPAPolicy(Policy):
    """VLA-JEPA headliner, adapted from a LeRobot ``PreTrainedPolicy``. Chunking is internal to
    the LeRobot policy; this adapter forwards ``reset``/``select_action`` and maps observations
    to the LeRobot batch dict (camera keys agentview/wrist). ``control_mode='relative'``.
    """

    control_mode = "relative"

    def __init__(
        self,
        policy_path: str = "lerobot/VLA-JEPA-LIBERO",
        device: str = "cuda",
        *,
        _policy=None,
    ) -> None:
        self.policy_path = policy_path
        self.device = device
        self._instruction = ""
        if _policy is not None:      # test/injection seam
            self.policy = _policy
        else:
            self._load()

    def _load(self) -> None:
        """Load the VLA-JEPA LeRobot policy via the factory (NOT a guessed class import)."""
        try:
            from lerobot.policies.factory import make_policy
        except ImportError as err:  # pragma: no cover - env-dependent
            raise ImportError(
                "VLA-JEPA needs the LeRobot stack: pip install -e '.[vla_jepa]'. If the "
                "'vla_jepa' policy type isn't registered, fall back to the research repo "
                "(github.com/ginwind/VLA-JEPA) — do NOT guess a class name (see NOTES.md)."
            ) from err
        self.policy = make_policy(policy_type="vla_jepa", policy_path=self.policy_path)
        self.policy.to(self.device).eval()

    def reset(self, instruction: str) -> None:
        """Store the instruction and CLEAR the action-chunk buffer (required between episodes)."""
        self._instruction = instruction
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, observation: Observation) -> np.ndarray:
        """Assemble the LeRobot batch and return one action (the policy buffers the chunk)."""
        action = self.policy.select_action(self._build_batch(observation))
        return np.asarray(_to_numpy(action), dtype=np.float32).reshape(-1)[: self.action_dim]

    def _build_batch(self, obs: Observation) -> dict:
        batch: dict = {"task": self._instruction}
        if obs.get("image") is not None:
            batch["observation.images.agentview"] = _to_batched_tensor(obs["image"], self.device)
        if obs.get("wrist_image") is not None:
            batch["observation.images.wrist"] = _to_batched_tensor(obs["wrist_image"], self.device)
        if obs.get("state") is not None:
            batch["observation.state"] = _to_batched_tensor(obs["state"], self.device)
        return batch

    def close(self) -> None:
        self.policy = None
