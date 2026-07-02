"""VLA-JEPA policy adapter — in-process via the lerobot port (the comparative HEADLINER).

lerobot ships a first-party VLA-JEPA policy (``lerobot.policies.vla_jepa``, merged upstream
2026-06 — main-only until the next release, so pin lerobot at a git SHA) plus the official
checkpoint ``lerobot/VLA-JEPA-LIBERO`` (safetensors). That makes VLA-JEPA a clean in-process
citizen of the stack: no StarVLA policy server, no WebSocket, no manual
``dataset_statistics.json`` handling — the git history holds the deleted server route.

Division of labor (verified against lerobot main @ 052d3294):

* ``from_pretrained`` loads weights (and pulls the Qwen3-VL-2B + V-JEPA2 backbones — three HF
  repos total). Processor pipelines are NOT auto-wired; ``make_pre_post_processors`` loads the
  hub's ``policy_preprocessor.json`` / ``policy_postprocessor.json``.
* The PREprocessor batches/normalizes (STATE mean-std, ACTION min-max; images identity).
* ``select_action`` dequeues ONE action per call from an internal ``n_action_steps=7`` queue —
  chunking lives in the lerobot policy, so this adapter keeps no chunk cache. ``reset()``
  clears that queue (mandatory between episodes).
* The POSTprocessor unnormalizes AND binarizes the gripper to LIBERO-ready {-1, +1}.

The runner's obs contract already matches lerobot's own LIBERO processor: agentview + wrist
de-rotated 180°, 8-dim state = eef_pos(3) + axis-angle(3) + gripper_qpos(2). The adapter only
converts HWC uint8 -> (1,3,H,W) float32 in [0,1] (mandatory: the Qwen processor runs with
``do_rescale=False``) — the model area-resizes to 224 internally.

Heavy deps stay lazy; tests inject ``_policy``/``_preprocessor``/``_postprocessor`` so no
lerobot/GPU/weights are needed for CPU verification.
"""

from __future__ import annotations

import numpy as np

from harness.rollout.policy import Observation, Policy

DEFAULT_MODEL_ID = "lerobot/VLA-JEPA-LIBERO"


class VLAJEPAPolicy(Policy):
    """In-process lerobot VLA-JEPA. One action out per ``act`` (queue inside the policy)."""

    control_mode = "relative"

    def __init__(
        self,
        policy_path: str | None = None,
        device: str | None = None,
        *,
        _policy=None,
        _preprocessor=None,
        _postprocessor=None,
    ) -> None:
        self.policy_path = str(policy_path or DEFAULT_MODEL_ID)
        self.device = device
        self._instruction = ""
        if _policy is not None:  # test seam: no lerobot / GPU / weights needed
            self._policy = _policy
            self._pre = _preprocessor if _preprocessor is not None else (lambda b: b)
            self._post = _postprocessor if _postprocessor is not None else (lambda a: a)
        else:
            self._load()

    def _load(self) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy as _LeRobotVLAJEPA

        config = None
        if self.device:  # hub config ships device=null -> auto-select unless forced
            config = PreTrainedConfig.from_pretrained(self.policy_path)
            config.device = self.device
        self._policy = _LeRobotVLAJEPA.from_pretrained(self.policy_path, config=config)
        self._policy.eval()
        self.device = str(self._policy.config.device)
        # The saved preprocessor JSON pins device=cpu; override to the policy's device
        # (mirrors lerobot_eval.py).
        self._pre, self._post = make_pre_post_processors(
            self._policy.config,
            pretrained_path=self.policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def reset(self, instruction: str) -> None:
        """Store the instruction and clear the policy's internal action queue (required)."""
        self._instruction = instruction
        self._policy.reset()

    def act(self, observation: Observation) -> np.ndarray:
        """Map the runner's obs dict onto the lerobot batch and return one 7-DoF action."""
        import torch

        if observation.get("wrist_image") is None:
            raise ValueError(
                "lerobot/VLA-JEPA-LIBERO requires the wrist camera "
                "(observation.images.image2) — the runner must pass wrist_image."
            )
        batch: dict = {
            "observation.images.image": self._img(observation["image"]),
            "observation.images.image2": self._img(observation["wrist_image"]),
            "task": observation.get("instruction") or self._instruction,
        }
        if observation.get("state") is not None:
            batch["observation.state"] = torch.from_numpy(
                np.asarray(observation["state"], np.float32)
            ).unsqueeze(0)
        with torch.inference_mode():
            action = self._post(self._policy.select_action(self._pre(batch)))
        return np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim]

    @staticmethod
    def _img(img):
        """HWC uint8 -> (1,3,H,W) float32 in [0,1].

        The /255 is load-bearing: lerobot's Qwen interface calls its processor with
        ``do_rescale=False``, so 0-255 floats would silently wreck inference. Pre-batching
        here (vs the pipeline's to_batch step) keeps the contract visible to injected fakes.
        """
        import torch

        arr = np.asarray(img)
        return (
            torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0).unsqueeze(0)
        )

    def close(self) -> None:
        self._policy = None
        self._pre = None
        self._post = None
