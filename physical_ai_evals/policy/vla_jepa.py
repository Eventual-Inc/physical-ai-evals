"""VLA-JEPA policy via in-process lerobot."""

from __future__ import annotations

from typing import Any

import numpy as np

from physical_ai_evals.policy.base import Observation, Policy

DEFAULT_MODEL_ID = "lerobot/VLA-JEPA-LIBERO"


class VLAJEPAPolicy(Policy):
    control_mode = "relative"

    _policy: Any
    _pre: Any
    _post: Any

    def __init__(self, policy_path: str | None = None, device: str | None = None) -> None:
        self.policy_path = str(policy_path or DEFAULT_MODEL_ID)
        self.device = device
        self._load()

    def _load(self) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy as _LRPolicy

        config = PreTrainedConfig.from_pretrained(self.policy_path)
        if self.device:
            config.device = self.device

        self._policy = _LRPolicy.from_pretrained(self.policy_path, config=config).eval()
        self.device = str(self._policy.config.device)
        self._pre, self._post = make_pre_post_processors(
            self._policy.config,
            pretrained_path=self.policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._policy.reset()

    def act(self, observation: Observation) -> np.ndarray:
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
        import torch

        arr = np.ascontiguousarray(np.asarray(img))
        return (
            torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0).unsqueeze(0)
        )

    def close(self) -> None:
        self._policy = None
        self._pre = None
        self._post = None
