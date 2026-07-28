"""OpenVLA policy adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from physical_ai_evals.policy.base import Observation, Policy

LIBERO_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "libero_spatial": ("openvla/openvla-7b-finetuned-libero-spatial", "libero_spatial"),
    "libero_object": ("openvla/openvla-7b-finetuned-libero-object", "libero_object"),
    "libero_goal": ("openvla/openvla-7b-finetuned-libero-goal", "libero_goal"),
    "libero_10": ("openvla/openvla-7b-finetuned-libero-10", "libero_10"),
}

PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def _derive_unnorm_key(model_id: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for _suite, (mid, key) in LIBERO_CHECKPOINTS.items():
        if mid == model_id:
            return key
    return None


def _center_crop(img, scale: float = 0.9):
    """Center-crop to ``scale`` area, resize back (matches OpenVLA LIBERO eval)."""
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    side = float(np.sqrt(scale))
    ch, cw = max(1, round(h * side)), max(1, round(w * side))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = arr[top:top + ch, left:left + cw]
    try:
        from PIL import Image
    except ImportError:
        return cropped
    return np.asarray(
        Image.fromarray(cropped.astype(np.uint8, copy=False)).resize((w, h), Image.Resampling.BILINEAR)
    )


def _as_pil(img):
    if not isinstance(img, np.ndarray):
        return img
    try:
        from PIL import Image
    except ImportError:
        return img
    return Image.fromarray(img)


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class OpenVLAPolicy(Policy):
    control_mode = "relative"

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b-finetuned-libero-spatial",
        unnorm_key: str | None = None,
        device: str = "cuda",
        attn_impl: str = "flash_attention_2",
        center_crop: bool = True,
        *,
        _vla=None,
        _processor=None,
    ) -> None:
        self.model_id = model_id
        self.unnorm_key = _derive_unnorm_key(model_id, unnorm_key)
        self.device = device
        self.attn_impl = attn_impl
        self.center_crop = center_crop
        self._instruction = ""
        self.vla: Any = None
        self.processor: Any = None
        if _vla is not None or _processor is not None:
            self.vla, self.processor = _vla, _processor
        else:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        on_cuda = self.device == "cuda"
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            attn_implementation=self.attn_impl if on_cuda else "sdpa",
            torch_dtype=torch.bfloat16 if on_cuda else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        stats = getattr(self.vla, "norm_stats", None)
        if stats and self.unnorm_key not in stats:
            keys = list(stats.keys())
            if len(keys) == 1:
                self.unnorm_key = keys[0]

    def reset(self, instruction: str) -> None:
        self._instruction = instruction

    def act(self, observation: Observation) -> np.ndarray:
        img = _center_crop(observation["image"]) if self.center_crop else observation["image"]
        prompt = PROMPT_TEMPLATE.format(instruction=self._instruction)
        inputs = self.processor(prompt, _as_pil(img))
        if hasattr(inputs, "to"):
            import torch
            inputs = inputs.to(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32)
        action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        action = np.asarray(_to_numpy(action), dtype=np.float32).reshape(-1)[: self.action_dim].copy()
        g = 2.0 * (action[-1] - 0.5)
        action[-1] = -(1.0 if g > 0.0 else -1.0)  # RLDS gripper -> LIBERO convention
        return action

    def close(self) -> None:
        self.vla = None
        self.processor = None
