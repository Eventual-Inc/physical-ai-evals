"""OpenVLA policy adapter — the comparative BASELINE.

OpenVLA is a released HF model loaded via ``AutoModelForVision2Seq`` + ``AutoProcessor`` with
``trust_remote_code=True``. It is version-sensitive: the remote ``predict_action`` path needs
transformers==4.40.1 / tokenizers==0.19.1 / timm==0.9.10 / torch==2.2.0 — NEWER transformers
silently breaks it (see NOTES.md). This stack CONFLICTS with VLA-JEPA's modern LeRobot stack,
so the two policies run in SEPARATE environments.

The single load-bearing call is ``vla.predict_action(**inputs, unnorm_key=..., do_sample=False)``.
``unnorm_key`` is a silent-failure trap (wrong key -> flailing-but-plausible actions, no error);
per LIBERO suite it is ``'<suite>_no_noops'``.

Heavy deps (torch/transformers/PIL) are imported lazily inside ``_load``/``act`` so importing
this module is cheap and the adapter is unit-testable with an injected backend (``_vla`` /
``_processor``) — no GPU or weights needed to exercise the plumbing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from harness.rollout.policy import Observation, Policy

#: Suite key -> (published OpenVLA fine-tune id, its unnorm_key). Wiring these correctly is
#: the difference between ~95% and ~0% success.
#: unnorm_key is the SUITE name (verified empirically: the fine-tuned checkpoint's norm_stats is
#: keyed 'libero_spatial', NOT the '<suite>_no_noops' training-dataset name; see NOTES.md).
LIBERO_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "libero_spatial": ("openvla/openvla-7b-finetuned-libero-spatial", "libero_spatial"),
    "libero_object": ("openvla/openvla-7b-finetuned-libero-object", "libero_object"),
    "libero_goal": ("openvla/openvla-7b-finetuned-libero-goal", "libero_goal"),
    "libero_10": ("openvla/openvla-7b-finetuned-libero-10", "libero_10"),
}

#: OpenVLA prompt template. The instruction goes inside the braces.
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def _derive_unnorm_key(model_id: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for _suite, (mid, key) in LIBERO_CHECKPOINTS.items():
        if mid == model_id:
            return key
    return None


def _center_crop(img, scale: float = 0.9):
    """Center-crop to ``scale`` of the AREA (side factor sqrt(scale)), resize back.

    The LIBERO fine-tunes were trained with random-crop augmentation; OpenVLA's own eval
    applies this at inference (``center_crop=True``). Skipping it silently costs success rate.
    """
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    side = float(np.sqrt(scale))
    ch, cw = max(1, round(h * side)), max(1, round(w * side))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = arr[top:top + ch, left:left + cw]
    try:
        from PIL import Image
    except ImportError:  # tests without pillow: return the crop un-resized
        return cropped
    return np.asarray(
        Image.fromarray(cropped.astype(np.uint8, copy=False)).resize((w, h), Image.Resampling.BILINEAR)
    )


def _as_pil(img):
    """ndarray -> PIL.Image for the processor. Degrades to the raw array when Pillow is absent
    (unit tests); the real OpenVLA env has Pillow (transformers/timm pull it)."""
    if not isinstance(img, np.ndarray):
        return img
    try:
        from PIL import Image
    except ImportError:
        return img
    return Image.fromarray(img)


def _to_numpy(x):
    if hasattr(x, "detach"):       # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class OpenVLAPolicy(Policy):
    """OpenVLA baseline. ``control_mode='relative'`` (delta actions).

    REQUIRES (own venv): transformers==4.40.1, tokenizers==0.19.1, timm==0.9.10, torch==2.2.0.
    On Apple Silicon/CPU drop ``flash_attention_2`` for ``attn_implementation='sdpa'`` + float32
    (slow, single-episode smoke throughput, not full-suite runs).
    """

    control_mode = "relative"

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b-finetuned-libero-spatial",
        unnorm_key: str | None = None,
        device: str = "cuda",
        attn_impl: str = "flash_attention_2",
        center_crop: bool = True,   # LIBERO fine-tunes trained w/ crop aug; eval must crop too
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
        # Backends are dynamically loaded heavy objects (or injected fakes) -> typed Any.
        self.vla: Any = None
        self.processor: Any = None
        if _vla is not None or _processor is not None:  # test/injection seam
            self.vla, self.processor = _vla, _processor
        else:
            self._load()

    def _load(self) -> None:
        """Load the OpenVLA model + processor (lazy heavy imports)."""
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

        # Robustness: if our derived unnorm_key isn't among the checkpoint's norm_stats, fall
        # back to the sole available key (the fine-tunes ship a single dataset's stats). Guards
        # against the suite-name vs '<suite>_no_noops' naming mismatch (NOTES.md).
        stats = getattr(self.vla, "norm_stats", None)
        if stats and self.unnorm_key not in stats:
            keys = list(stats.keys())
            if len(keys) == 1:
                self.unnorm_key = keys[0]

    def reset(self, instruction: str) -> None:
        """Store the instruction for prompt construction (OpenVLA is stateless per step)."""
        self._instruction = instruction

    def act(self, observation: Observation) -> np.ndarray:
        """Predict one 7-DoF action from the (already de-rotated) image + stored instruction.

        Mirrors OpenVLA's official eval post-processing (``experiments/robot``): center-crop
        (the LIBERO fine-tunes train with random-crop augmentation; eval crops to 0.9 area) and
        the gripper convention conversion — ``predict_action`` returns RLDS convention
        (gripper in [0,1], ~1=open) which LIBERO reads as "never open" (verified: 0/7 SR, every
        commanded gripper >= 0; NOTES.md). normalize [0,1] -> [-1,1], binarize, then INVERT.
        """
        img = _center_crop(observation["image"]) if self.center_crop else observation["image"]
        prompt = PROMPT_TEMPLATE.format(instruction=self._instruction)
        inputs = self.processor(prompt, _as_pil(img))
        if hasattr(inputs, "to"):  # real transformers BatchFeature; plain dict in tests has no .to
            import torch
            inputs = inputs.to(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32)
        action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        action = np.asarray(_to_numpy(action), dtype=np.float32).reshape(-1)[: self.action_dim].copy()
        g = 2.0 * (action[-1] - 0.5)                     # RLDS [0,1] -> [-1,1]
        action[-1] = -(1.0 if g > 0.0 else -1.0)         # binarize, then invert to LIBERO -1=open/+1=close
        return action

    def close(self) -> None:
        self.vla = None
        self.processor = None
