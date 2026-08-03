"""Built-in OpenVLA and VLA-JEPA policy factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np

from physical_ai_evals.schema import ACTION_DIM


class Observation(TypedDict, total=False):
    """Normalized policy input supplied by the LIBERO runtime."""

    image: np.ndarray
    wrist_image: np.ndarray | None
    state: np.ndarray | None
    instruction: str
    episode_key: str
    seed: int


class Policy(Protocol):
    """Minimal structural policy contract; inheritance is not required."""

    action_dim: int
    control_mode: str

    def reset(self, instruction: str) -> None: ...

    def act(self, observation: Observation) -> np.ndarray: ...

    def close(self) -> None: ...


class BatchPolicy(Policy, Protocol):
    """Optional policy extension for synchronized environment batches."""

    def reset_batch(self, instructions: Sequence[str]) -> None: ...

    def act_batch(self, observations: Sequence[Observation]) -> np.ndarray: ...


@dataclass(frozen=True)
class PolicySpec:
    """Reproducible policy factory loaded once by the rollout process."""

    factory: Callable[[], Policy]
    policy_id: str
    revision: str
    control_mode: str = "relative"
    camera_height: int = 256
    camera_width: int = 256
    num_steps_wait: int = 10
    frames_per_second: int = 20
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if not self.revision.strip():
            raise ValueError(
                "revision must identify the model and user policy implementation; "
                "resume is unsafe without it"
            )
        if self.control_mode != "relative":
            raise ValueError("LIBERO currently supports only relative actions")


OPENVLA_CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "libero_spatial": (
        "openvla/openvla-7b-finetuned-libero-spatial",
        "962318cec55ac10993ff0f5f43eda9a270b4c873",
        "libero_spatial",
    ),
    "libero_object": (
        "openvla/openvla-7b-finetuned-libero-object",
        "287d6cfdf12d07b1449505f66d9bf3550257e9b3",
        "libero_object",
    ),
    "libero_goal": (
        "openvla/openvla-7b-finetuned-libero-goal",
        "fa5ae1e7509348889295bba8e08621d8b55e9baf",
        "libero_goal",
    ),
    "libero_10": (
        "openvla/openvla-7b-finetuned-libero-10",
        "80970322773f81baa2e22fe495d0487b93a05cfa",
        "libero_10",
    ),
}
VLA_JEPA_MODEL_ID = "lerobot/VLA-JEPA-LIBERO"
VLA_JEPA_REVISION = "735d9f692981e286ade093b5046627eda876e5d0"
QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_VL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
_PROMPT = "In: What action should the robot take to {instruction}?\nOut:"


def _snapshot(model_id: str, revision: str) -> str:
    path = Path(model_id).expanduser()
    if path.exists():
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, revision=revision)


def _center_crop(image: Any, scale: float = 0.9) -> np.ndarray:
    array = np.asarray(image)
    height, width = array.shape[:2]
    side = float(np.sqrt(scale))
    crop_height = max(1, round(height * side))
    crop_width = max(1, round(width * side))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    cropped = array[top : top + crop_height, left : left + crop_width]
    try:
        from PIL import Image
    except ImportError:
        return cropped
    return np.asarray(
        Image.fromarray(cropped.astype(np.uint8, copy=False)).resize(
            (width, height), Image.Resampling.BILINEAR
        )
    )


def _as_pil(image: Any) -> Any:
    if not isinstance(image, np.ndarray):
        return image
    try:
        from PIL import Image
    except ImportError:
        return image
    return Image.fromarray(image)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class OpenVLAPolicy:
    """OpenVLA adapter implementing the structural policy protocol."""

    action_dim = ACTION_DIM
    control_mode = "relative"

    def __init__(
        self,
        model_path: str,
        *,
        unnorm_key: str,
        device: str = "cuda",
        attention: str = "sdpa",
        center_crop: bool = True,
        _model: Any = None,
        _processor: Any = None,
    ) -> None:
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.device = device
        self.attention = attention
        self.center_crop = center_crop
        self._instruction = ""
        self.model: Any = _model
        self.processor: Any = _processor
        if self.model is None or self.processor is None:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        on_cuda = self.device == "cuda"
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=self.attention if on_cuda else "sdpa",
            torch_dtype=torch.bfloat16 if on_cuda else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        statistics = getattr(self.model, "norm_stats", None)
        if statistics and self.unnorm_key not in statistics:
            raise ValueError(
                f"OpenVLA checkpoint does not contain unnorm_key "
                f"{self.unnorm_key!r}; available keys: {sorted(statistics)}"
            )

    def reset(self, instruction: str) -> None:
        self._instruction = instruction

    def _act(self, observation: Observation, instruction: str) -> np.ndarray:
        image = _center_crop(observation["image"]) if self.center_crop else observation["image"]
        inputs = self.processor(_PROMPT.format(instruction=instruction), _as_pil(image))
        if hasattr(inputs, "to"):
            import torch

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            inputs = inputs.to(self.device, dtype=dtype)
        action = self.model.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=False,
        )
        result = np.asarray(_to_numpy(action), dtype=np.float32).reshape(-1)[:ACTION_DIM].copy()
        gripper = 2.0 * (result[-1] - 0.5)
        result[-1] = -(1.0 if gripper > 0.0 else -1.0)
        return result

    def act(self, observation: Observation) -> np.ndarray:
        return self._act(observation, observation.get("instruction") or self._instruction)

    def reset_batch(self, instructions: Sequence[str]) -> None:
        self._instructions = list(instructions)

    def act_batch(self, observations: Sequence[Observation]) -> np.ndarray:
        """Run a CPU environment batch through upstream batch-one decoding.

        OpenVLA's pinned remote implementation rejects generation batches larger
        than one. Keeping the loop here still lets MuJoCo environments advance
        concurrently without loading more than one copy of the model.
        """
        if len(observations) != len(self._instructions):
            raise ValueError("OpenVLA observation batch does not match the reset instruction batch")
        return np.stack(
            [
                self._act(observation, observation.get("instruction") or instruction)
                for observation, instruction in zip(observations, self._instructions, strict=True)
            ]
        )

    def close(self) -> None:
        self.model = None
        self.processor = None


@dataclass(frozen=True)
class _OpenVLAFactory:
    model_id: str
    revision: str
    unnorm_key: str
    device: str
    attention: str

    def __call__(self) -> OpenVLAPolicy:
        return OpenVLAPolicy(
            _snapshot(self.model_id, self.revision),
            unnorm_key=self.unnorm_key,
            device=self.device,
            attention=self.attention,
        )


def openvla(
    suite: str,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    unnorm_key: str | None = None,
    device: str = "cuda",
    attention: str = "sdpa",
) -> PolicySpec:
    """Pinned OpenVLA policy for one base LIBERO suite."""
    if suite not in OPENVLA_CHECKPOINTS:
        raise ValueError(f"OpenVLA requires one of {sorted(OPENVLA_CHECKPOINTS)}, got {suite!r}")
    default_model, default_revision, default_key = OPENVLA_CHECKPOINTS[suite]
    resolved_model = model_id or default_model
    resolved_revision = revision or (default_revision if resolved_model == default_model else None)
    if resolved_revision is None:
        raise ValueError("a custom OpenVLA model requires an immutable revision")
    resolved_key = unnorm_key or default_key
    if resolved_key != suite:
        raise ValueError(f"OpenVLA unnorm_key {resolved_key!r} does not match suite {suite!r}")
    return PolicySpec(
        factory=_OpenVLAFactory(
            resolved_model,
            resolved_revision,
            resolved_key,
            device,
            attention,
        ),
        policy_id=resolved_model,
        revision=resolved_revision,
        camera_height=256,
        camera_width=256,
        metadata={
            "adapter": "openvla",
            "unnorm_key": resolved_key,
            "attention": attention,
        },
    )


class VLAJEPAPolicy:
    """VLA-JEPA adapter through its in-process LeRobot implementation."""

    action_dim = ACTION_DIM
    control_mode = "relative"

    def __init__(self, model_path: str, *, device: str = "cuda") -> None:
        self.model_path = model_path
        self.device = device
        self.model: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self._load()

    def _load(self) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.vla_jepa.modeling_vla_jepa import (
            VLAJEPAPolicy as LeRobotVLAJEPA,
        )

        config = PreTrainedConfig.from_pretrained(self.model_path)
        config.device = self.device
        # The checkpoint config names mutable upstream repositories. Resolve
        # them to pinned local snapshots before constructing either backbone.
        config.qwen_model_name = _snapshot(QWEN3_VL_MODEL_ID, QWEN3_VL_REVISION)
        config.jepa_encoder_name = _snapshot(VJEPA2_MODEL_ID, VJEPA2_REVISION)
        self.model = LeRobotVLAJEPA.from_pretrained(self.model_path, config=config).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.model.config,
            pretrained_path=self.model_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def reset(self, instruction: str) -> None:
        self.reset_batch([instruction])

    def reset_batch(self, instructions: Sequence[str]) -> None:
        self._instructions = list(instructions)
        self.model.reset()

    def act(self, observation: Observation) -> np.ndarray:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: Sequence[Observation]) -> np.ndarray:
        import torch

        wrists = [observation.get("wrist_image") for observation in observations]
        if any(wrist is None for wrist in wrists):
            raise ValueError("VLA-JEPA requires the LIBERO wrist camera")
        if len(observations) != len(self._instructions):
            raise ValueError(
                "VLA-JEPA observation batch does not match the reset instruction batch"
            )
        batch: dict[str, Any] = {
            "observation.images.image": torch.stack(
                [self._image(observation["image"]) for observation in observations]
            ),
            "observation.images.image2": torch.stack([self._image(wrist) for wrist in wrists]),
            "task": [
                observation.get("instruction") or instruction
                for observation, instruction in zip(observations, self._instructions, strict=True)
            ],
        }
        states = [observation.get("state") for observation in observations]
        if all(state is not None for state in states):
            batch["observation.state"] = torch.from_numpy(
                np.stack([np.asarray(state, np.float32) for state in states])
            )
        with torch.inference_mode():
            action = self.postprocessor(self.model.select_action(self.preprocessor(batch)))
        result = np.asarray(_to_numpy(action), dtype=np.float32)
        return result.reshape(len(observations), -1)[:, :ACTION_DIM].copy()

    @staticmethod
    def _image(image: Any):
        import torch

        array = np.ascontiguousarray(np.asarray(image))
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)

    def close(self) -> None:
        self.model = None
        self.preprocessor = None
        self.postprocessor = None


@dataclass(frozen=True)
class _VLAJEPAFactory:
    model_id: str
    revision: str
    device: str

    def __call__(self) -> VLAJEPAPolicy:
        return VLAJEPAPolicy(_snapshot(self.model_id, self.revision), device=self.device)


def vla_jepa(
    *,
    model_id: str = VLA_JEPA_MODEL_ID,
    revision: str = VLA_JEPA_REVISION,
    device: str = "cuda",
) -> PolicySpec:
    """Pinned VLA-JEPA policy using LeRobot in the rollout process."""
    return PolicySpec(
        factory=_VLAJEPAFactory(model_id, revision, device),
        policy_id=model_id,
        revision=revision,
        camera_height=224,
        camera_width=224,
        metadata={
            "adapter": "vla_jepa",
            "qwen3_vl": f"{QWEN3_VL_MODEL_ID}@{QWEN3_VL_REVISION}",
            "vjepa2": f"{VJEPA2_MODEL_ID}@{VJEPA2_REVISION}",
        },
    )


def vla_jepa_cutile(
    *,
    model_id: str = VLA_JEPA_MODEL_ID,
    revision: str = VLA_JEPA_REVISION,
    qwen_model_id: str = QWEN3_VL_MODEL_ID,
    qwen_revision: str = QWEN3_VL_REVISION,
    device_id: int = 0,
) -> PolicySpec:
    """Pinned action-only VLA-JEPA policy using daft-cuTile on one GPU."""

    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError("device_id must be a non-negative integer")
    from physical_ai_evals.cutile_vla_jepa import (
        CUTILE_DAFT_REVISION,
        CUTILE_VLA_STATIC_BATCH_SIZE,
        CutileVLAJEPAFactory,
    )

    return PolicySpec(
        factory=CutileVLAJEPAFactory(
            model_id,
            revision,
            qwen_model_id,
            qwen_revision,
            device_id,
            _snapshot,
        ),
        policy_id=model_id,
        revision=revision,
        camera_height=224,
        camera_width=224,
        metadata={
            "adapter": "vla_jepa_cutile",
            "engine": "daft_cutile",
            "engine_revision": CUTILE_DAFT_REVISION,
            "qwen3_vl": f"{qwen_model_id}@{qwen_revision}",
            "static_batch_size": CUTILE_VLA_STATIC_BATCH_SIZE,
            "action_horizon": 7,
            "noise_scheme": "episode-keyed-sha256-pcg64-v1",
        },
    )


__all__ = [
    "BatchPolicy",
    "Observation",
    "OpenVLAPolicy",
    "Policy",
    "PolicySpec",
    "VLAJEPAPolicy",
    "openvla",
    "vla_jepa",
    "vla_jepa_cutile",
]
