# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "bddl==1.0.1",
#   "cloudpickle",
#   "easydict==1.9",
#   "einops",
#   "future",
#   "gym==0.25.2",
#   "gymnasium>=0.29.0",
#   "hf-xet==1.5.2",
#   "huggingface-hub==1.16.4",
#   "imageio[ffmpeg]>=2.34",
#   "lerobot[vla_jepa] @ git+https://github.com/huggingface/lerobot@052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34",
#   "libero==0.1.1",
#   "matplotlib",
#   "mujoco==3.9.0",
#   "numba>=0.49.1",
#   "numpy==2.2.6",
#   "pillow>=10.0",
#   "robosuite==1.4.1",
#   "scipy==1.15.3",
#   "termcolor",
#   "torch==2.11.0",
#   "torchvision==0.26.0",
#   "transformers==5.5.4",
# ]
# [tool.uv]
# override-dependencies = [
#   "hf-egl-probe>=1.0.1; sys_platform == 'linux'",
#   "robomimic==0.2.0; sys_platform == 'linux'",
#   "robosuite==1.4.1",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
#   { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
# ]
# torchvision = [
#   { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
#   { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
# ]
# ///
# ruff: noqa: E402
"""Run VLA-JEPA on one pinned LIBERO-Pro task with Python 3.12.

On a Linux CUDA host or Apple Silicon Mac:

    uv run examples/vla_jepa_libero_pro.py

For a ready-made CUDA host, run examples/vla_jepa_libero_pro_modal.py.
"""

import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, TypedDict


def configure_libero() -> tempfile.TemporaryDirectory[str]:
    """Point LIBERO at its installed assets before importing the package."""
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    elif sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "cgl")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    else:
        raise RuntimeError(f"LIBERO is not configured for {sys.platform}")

    libero_root = Path(str(distribution("libero").locate_file("libero/libero")))
    config = tempfile.TemporaryDirectory(prefix="vla-jepa-libero-pro-")
    os.environ["LIBERO_CONFIG_PATH"] = config.name
    Path(config.name, "config.yaml").write_text(
        json.dumps(
            {
                "benchmark_root": str(libero_root),
                "bddl_files": str(libero_root / "bddl_files"),
                "init_states": str(libero_root / "init_files"),
                "datasets": str(libero_root.parent / "datasets"),
                "assets": str(libero_root / "assets"),
            }
        ),
        encoding="utf-8",
    )
    return config


# ----------------------------------------------------------------------------
# LIBERO reads this configuration during import.
libero_config = configure_libero()

import imageio.v3 as iio
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation


class LiberoObservation(TypedDict):
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class EpisodeResult:
    device: str
    suite_variant: str
    perturbation: str
    task: str
    instruction: str
    initial_state_id: int
    success: bool
    control_steps: int


def download_task_file(repo_id: str, repo_revision: str, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=repo_revision,
            filename=filename,
        )
    )


def read_instruction(bddl_path: Path) -> str:
    match = re.search(r"\(:language\s+(.+?)\s*\)", bddl_path.read_text(), re.DOTALL)
    if match is None:
        raise ValueError(f"LIBERO-Pro task has no language instruction: {bddl_path}")
    return " ".join(match.group(1).split())


def quaternion_to_axis_angle(quaternion: Any) -> np.ndarray:
    return (
        Rotation.from_quat(np.asarray(quaternion, dtype=np.float32).reshape(4))
        .as_rotvec()
        .astype(np.float32)
    )


class LiberoProEpisode:
    """One concrete LIBERO-Pro task and initial state."""

    def __init__(
        self,
        repo_id: str,
        repo_revision: str,
        suite_variant: str,
        perturbation: str,
        task: str,
        initial_state_id: int,
        environment_seed: int,
    ) -> None:
        self.suite_variant = suite_variant
        self.perturbation = perturbation
        self.task = task
        self.initial_state_id = initial_state_id

        bddl_path = download_task_file(
            repo_id,
            repo_revision,
            f"bddl_files/{suite_variant}/{task}.bddl",
        )
        init_path = download_task_file(
            repo_id,
            repo_revision,
            f"init_files/{suite_variant}/{task}.pruned_init",
        )
        self.instruction = read_instruction(bddl_path)
        initial_states = torch.load(init_path, map_location="cpu", weights_only=False)
        if not 0 <= initial_state_id < len(initial_states):
            raise IndexError(
                f"initial_state_id must be in [0, {len(initial_states)}), got {initial_state_id}"
            )
        self._initial_state = initial_states[initial_state_id]
        self._environment = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=224,
            camera_widths=224,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )
        self._environment.seed(environment_seed)

    @staticmethod
    def _observation(raw: dict[str, Any]) -> LiberoObservation:
        return {
            "image": np.ascontiguousarray(
                np.asarray(raw["agentview_image"], dtype=np.uint8)[::-1, ::-1]
            ),
            "wrist_image": np.ascontiguousarray(
                np.asarray(raw["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1, ::-1]
            ),
            "state": np.concatenate(
                (
                    np.asarray(raw["robot0_eef_pos"], dtype=np.float32).ravel()[:3],
                    quaternion_to_axis_angle(raw["robot0_eef_quat"]),
                    np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).ravel()[:2],
                )
            ),
        }

    def reset(self) -> LiberoObservation:
        """Restore the initial state and let MuJoCo contacts settle."""
        self._environment.reset()
        observation = self._environment.set_init_state(self._initial_state)
        settle_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(10):
            observation, _, _, _ = self._environment.step(settle_action)
        return self._observation(observation)

    def step(self, action: np.ndarray) -> tuple[LiberoObservation, bool]:
        action = np.asarray(action, np.float32)
        if action.shape != (7,):
            raise ValueError(f"LIBERO action must have shape (7,), got {action.shape}")
        observation, _, done, _ = self._environment.step(action)
        return self._observation(observation), bool(done)

    def close(self) -> None:
        self._environment.close()


class VLAJEPAPolicy:
    """Pinned VLA-JEPA inference through its native LeRobot implementation."""

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        qwen_model_id: str,
        qwen_revision: str,
        vjepa2_model_id: str,
        vjepa2_revision: str,
    ) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            raise RuntimeError("VLA-JEPA requires a CUDA GPU or Apple Silicon MPS")
        self._instruction = ""

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.vla_jepa.modeling_vla_jepa import (
            VLAJEPAPolicy as LeRobotVLAJEPA,
        )

        # Download and configure the VLA-JEPA model.
        model_path = snapshot_download(model_id, revision=model_revision)
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = self.device
        config.qwen_model_name = snapshot_download(qwen_model_id, revision=qwen_revision)
        config.jepa_encoder_name = snapshot_download(vjepa2_model_id, revision=vjepa2_revision)

        # Instantiate the model, preprocessor, and postprocessor.
        self._model = LeRobotVLAJEPA.from_pretrained(model_path, config=config).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._model.config,
            pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    @staticmethod
    def _image(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255)

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._model.reset()

    def act(self, observation: LiberoObservation) -> np.ndarray:
        if not self._instruction:
            raise RuntimeError("policy must be reset with an instruction")
        batch = {
            "observation.images.image": self._image(observation["image"]).unsqueeze(0),
            "observation.images.image2": self._image(observation["wrist_image"]).unsqueeze(0),
            "observation.state": torch.from_numpy(observation["state"][None]),
            "task": [self._instruction],
        }
        with torch.inference_mode():
            action = self._postprocessor(      # Postprocess action
                self._model.select_action(     # Predict action
                    self._preprocessor(batch)  # Preprocess observations
                )
            )
        action = np.asarray(action.detach().cpu(), np.float32).reshape(-1)[:7].copy()
        return np.clip(action, -1, 1)


def run_episode(
    policy: VLAJEPAPolicy,
    environment: LiberoProEpisode,
    max_steps: int,
    video_path: str | Path | None = None,
) -> EpisodeResult:
    """Run one control tick per action; LeRobot manages its seven-action queue."""
    observation = environment.reset()
    policy.reset(environment.instruction)
    frames = [observation["image"]] if video_path is not None else None
    success = False
    control_steps = 0

    for _ in range(max_steps):
        observation, done = environment.step(policy.act(observation))
        if frames is not None:
            frames.append(observation["image"])
        control_steps += 1
        if done:
            success = True
            break

    if frames is not None and video_path is not None:
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(video_path, np.stack(frames), fps=20)

    return EpisodeResult(
        device=policy.device,
        suite_variant=environment.suite_variant,
        perturbation=environment.perturbation,
        task=environment.task,
        instruction=environment.instruction,
        initial_state_id=environment.initial_state_id,
        success=success,
        control_steps=control_steps,
    )


def run_vla_jepa_on_libero_pro(
    model_id: str,
    model_revision: str,
    qwen_model_id: str,
    qwen_revision: str,
    vjepa2_model_id: str,
    vjepa2_revision: str,
    repo_id: str,
    repo_revision: str,
    suite_variant: str,
    perturbation: str,
    task: str,
    initial_state_id: int,
    max_steps: int,
    environment_seed: int,
    video_path: str | Path | None = None,
) -> EpisodeResult:
    """Construct the concrete policy and environment, then run one episode."""

    environment = LiberoProEpisode(
        repo_id,
        repo_revision,
        suite_variant,
        perturbation,
        task,
        initial_state_id,
        environment_seed,
    )
    try:
        policy = VLAJEPAPolicy(
            model_id,
            model_revision,
            qwen_model_id,
            qwen_revision,
            vjepa2_model_id,
            vjepa2_revision,
        )
        return run_episode(policy, environment, max_steps, video_path)
    finally:
        environment.close()


if __name__ == "__main__":
    # Set the random seed for reproducibility.
    np.random.seed(7)
    torch.manual_seed(7)

    try:
        result = run_vla_jepa_on_libero_pro(
            model_id="lerobot/VLA-JEPA-LIBERO",
            model_revision="735d9f692981e286ade093b5046627eda876e5d0",
            qwen_model_id="Qwen/Qwen3-VL-2B-Instruct",
            qwen_revision="89644892e4d85e24eaac8bacfd4f463576704203",
            vjepa2_model_id="facebook/vjepa2-vitl-fpc64-256",
            vjepa2_revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
            repo_id="zhouxueyang/LIBERO-Pro",
            repo_revision="c86fc3b8293185a6f373677018ff3e37f8391602",
            suite_variant="libero_spatial_lan",
            perturbation="lan",
            task="pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
            initial_state_id=0,
            max_steps=250,
            environment_seed=7,
            video_path=os.environ.get(
                "VLA_JEPA_LIBERO_PRO_VIDEO",
                "vla_jepa_libero_pro.mp4",
            ),
        )
        print(asdict(result))
    finally:
        libero_config.cleanup()
