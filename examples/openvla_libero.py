# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate>=0.25.0",
#   "bddl==1.0.1",
#   "cloudpickle",
#   "easydict==1.9",
#   "einops",
#   "future",
#   "gym==0.25.2",
#   "imageio[ffmpeg]>=2.34",
#   "json-numpy",
#   "libero==0.1.1",
#   "matplotlib",
#   "mujoco==3.9.0",
#   "numpy==1.26.4",
#   "opencv-python==4.9.0.80",
#   "pillow>=10.0",
#   "robosuite==1.4.1",
#   "scipy==1.15.3",
#   "timm==0.9.10",
#   "tokenizers==0.19.1",
#   "torch==2.2.0",
#   "torchvision==0.17.0",
#   "transformers==4.40.1",
# ]
# [tool.uv]
# override-dependencies = [
#   "hf-egl-probe>=1.0.1; sys_platform == 'linux'",
#   "robomimic==0.2.0; sys_platform == 'linux'",
#   "robosuite==1.4.1",
# ]
# ///
# ruff: noqa: E402
"""Run one OpenVLA episode with separate policy and LIBERO environment objects.

Run on a Linux CUDA host or Apple Silicon Mac:

    uv run examples/openvla_libero.py
"""

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, TypedDict


def configure_libero() -> tempfile.TemporaryDirectory[str]:
    """Configure LIBERO paths and offscreen rendering."""
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    elif sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "cgl")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    libero_root = Path(str(distribution("libero").locate_file("libero/libero")))
    config = tempfile.TemporaryDirectory(prefix="openvla-libero-")
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


# ------------------------------------------------------------
# LIBERO reads this configuration while it imports, so create it first.
libero_config = configure_libero()

import imageio.v3 as iio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class LiberoObservation(TypedDict):
    image: np.ndarray
    wrist_image: np.ndarray | None
    robot_state: dict[str, np.ndarray]


@dataclass(frozen=True)
class EpisodeResult:
    device: str
    suite: str
    task_id: int
    task: str
    initial_state_id: int
    success: bool
    policy_steps: int


class LiberoEpisode:
    """One concrete LIBERO task, initial state, and MuJoCo environment."""

    def __init__(self, suite: str, task_id: int, initial_state_id: int) -> None:
        self._suite = suite
        self._task_id = task_id
        self._initial_state_id = initial_state_id

        task_suite = benchmark.get_benchmark_dict()[suite]()
        self._task = task_suite.get_task(task_id)
        self._initial_state = task_suite.get_task_init_states(task_id)[initial_state_id]

        bddl_file = (
            Path(get_libero_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=256,
            camera_widths=256,
        )
        self._env.seed(0)

    @property
    def suite(self) -> str:
        return self._suite

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def task(self) -> str:
        return self._task.language

    @property
    def instruction(self) -> str:
        return self._task.language

    @property
    def initial_state_id(self) -> int:
        return self._initial_state_id

    @staticmethod
    def _observation(raw: dict[str, Any]) -> LiberoObservation:
        def derotate(name: str) -> np.ndarray | None:
            value = raw.get(name)
            if value is None:
                return None
            return np.ascontiguousarray(np.asarray(value, dtype=np.uint8)[::-1, ::-1])

        image = derotate("agentview_image")
        if image is None:
            raise ValueError("LIBERO observation has no agent-view image")
        return {
            "image": image,
            "wrist_image": derotate("robot0_eye_in_hand_image"),
            "robot_state": {
                name: np.asarray(raw[name])
                for name in (
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                )
                if name in raw
            },
        }

    def reset(self) -> LiberoObservation:
        """Restore the initial state and let MuJoCo contacts settle."""
        self._env.reset()
        observation = self._env.set_init_state(self._initial_state)
        settle_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(10):
            observation, _, _, _ = self._env.step(settle_action)
        return self._observation(observation)

    def step(self, action: np.ndarray) -> tuple[LiberoObservation, bool]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"LIBERO action must have shape (7,), got {action.shape}")
        observation, _, done, _ = self._env.step(action)
        return self._observation(observation), bool(done)

    def close(self) -> None:
        self._env.close()


class OpenVLAPolicy:
    """OpenVLA inference and conversion to LIBERO's seven-value action."""

    def __init__(self, model_id: str, model_revision: str, unnorm_key: str) -> None:
        self._device, self._dtype = self._select_device()
        self._unnorm_key = unnorm_key
        self._prompt: str | None = None
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=True,
        )
        self._model = (
            AutoModelForVision2Seq.from_pretrained(
                model_id,
                revision=model_revision,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self._device)
            .eval()
        )

    @property
    def device(self) -> str:
        return self._device

    @staticmethod
    def _select_device() -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    @staticmethod
    def _image(pixels: np.ndarray) -> Image.Image:
        image = Image.fromarray(pixels).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        side = round(224 * np.sqrt(0.9))
        offset = (224 - side) // 2
        image = image.crop((offset, offset, offset + side, offset + side))
        return image.resize((224, 224), Image.Resampling.BILINEAR)

    def reset(self, instruction: str) -> None:
        self._prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def act(self, observation: LiberoObservation) -> np.ndarray:
        if self._prompt is None:
            raise RuntimeError("policy must be reset with an instruction")
        inputs = self._processor(self._prompt, self._image(observation["image"])).to(
            self._device,
            dtype=self._dtype,
        )
        with torch.inference_mode():
            action = self._model.predict_action(
                **inputs,
                unnorm_key=self._unnorm_key,
                do_sample=False,
            )

        action = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        action[-1] = -np.sign(2 * action[-1] - 1)
        return action


def run_episode(
    policy: OpenVLAPolicy,
    environment: LiberoEpisode,
    max_steps: int = 220,
    video_path: str | Path | None = None,
) -> EpisodeResult:
    """Run the policy/environment control loop."""
    observation = environment.reset()
    frames = [observation["image"]] if video_path is not None else None
    policy.reset(environment.instruction)
    success = False
    policy_steps = 0

    for _ in range(max_steps):
        observation, done = environment.step(policy.act(observation))
        if frames is not None:
            frames.append(observation["image"])
        policy_steps += 1
        if done:
            success = True
            break

    if frames is not None and video_path is not None:
        iio.imwrite(video_path, np.stack(frames), fps=20)

    return EpisodeResult(
        device=policy.device,
        suite=environment.suite,
        task_id=environment.task_id,
        task=environment.task,
        initial_state_id=environment.initial_state_id,
        success=success,
        policy_steps=policy_steps,
    )


def run_openvla_on_libero(
    model_id: str,
    model_revision: str,
    suite: str,
    task_id: int,
    initial_state_id: int,
    video_path: str | Path | None = None,
) -> EpisodeResult:
    """Construct the concrete policy and environment, then run one episode."""
    policy = OpenVLAPolicy(model_id, model_revision, unnorm_key=suite)
    environment = LiberoEpisode(suite, task_id, initial_state_id)
    try:
        return run_episode(policy, environment, video_path=video_path)
    finally:
        environment.close()


if __name__ == "__main__":
    MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
    MODEL_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
    SUITE = "libero_spatial"
    TASK_ID = 0
    INITIAL_STATE_ID = 0
    VIDEO_PATH = os.environ.get("OPENVLA_LIBERO_VIDEO", "openvla_libero.mp4")

    np.random.seed(7)
    torch.manual_seed(7)

    try:
        result = run_openvla_on_libero(
            MODEL_ID,
            MODEL_REVISION,
            SUITE,
            TASK_ID,
            INITIAL_STATE_ID,
            video_path=VIDEO_PATH,
        )
        print(asdict(result))
    finally:
        libero_config.cleanup()
