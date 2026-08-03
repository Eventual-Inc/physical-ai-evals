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
"""Run one OpenVLA episode on the first LIBERO Spatial task.

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
from typing import Any


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


# LIBERO reads this configuration while it imports, so create it first.
libero_config = configure_libero()

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass(frozen=True)
class OpenVLAEpisodeResult:
    device: str
    suite: str
    task_id: int
    task: str
    initial_state_id: int
    success: bool
    policy_steps: int


class OpenVLAEpisode:
    def __init__(self, model_id: str, model_revision: str) -> None:
        self._device, self._dtype = self._select_device()
        self._env: OffScreenRenderEnv | None = None
        self._task: Any | None = None
        self._initial_state: np.ndarray | None = None
        self._initial_state_id: int | None = None
        self._init_model(model_id, model_revision)

    @staticmethod
    def _select_device() -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _init_model(self, model_id: str, model_revision: str) -> None:
        """Initialize the Transformers model and processor."""
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

    def _init_libero_env(self, suite: str, task_id: int, initial_state_id: int) -> None:
        """Initialize one LIBERO task and initial state."""
        self.close()
        task_suite = benchmark.get_benchmark_dict()[suite]()
        self._task = task_suite.get_task(task_id)
        self._initial_state = task_suite.get_task_init_states(task_id)[initial_state_id]
        self._initial_state_id = initial_state_id

        bddl_file = (
            Path(get_libero_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=256,
            camera_widths=256,
        )
        self._env.seed(0)

    @staticmethod
    def _openvla_image(observation: dict[str, Any]) -> Image.Image:
        """Convert the agent-view image to OpenVLA's expected image."""
        pixels = np.asarray(observation["agentview_image"], dtype=np.uint8)[::-1, ::-1]
        image = Image.fromarray(np.ascontiguousarray(pixels)).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        side = round(224 * np.sqrt(0.9))
        offset = (224 - side) // 2
        image = image.crop((offset, offset, offset + side, offset + side))
        return image.resize((224, 224), Image.Resampling.BILINEAR)

    def _init_and_settle_env(self) -> dict[str, Any]:
        """Reset the task and let MuJoCo contacts settle."""
        if self._env is None or self._initial_state is None:
            raise RuntimeError("LIBERO environment is not initialized")

        self._env.reset()
        observation = self._env.set_init_state(self._initial_state)
        settle_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(10):
            observation, _, _, _ = self._env.step(settle_action)
        return observation

    def run_episode(
        self,
        suite: str,
        task_id: int,
        initial_state_id: int,
        max_steps: int = 220,
    ) -> OpenVLAEpisodeResult:
        """Run one OpenVLA episode on a LIBERO task and initial state."""
        self._init_libero_env(suite, task_id, initial_state_id)
        observation = self._init_and_settle_env()
        if self._env is None or self._task is None or self._initial_state_id is None:
            raise RuntimeError("LIBERO episode is not initialized")

        prompt = f"In: What action should the robot take to {self._task.language.lower()}?\nOut:"
        success = False
        policy_steps = 0

        for _ in range(max_steps):
            image = self._openvla_image(observation)
            inputs = self._processor(prompt, image).to(
                self._device,
                dtype=self._dtype,
            )

            with torch.inference_mode():
                action = self._model.predict_action(
                    **inputs,
                    unnorm_key=suite,
                    do_sample=False,
                )
            action = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
            action[-1] = -np.sign(2 * action[-1] - 1)

            observation, _, done, _ = self._env.step(action)
            policy_steps += 1

            if done:
                success = True
                break

        return OpenVLAEpisodeResult(
            device=self._device,
            suite=suite,
            task_id=task_id,
            task=self._task.language,
            initial_state_id=self._initial_state_id,
            success=success,
            policy_steps=policy_steps,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._task = None
        self._initial_state = None
        self._initial_state_id = None


def run_openvla_on_libero(
    model_id: str,
    model_revision: str,
    suite: str,
    task_id: int,
    initial_state_id: int,
) -> OpenVLAEpisodeResult:
    """Load OpenVLA and run one LIBERO episode."""
    episode_runner = OpenVLAEpisode(model_id, model_revision)
    try:
        return episode_runner.run_episode(suite, task_id, initial_state_id)
    finally:
        episode_runner.close()


if __name__ == "__main__":
    MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
    MODEL_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
    SUITE = "libero_spatial"
    TASK_ID = 0
    INITIAL_STATE_ID = 0

    np.random.seed(7)
    torch.manual_seed(7)

    try:
        result = run_openvla_on_libero(
            MODEL_ID,
            MODEL_REVISION,
            SUITE,
            TASK_ID,
            INITIAL_STATE_ID,
        )
        print(asdict(result))
    finally:
        libero_config.cleanup()
