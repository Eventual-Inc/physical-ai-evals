# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "bddl==1.0.1",
#   "cloudpickle",
#   "daft==0.7.21",
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
"""Run dynamically sized VLA-JEPA rollouts on LIBERO-Pro with Daft.

Daft materializes episode rows through one stateful rollout worker. The worker
keeps each live simulator and its action queue, then asks the policy for a
batch only when those queues are empty. Finished episodes never enter another
inference batch.

On a Linux CUDA host or Apple Silicon Mac:

    uv run examples/vla_jepa_libero_pro_daft.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, TypedDict

import daft

ACTION_DIM = 7
ACTION_HORIZON = 7
INFERENCE_BATCH_SIZE = 16
ROLLOUT_BATCH_SIZE = 64
STATE_DIM = 8


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
    config = tempfile.TemporaryDirectory(prefix="vla-jepa-libero-pro-daft-")
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


# LIBERO reads this configuration during import.
libero_config = configure_libero()

import imageio.v3 as iio
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from libero.libero.envs import OffScreenRenderEnv


class LiberoObservation(TypedDict):
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class EpisodeSpec:
    episode_key: str
    repo_id: str
    repo_revision: str
    suite_variant: str
    perturbation: str
    task: str
    initial_state_id: int
    max_steps: int
    environment_seed: int
    video_path: str | None = None


@dataclass(frozen=True)
class EpisodeResult:
    device: str
    episode_key: str
    suite_variant: str
    perturbation: str
    task: str
    instruction: str
    initial_state_id: int
    success: bool
    control_steps: int


@dataclass
class _Lane:
    spec: Mapping[str, Any]
    environment: Any
    observation: LiberoObservation
    actions: deque[np.ndarray]
    frames: list[np.ndarray] | None
    success: bool = False
    control_steps: int = 0
    complete: bool = False


_ACTION_CHUNK_DTYPE = daft.DataType.tensor(
    daft.DataType.float32(),
    shape=(ACTION_HORIZON, ACTION_DIM),
)

_EPISODE_RESULT_DTYPE = daft.DataType.struct(
    {
        "device": daft.DataType.string(),
        "episode_key": daft.DataType.string(),
        "suite_variant": daft.DataType.string(),
        "perturbation": daft.DataType.string(),
        "task": daft.DataType.string(),
        "instruction": daft.DataType.string(),
        "initial_state_id": daft.DataType.int64(),
        "success": daft.DataType.bool(),
        "control_steps": daft.DataType.int64(),
    }
)

_EPISODE_SPEC_COLUMNS = tuple(EpisodeSpec.__dataclass_fields__)


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
    """Convert an xyzw quaternion the way LeRobot's LIBERO processor does.

    The angle is 2 * acos(w) with no sign canonicalization, so it stays
    continuous near pi. SciPy's as_rotvec() flips to w >= 0 and jumps by
    about 2 * pi when w changes sign, which LIBERO's downward-facing
    gripper does constantly.
    """
    x, y, z, w = np.asarray(quaternion, dtype=np.float32).reshape(4)
    w = np.clip(w, -1.0, 1.0)
    scale = np.sqrt(1.0 - w * w)
    if scale <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (np.array([x, y, z]) * (2.0 * np.arccos(w)) / scale).astype(np.float32)


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
        import torch

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
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"LIBERO action must have shape {(ACTION_DIM,)}, got {action.shape}")
        observation, _, done, _ = self._environment.step(action)
        return self._observation(observation), bool(done)

    def close(self) -> None:
        self._environment.close()


def _series_values(value: daft.Series | Sequence[Any] | np.ndarray) -> list[Any]:
    if isinstance(value, daft.Series):
        return value.to_pylist()
    if isinstance(value, np.ndarray):
        return list(value)
    return list(value)


@daft.cls(gpus=0.9, use_process=False, max_concurrency=1)
class VLAJEPAPolicy:
    """Pinned VLA-JEPA with one explicit host-to-device inference boundary."""

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        qwen_model_id: str,
        qwen_revision: str,
        vjepa2_model_id: str,
        vjepa2_revision: str,
    ) -> None:
        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            raise RuntimeError("VLA-JEPA requires a CUDA GPU or Apple Silicon MPS")

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.vla_jepa.modeling_vla_jepa import (
            VLAJEPAPolicy as LeRobotVLAJEPA,
        )

        model_path = snapshot_download(model_id, revision=model_revision)
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = self.device
        config.qwen_model_name = snapshot_download(qwen_model_id, revision=qwen_revision)
        config.jepa_encoder_name = snapshot_download(vjepa2_model_id, revision=vjepa2_revision)

        self._model = LeRobotVLAJEPA.from_pretrained(model_path, config=config).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._model.config,
            pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def device_name(self) -> str:
        return self.device

    def inference(
        self,
        images: np.ndarray,
        telemetry: np.ndarray,
        instructions: list[str],
    ) -> Any:
        """Cross H2D once, run the model, and return postprocessed action chunks."""
        import torch

        batch = {
            "observation.images.image": (
                torch.from_numpy(images[:, 0]).permute(0, 3, 1, 2).float().div_(255)
            ),
            "observation.images.image2": (
                torch.from_numpy(images[:, 1]).permute(0, 3, 1, 2).float().div_(255)
            ),
            "observation.state": torch.from_numpy(telemetry),
            "task": instructions,
        }
        with torch.inference_mode():
            return self._postprocessor(self._model.predict_action_chunk(self._preprocessor(batch)))

    @daft.method.batch(
        return_dtype=_ACTION_CHUNK_DTYPE,
        batch_size=INFERENCE_BATCH_SIZE,
    )
    def predict(
        self,
        images: daft.Series | Sequence[np.ndarray] | np.ndarray,
        telemetry: daft.Series | Sequence[np.ndarray] | np.ndarray,
        instructions: daft.Series | Sequence[str],
    ) -> np.ndarray:
        """Validate host observations around the concrete inference call."""
        image_batch = np.ascontiguousarray(np.stack(_series_values(images)))
        state_batch = np.ascontiguousarray(
            np.stack(_series_values(telemetry)),
            dtype=np.float32,
        )
        instruction_batch = [str(value) for value in _series_values(instructions)]

        batch_size = len(instruction_batch)
        if image_batch.dtype != np.uint8 or image_batch.shape != (
            batch_size,
            2,
            224,
            224,
            3,
        ):
            raise ValueError(
                f"images must have dtype uint8 and shape {(batch_size, 2, 224, 224, 3)!r}"
            )
        if state_batch.shape != (batch_size, STATE_DIM):
            raise ValueError(f"telemetry must have shape {(batch_size, STATE_DIM)!r}")

        chunks = np.asarray(
            self.inference(image_batch, state_batch, instruction_batch).detach().cpu(),
            dtype=np.float32,
        )
        expected = (batch_size, ACTION_HORIZON, ACTION_DIM)
        if chunks.shape != expected:
            raise RuntimeError(
                f"VLA-JEPA action chunks must have shape {expected!r}, got {chunks.shape!r}"
            )
        return np.ascontiguousarray(np.clip(chunks, -1, 1))


def _policy_device(policy: Any) -> str:
    device = getattr(policy, "device", None)
    if isinstance(device, str):
        return device
    return str(policy.device_name())


def _run_rollouts(
    policy: Any,
    environments: Sequence[Any],
    specs: Sequence[Mapping[str, Any]],
) -> list[EpisodeResult]:
    """Run independent lanes while batching only observations that need chunks."""
    if len(environments) != len(specs):
        raise ValueError("environments and episode specs must have the same length")

    lanes = []
    for environment, spec in zip(environments, specs, strict=True):
        observation = environment.reset()
        frames = [observation["image"]] if spec.get("video_path") else None
        lanes.append(
            _Lane(
                spec=spec,
                environment=environment,
                observation=observation,
                actions=deque(),
                frames=frames,
                complete=int(spec["max_steps"]) <= 0,
            )
        )

    while any(not lane.complete for lane in lanes):
        ready = [lane for lane in lanes if not lane.complete and not lane.actions]
        for offset in range(0, len(ready), INFERENCE_BATCH_SIZE):
            batch = ready[offset : offset + INFERENCE_BATCH_SIZE]
            images = np.stack(
                [
                    np.stack((lane.observation["image"], lane.observation["wrist_image"]))
                    for lane in batch
                ]
            )
            telemetry = np.stack([lane.observation["state"] for lane in batch])
            instructions = [lane.environment.instruction for lane in batch]
            chunks = np.asarray(
                policy.predict(images, telemetry, instructions),
                dtype=np.float32,
            )
            expected = (len(batch), ACTION_HORIZON, ACTION_DIM)
            if chunks.shape != expected:
                raise RuntimeError(
                    f"policy returned action chunks with shape {chunks.shape}, expected {expected}"
                )
            for lane, chunk in zip(batch, chunks, strict=True):
                lane.actions.extend(chunk)

        for lane in lanes:
            if lane.complete:
                continue
            observation, done = lane.environment.step(lane.actions.popleft())
            lane.observation = observation
            lane.control_steps += 1
            if lane.frames is not None:
                lane.frames.append(observation["image"])
            if done:
                lane.success = True
                lane.complete = True
            elif lane.control_steps >= int(lane.spec["max_steps"]):
                lane.complete = True

    device = _policy_device(policy)
    results = []
    for lane in lanes:
        video_path = lane.spec.get("video_path")
        if lane.frames is not None and video_path:
            path = Path(str(video_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, np.stack(lane.frames), fps=20)
        results.append(
            EpisodeResult(
                device=device,
                episode_key=str(lane.spec["episode_key"]),
                suite_variant=lane.environment.suite_variant,
                perturbation=lane.environment.perturbation,
                task=lane.environment.task,
                instruction=lane.environment.instruction,
                initial_state_id=lane.environment.initial_state_id,
                success=lane.success,
                control_steps=lane.control_steps,
            )
        )
    return results


@daft.cls(
    cpus=1,
    gpus=0.9,
    use_process=False,
    max_concurrency=1,
    name_override="LIBERO-Pro rollouts",
)
class RolloutOrchestrator:
    """One stateful worker containing the live rollout feedback loop."""

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        qwen_model_id: str,
        qwen_revision: str,
        vjepa2_model_id: str,
        vjepa2_revision: str,
    ) -> None:
        self.policy = VLAJEPAPolicy(
            model_id,
            model_revision,
            qwen_model_id,
            qwen_revision,
            vjepa2_model_id,
            vjepa2_revision,
        )

    @daft.method.batch(
        return_dtype=_EPISODE_RESULT_DTYPE,
        batch_size=ROLLOUT_BATCH_SIZE,
    )
    def rollout(
        self,
        episode_specs: daft.Series | Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        specs = _series_values(episode_specs)
        environments = [
            LiberoProEpisode(
                repo_id=str(spec["repo_id"]),
                repo_revision=str(spec["repo_revision"]),
                suite_variant=str(spec["suite_variant"]),
                perturbation=str(spec["perturbation"]),
                task=str(spec["task"]),
                initial_state_id=int(spec["initial_state_id"]),
                environment_seed=int(spec["environment_seed"]),
            )
            for spec in specs
        ]
        try:
            return [asdict(result) for result in _run_rollouts(self.policy, environments, specs)]
        finally:
            for environment in environments:
                environment.close()


def run_vla_jepa_rollouts(
    *,
    model_id: str,
    model_revision: str,
    qwen_model_id: str,
    qwen_revision: str,
    vjepa2_model_id: str,
    vjepa2_revision: str,
    episodes: Sequence[EpisodeSpec],
) -> list[EpisodeResult]:
    """Map episode rows to the rollout worker and materialize once."""
    if not episodes:
        return []

    from daft import col
    from daft.functions import to_struct

    orchestrator = RolloutOrchestrator(
        model_id,
        model_revision,
        qwen_model_id,
        qwen_revision,
        vjepa2_model_id,
        vjepa2_revision,
    )
    planned = daft.from_pylist([asdict(episode) for episode in episodes]).into_batches(
        ROLLOUT_BATCH_SIZE
    )
    completed = planned.with_column(
        "result",
        orchestrator.rollout(to_struct(*(col(name) for name in _EPISODE_SPEC_COLUMNS))),
    ).select("result")
    return [EpisodeResult(**row) for row in completed.to_pydict()["result"]]


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
    """Preserve the original one-episode entry point through the Daft plan."""
    return run_vla_jepa_rollouts(
        model_id=model_id,
        model_revision=model_revision,
        qwen_model_id=qwen_model_id,
        qwen_revision=qwen_revision,
        vjepa2_model_id=vjepa2_model_id,
        vjepa2_revision=vjepa2_revision,
        episodes=[
            EpisodeSpec(
                episode_key=f"{suite_variant}/{task}/{initial_state_id}",
                repo_id=repo_id,
                repo_revision=repo_revision,
                suite_variant=suite_variant,
                perturbation=perturbation,
                task=task,
                initial_state_id=initial_state_id,
                max_steps=max_steps,
                environment_seed=environment_seed,
                video_path=str(video_path) if video_path is not None else None,
            )
        ],
    )[0]


if __name__ == "__main__":
    import torch

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
            max_steps=280,  # LeRobot's libero_spatial budget
            environment_seed=7,
            video_path=os.environ.get(
                "VLA_JEPA_LIBERO_PRO_VIDEO",
                "vla_jepa_libero_pro.mp4",
            ),
        )
        print(asdict(result))
    finally:
        libero_config.cleanup()
