"""One Modal app for both policy stacks and all three LIBERO benchmarks.

The orchestration is shared. OpenVLA and VLA-JEPA intentionally retain
separate images because their Torch, NumPy, and Transformers constraints
conflict.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import modal

from physical_ai_evals.libero import (
    LIBERO_PRO_CODE_REPOSITORY,
    LIBERO_PRO_CODE_REVISION,
)
from physical_ai_evals.policy import (
    QWEN3_VL_MODEL_ID,
    QWEN3_VL_REVISION,
    VJEPA2_MODEL_ID,
    VJEPA2_REVISION,
    VLA_JEPA_MODEL_ID,
    VLA_JEPA_REVISION,
)

logger = logging.getLogger(__name__)

APP_DIR = "/workspace"
MODEL_CACHE_DIR = "/models"
OUTPUT_DIR = "/outputs"
GPU_TYPE = "A10G"
MODAL_REGION = ["us-west"]
CUDA_BASE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
PYTHON_VERSION = "3.12"
PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
LEROBOT_REVISION = "052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34"
LEROBOT_SPEC = f"lerobot[vla_jepa] @ git+https://github.com/huggingface/lerobot@{LEROBOT_REVISION}"

app = modal.App("physical-ai-evals")
MODEL_CACHE = modal.Volume.from_name("daft-model-cache", create_if_missing=True)
OUTPUTS = modal.Volume.from_name("daft-model-outputs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("HF_TOKEN")
VOLUMES = {MODEL_CACHE_DIR: MODEL_CACHE, OUTPUT_DIR: OUTPUTS}

_APT = (
    "git",
    "ffmpeg",
    "build-essential",
    "clang",
    "cmake",
    "linux-libc-dev",
    "libgl1",
    "libglib2.0-0",
    "libegl1",
    "libgles2",
    "libosmesa6",
    "libosmesa6-dev",
    "libsm6",
    "libxext6",
    "patchelf",
)
_SIM = (
    "robosuite==1.4.1",
    "bddl==1.0.1",
    "easydict==1.9",
    "cloudpickle",
    "gym==0.25.2",
    "imageio[ffmpeg]>=2.34",
    "matplotlib",
    "einops",
    "future",
    "mujoco==3.9.0",
    "scipy==1.15.3",
)


def _cache_environment() -> dict[str, str]:
    return {
        "HF_HOME": f"{MODEL_CACHE_DIR}/huggingface",
        "HF_HUB_CACHE": f"{MODEL_CACHE_DIR}/huggingface/hub",
        "TRANSFORMERS_CACHE": f"{MODEL_CACHE_DIR}/huggingface/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBERO_CONFIG_PATH": "/opt/libero-config",
        "PYTHONPATH": f"/opt/LIBERO-PRO:{APP_DIR}",
    }


def _with_libero_pro(image: modal.Image) -> modal.Image:
    benchmark_root = "/opt/LIBERO-PRO/libero/libero"
    return (
        image.run_commands(
            "git init /opt/LIBERO-PRO",
            f"git -C /opt/LIBERO-PRO remote add origin {LIBERO_PRO_CODE_REPOSITORY}",
            f"git -C /opt/LIBERO-PRO fetch --depth 1 origin {LIBERO_PRO_CODE_REVISION}",
            "git -C /opt/LIBERO-PRO checkout --detach FETCH_HEAD",
            "pip install --no-deps -e /opt/LIBERO-PRO",
            "mkdir -p /opt/libero-config",
            "printf 'benchmark_root: %s\\nbddl_files: %s\\ninit_states: %s\\n"
            "assets: %s\\ndatasets: %s\\n' "
            f"{benchmark_root} {benchmark_root}/bddl_files "
            f"{benchmark_root}/init_files {benchmark_root}/assets "
            f"/opt/LIBERO-PRO/libero/datasets "
            "> /opt/libero-config/config.yaml",
        )
        .env(_cache_environment())
        .add_local_python_source("physical_ai_evals")
    )


def openvla_image() -> modal.Image:
    """OpenVLA's frozen Transformers 4.40 / Torch 2.2 environment."""
    image = (
        modal.Image.from_registry(CUDA_BASE, add_python=PYTHON_VERSION)
        .apt_install(*_APT)
        .pip_install(
            "torch==2.2.0",
            "torchvision==0.17.0",
            "torchaudio==2.2.0",
            "numpy==1.26.4",
            "transformers==4.40.1",
            "tokenizers==0.19.1",
            "timm==0.9.10",
            "accelerate>=0.25.0",
            "json-numpy",
            "pillow>=10",
            "opencv-python==4.9.0.80",
            "daft[huggingface,video]==0.7.21",
            "huggingface_hub==0.36.2",
            "hf_xet==1.5.2",
            *_SIM,
        )
    )
    return _with_libero_pro(image)


def vla_jepa_image() -> modal.Image:
    """VLA-JEPA's LeRobot / Transformers 5.x environment."""
    image = (
        modal.Image.from_registry(CUDA_BASE, add_python=PYTHON_VERSION)
        .apt_install(*_APT)
        # Match the pinned LeRobot revision's Linux lock, including its CUDA
        # wheel index; PyPI otherwise resolves the newer CUDA 13 variant.
        .pip_install(
            "numpy==2.2.6",
            "torch==2.11.0+cu128",
            "torchvision==0.26.0+cu128",
            extra_index_url=PYTORCH_CU128_INDEX,
        )
        .pip_install(
            LEROBOT_SPEC,
            "numpy==2.2.6",
            "transformers==5.5.4",
            "daft[huggingface,video]==0.7.21",
            "huggingface_hub==1.16.4",
            "hf_xet==1.5.2",
            *_SIM,
        )
    )
    return _with_libero_pro(image)


# Construct each dependency graph once. Repeating the builders in decorators makes
# Modal treat otherwise identical graphs as distinct image builds during app startup.
OPENVLA_IMAGE = openvla_image()
VLA_JEPA_IMAGE = vla_jepa_image()


def _function_options(
    image: modal.Image,
    *,
    gpu: str | None = None,
    cpu: float = 8,
    memory: int = 65536,
    timeout: int = 86400,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "image": image,
        "cpu": cpu,
        "memory": memory,
        "timeout": timeout,
        "region": MODAL_REGION,
        "volumes": VOLUMES,
        "secrets": [HF_SECRET],
        "enable_memory_snapshot": False,
    }
    if gpu is not None:
        options["gpu"] = gpu
    return options


def _benchmark(
    benchmark_name: str,
    suite: str,
    tasks: list[str] | None,
    perturbations: list[str] | None,
    episodes: int,
    seed: int,
):
    from physical_ai_evals import (
        libero,
        libero_para,
        libero_pro,
    )

    if benchmark_name == "libero":
        task_ids = [int(task) for task in tasks] if tasks else None
        return libero(
            suite,
            task_ids=task_ids,
            episodes=episodes,
            seed=seed,
        )
    if benchmark_name == "libero_para":
        task_ids = (
            [int(task) for task in tasks]
            if tasks and all(task.isdigit() for task in tasks)
            else None
        )
        task_keys = tasks if tasks and task_ids is None else None
        return libero_para(
            task_ids=task_ids,
            task_keys=task_keys,
            paraphrase_types=perturbations,
            episodes=episodes,
            seed=seed,
        )
    if benchmark_name == "libero_pro":
        return libero_pro(
            suite,
            task_keys=tasks,
            perturbations=perturbations,
            episodes=episodes,
            seed=seed,
        )
    raise ValueError("benchmark must be 'libero', 'libero_para', or 'libero_pro'")


def _smoke_benchmark(
    benchmark_name: str,
    suite: str,
    tasks: list[str] | None,
    perturbations: list[str] | None,
    env_batch_size: int,
) -> dict[str, Any]:
    """Construct, reset, and step a real MuJoCo subprocess vector."""
    import numpy as np

    from physical_ai_evals import (
        libero,
        libero_para,
        libero_pro,
    )

    logger.info("selecting %s rollouts", benchmark_name)
    if benchmark_name == "libero":
        task_ids = [int(task) for task in tasks] if tasks else [0]
        benchmark = libero(suite, task_ids=task_ids, episodes=env_batch_size)
    elif benchmark_name == "libero_para":
        task_ids = [int(task) for task in tasks] if tasks else [0]
        benchmark = libero_para(
            task_ids=task_ids,
            paraphrase_types=perturbations,
            episodes=env_batch_size,
        )
        benchmark = replace(
            benchmark,
            rollouts=benchmark.rollouts.sort(["task_key", "init_state_id"]).limit(
                env_batch_size
            ),
        )
    elif benchmark_name == "libero_pro":
        benchmark = libero_pro(
            suite,
            perturbations=perturbations or ["lan"],
            task_keys=tasks,
            episodes=env_batch_size,
        )
        benchmark = replace(
            benchmark,
            rollouts=benchmark.rollouts.sort(["task_key", "init_state_id"]).limit(
                env_batch_size
            ),
        )
    else:
        raise ValueError("unknown benchmark")

    rollouts = list(benchmark.rollouts.sort("init_state_id").iter_rows())
    logger.info("%d rollouts materialized; constructing runtime", len(rollouts))
    runtime = benchmark.runtime_factory(camera_height=64, camera_width=64)
    try:
        prepare_runtime = getattr(runtime, "prepare", None)
        if callable(prepare_runtime):
            prepare_runtime()
        logger.info("opening vector environment")
        environment, instructions, init_states, task_names = runtime.open_batch(rollouts)
        logger.info("resetting vector environment")
        environment.reset()
        observations = environment.set_init_state(init_states)
        logger.info("stepping vector environment")
        next_observations, rewards, dones, _ = environment.step(
            np.zeros((len(rollouts), 7), dtype=np.float32)
        )
        logger.info("vector environment step complete")
        observation = observations[0]
        next_observation = next_observations[0]
        return {
            "benchmark": benchmark_name,
            "suite": suite,
            "env_batch_size": len(rollouts),
            "task_name": task_names[0],
            "instruction": instructions[0],
            "primary_shape": list(observation["agentview_image"].shape),
            "next_primary_shape": list(next_observation["agentview_image"].shape),
            "rewards": np.asarray(rewards).tolist(),
            "dones": np.asarray(dones).tolist(),
            "libero_pro_revision": LIBERO_PRO_CODE_REVISION,
        }
    finally:
        logger.info("closing runtime")
        runtime.close()


@app.function(**_function_options(OPENVLA_IMAGE, cpu=4, memory=32768))
def smoke_openvla(
    benchmark_name: str = "libero",
    suite: str = "libero_spatial",
    tasks: list[str] | None = None,
    perturbations: list[str] | None = None,
    env_batch_size: int = 2,
) -> dict[str, Any]:
    logger.info("OpenVLA container entered")
    import numpy

    logger.info("NumPy imported")
    import torch

    logger.info("Torch imported")
    import transformers

    logger.info("Transformers imported")
    result = _smoke_benchmark(
        benchmark_name,
        suite,
        tasks,
        perturbations,
        env_batch_size,
    )
    return {
        **result,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


@app.function(**_function_options(VLA_JEPA_IMAGE, cpu=4, memory=32768))
def smoke_vla_jepa(
    benchmark_name: str = "libero",
    suite: str = "libero_spatial",
    tasks: list[str] | None = None,
    perturbations: list[str] | None = None,
    env_batch_size: int = 2,
) -> dict[str, Any]:
    import lerobot
    import numpy
    import torch
    import transformers
    from lerobot.policies.factory import get_policy_class

    result = _smoke_benchmark(
        benchmark_name,
        suite,
        tasks,
        perturbations,
        env_batch_size,
    )
    return {
        **result,
        "lerobot": getattr(lerobot, "__version__", LEROBOT_REVISION),
        "vla_jepa_policy_class": get_policy_class("vla_jepa").__name__,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


@app.function(**_function_options(OPENVLA_IMAGE, cpu=4, memory=32768))
def download_openvla(
    suite: str = "libero_spatial",
    model_id: str = "",
    revision: str = "",
) -> dict[str, str]:
    from physical_ai_evals import openvla
    from physical_ai_evals.policy import _snapshot

    policy = openvla(
        suite,
        model_id=model_id or None,
        revision=revision or None,
        device="cuda",
    )
    path = _snapshot(policy.policy_id, policy.revision)
    MODEL_CACHE.commit()
    return {"model": f"{policy.policy_id}@{policy.revision}", "path": path}


@app.function(**_function_options(VLA_JEPA_IMAGE, cpu=4, memory=32768))
def download_vla_jepa() -> dict[str, str]:
    from physical_ai_evals.policy import _snapshot

    paths = {
        VLA_JEPA_MODEL_ID: _snapshot(VLA_JEPA_MODEL_ID, VLA_JEPA_REVISION),
        QWEN3_VL_MODEL_ID: _snapshot(QWEN3_VL_MODEL_ID, QWEN3_VL_REVISION),
        VJEPA2_MODEL_ID: _snapshot(VJEPA2_MODEL_ID, VJEPA2_REVISION),
    }
    MODEL_CACHE.commit()
    return paths


def _run(
    policy_name: str,
    benchmark_name: str,
    suite: str,
    tasks: list[str] | None,
    perturbations: list[str] | None,
    episodes: int,
    seed: int,
    model_id: str,
    revision: str,
    write_video: bool,
    env_batch_size: int,
) -> dict[str, Any]:
    from physical_ai_evals import evaluate, openvla, vla_jepa

    benchmark = _benchmark(
        benchmark_name,
        suite,
        tasks,
        perturbations,
        episodes,
        seed,
    )
    if policy_name == "openvla":
        policy_suite = "libero_goal" if benchmark_name == "libero_para" else suite
        policy = openvla(
            policy_suite,
            model_id=model_id or None,
            revision=revision or None,
            device="cuda",
        )
    elif policy_name == "vla_jepa":
        policy = vla_jepa(
            model_id=model_id or VLA_JEPA_MODEL_ID,
            revision=revision or VLA_JEPA_REVISION,
            device="cuda",
        )
    else:
        raise ValueError("policy must be 'openvla' or 'vla_jepa'")

    def checkpoint() -> None:
        # Modal does not preserve uncommitted Volume mutations after a crash.
        # Commit after every completed environment cohort.
        MODEL_CACHE.commit()
        OUTPUTS.commit()

    output_root = f"{OUTPUT_DIR}/evaluations/{policy_name}/{benchmark_name}"
    evaluation = evaluate(
        policy,
        benchmark,
        out=output_root,
        write_video=write_video,
        checkpoint=checkpoint,
        env_batch_size=env_batch_size,
    )
    checkpoint()
    summary = evaluation.metrics().to_pydict()
    return {
        "policy": policy_name,
        "benchmark": benchmark_name,
        "evaluation_id": evaluation.evaluation_id,
        "episodes": int(summary["episodes"][0]),
        "successes": int(summary["successes"][0] or 0),
        "success_rate": evaluation.success_rate(),
        "out_dir": str(evaluation.path),
        "timing_path": str(evaluation.path / "timings.jsonl"),
    }


@app.function(
    **_function_options(
        OPENVLA_IMAGE,
        gpu=GPU_TYPE,
        cpu=32,
        memory=65536,
    )
)
def run_openvla(
    benchmark_name: str,
    suite: str,
    tasks: list[str] | None = None,
    perturbations: list[str] | None = None,
    episodes: int = 50,
    seed: int = 7,
    model_id: str = "",
    revision: str = "",
    write_video: bool = True,
    env_batch_size: int = 8,
) -> dict[str, Any]:
    return _run(
        "openvla",
        benchmark_name,
        suite,
        tasks,
        perturbations,
        episodes,
        seed,
        model_id,
        revision,
        write_video,
        env_batch_size,
    )


@app.function(
    **_function_options(
        VLA_JEPA_IMAGE,
        gpu=GPU_TYPE,
        cpu=32,
        memory=98304,
    )
)
def run_vla_jepa(
    benchmark_name: str,
    suite: str,
    tasks: list[str] | None = None,
    perturbations: list[str] | None = None,
    episodes: int = 50,
    seed: int = 7,
    model_id: str = "",
    revision: str = "",
    write_video: bool = True,
    env_batch_size: int = 8,
) -> dict[str, Any]:
    return _run(
        "vla_jepa",
        benchmark_name,
        suite,
        tasks,
        perturbations,
        episodes,
        seed,
        model_id,
        revision,
        write_video,
        env_batch_size,
    )


@app.local_entrypoint()
def modal_main(
    policy: str = "openvla",
    benchmark: str = "libero",
    suite: str = "libero_spatial",
    tasks: str = "",
    perturbations: str = "",
    episodes: int = 50,
    seed: int = 7,
    model_id: str = "",
    revision: str = "",
    write_video: bool = True,
    env_batch_size: int = 8,
    download_only: bool = False,
    smoke_test: bool = False,
) -> None:
    task_list = [item.strip() for item in tasks.split(",") if item.strip()] or None
    perturbation_list = [item.strip() for item in perturbations.split(",") if item.strip()] or None

    if policy not in {"openvla", "vla_jepa"}:
        raise SystemExit("--policy must be openvla or vla_jepa")
    if smoke_test:
        function = smoke_openvla if policy == "openvla" else smoke_vla_jepa
        print(
            function.remote(
                benchmark_name=benchmark,
                suite=suite,
                tasks=task_list,
                perturbations=perturbation_list,
                env_batch_size=env_batch_size,
            )
        )
        return
    if download_only:
        if policy == "openvla":
            policy_suite = "libero_goal" if benchmark == "libero_para" else suite
            print(download_openvla.remote(policy_suite, model_id, revision))
        else:
            print(download_vla_jepa.remote())
        return

    function = run_openvla if policy == "openvla" else run_vla_jepa
    result = function.remote(
        benchmark_name=benchmark,
        suite=suite,
        tasks=task_list,
        perturbations=perturbation_list,
        episodes=episodes,
        seed=seed,
        model_id=model_id,
        revision=revision,
        write_video=write_video,
        env_batch_size=env_batch_size,
    )
    print(
        f"{result['successes']}/{result['episodes']} succeeded "
        f"({result['success_rate']:.3f}) -> {result['out_dir']}"
    )
