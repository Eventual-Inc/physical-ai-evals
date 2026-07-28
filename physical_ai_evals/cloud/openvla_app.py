"""Modal deployment shell for the OpenVLA LIBERO rollout UDF.

Run::

    modal run physical_ai_evals/cloud/openvla_app.py --suites libero_spatial --episodes 5 --seed 7
    modal run physical_ai_evals/cloud/openvla_app.py --download-only
"""

from __future__ import annotations

import modal

from physical_ai_evals.cloud.modal_infra import (
    APP_DIR,
    MODAL_LOCAL_DIR_IGNORE,
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    hf_cache_env,
)

GPU_TYPE = "A100-40GB"
MODAL_REGION = ["us-west"]
CUDA_BASE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"

app = modal.App("daft-libero-rollout")

MODEL_CACHE = modal.Volume.from_name("daft-model-cache", create_if_missing=True)
OUTPUTS = modal.Volume.from_name("daft-model-outputs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("hf-token")
VOLUMES = {MODEL_CACHE_DIR: MODEL_CACHE, OUTPUT_DIR: OUTPUTS}

_GL_APT = (
    "git", "ffmpeg", "build-essential", "clang", "linux-libc-dev",
    "libgl1", "libglib2.0-0", "libegl1", "libgles2",
    "libosmesa6", "libosmesa6-dev", "libsm6", "libxext6", "patchelf",
)

_LIBERO_SIM_PINS = (
    "robosuite==1.4.1", "bddl", "easydict", "cloudpickle", "gym",
    "imageio[ffmpeg]", "opencv-python==4.9.0.80", "matplotlib", "einops",
    "mujoco==3.9.0",
    "scipy==1.15.3",
)
_LIBERO_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
_LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
_PY = "3.12"

_LIBERO_CFG = "/opt/LIBERO/.libero_config"
_LIBERO_ROOT = "/opt/LIBERO/libero/libero"


def _with_libero(image: modal.Image) -> modal.Image:
    return (
        image.pip_install(*_LIBERO_SIM_PINS)
        .run_commands(
            "git init /opt/LIBERO",
            f"git -C /opt/LIBERO remote add origin {_LIBERO_REPO}",
            f"git -C /opt/LIBERO fetch --depth 1 origin {_LIBERO_REVISION}",
            "git -C /opt/LIBERO checkout --detach FETCH_HEAD",
            "pip install --no-deps -e /opt/LIBERO",
            f"mkdir -p {_LIBERO_CFG}",
            f"echo 'benchmark_root: {_LIBERO_ROOT}' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'bddl_files: {_LIBERO_ROOT}/bddl_files' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'init_states: {_LIBERO_ROOT}/init_files' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'assets: {_LIBERO_ROOT}/assets' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'datasets: {_LIBERO_ROOT}/../datasets' >> {_LIBERO_CFG}/config.yaml",
        )
        .env({
            "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": "/opt/LIBERO", "LIBERO_CONFIG_PATH": _LIBERO_CFG,
        })
    )


def _with_pipeline(image: modal.Image) -> modal.Image:
    return (
        image.pip_install("daft>=0.7.17", "huggingface_hub", "hf_xet", "numpy==1.26.4")
        .env(hf_cache_env())
        .add_local_dir(".", remote_path=APP_DIR, copy=True, ignore=MODAL_LOCAL_DIR_IGNORE)
        .add_local_python_source("physical_ai_evals")
    )


def openvla_image() -> modal.Image:
    base = modal.Image.from_registry(CUDA_BASE, add_python=_PY).apt_install(*_GL_APT).pip_install(
        "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0",
        "transformers==4.40.1", "tokenizers==0.19.1", "timm==0.9.10",
        "accelerate>=0.25.0", "json-numpy", "pillow",
    )
    return _with_pipeline(_with_libero(base))


def _fn_kwargs(image: modal.Image, *, gpu: str | None = None, cpu: float = 8, memory: int = 32768,
               timeout: int = 14400) -> dict:
    kwargs: dict = {
        "image": image, "cpu": cpu, "memory": memory, "timeout": timeout, "region": MODAL_REGION,
        "volumes": VOLUMES, "secrets": [HF_SECRET], "enable_memory_snapshot": False,
    }
    if gpu is not None:
        kwargs["gpu"] = gpu
    return kwargs


@app.function(**_fn_kwargs(openvla_image(), cpu=4))
def download_openvla(
    model_id: str = "openvla/openvla-7b-finetuned-libero-spatial",
    model_revision: str = "",
) -> dict:
    return _download(model_id, model_revision)


@app.function(**_fn_kwargs(openvla_image(), cpu=2))
def smoke() -> dict:
    import os

    import daft
    import numpy
    import robosuite
    import transformers
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return {
        "numpy": numpy.__version__,
        "transformers": transformers.__version__,
        "robosuite": robosuite.__version__,
        "daft": daft.__version__,
        "libero_revision": _LIBERO_REVISION,
        "libero_spatial_n_tasks": suite.get_num_tasks(),
        "task0_instruction": task.language,
        "bddl_exists": os.path.exists(bddl),
        "bddl_path": bddl,
    }


def _download(model_id: str, model_revision: str = "") -> dict:
    from physical_ai_evals.cloud.modal_infra import resolve_hf_model_path
    from physical_ai_evals.cloud.sweep import OPENVLA_REVISIONS, immutable_model_id

    revision = model_revision or OPENVLA_REVISIONS.get(model_id)
    if revision is None:
        raise ValueError("a pinned model_revision is required for a custom OpenVLA checkpoint")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("model_revision must be an immutable 40-character commit SHA")
    path = resolve_hf_model_path(model_id, MODEL_CACHE_DIR, revision=revision)
    MODEL_CACHE.commit()
    return {
        "model_id": immutable_model_id(model_id, revision),
        "model_revision": revision,
        "model_path": str(path),
    }


def _run_sweep(policy_type: str, suites: list[str], task_ids: list[int] | None, episodes: int,
               model_id: str, model_revision: str, seed: int, write_video: bool) -> dict:
    from physical_ai_evals.cloud.rollout_udf import build_rollout_dataframe
    from physical_ai_evals.cloud.sweep import (
        enumerate_specs,
        evaluation_fingerprint,
        immutable_model_id,
        implementation_fingerprint,
        resolve_openvla_config,
        runtime_provenance,
        write_evaluation_manifest,
    )
    from physical_ai_evals.core.config import SUITE_MAX_STEPS

    model_id, model_revision, unnorm_key = resolve_openvla_config(
        suites,
        model_id=model_id or None,
        model_revision=model_revision or None,
    )
    recorded_model_id = immutable_model_id(model_id, model_revision)
    evaluation_config = {
        "policy_type": policy_type,
        "model": recorded_model_id,
        "suites": sorted(set(suites)),
        "seed": seed,
        "unnorm_key": unnorm_key,
        "attn_impl": "sdpa",
        "center_crop_scale": 0.9,
        "camera_height": 256,
        "camera_width": 256,
        "num_steps_wait": 10,
        "max_steps": {suite: SUITE_MAX_STEPS.get(suite, 300) for suite in sorted(set(suites))},
        "control_mode": "relative",
        "write_frames": True,
        "write_video": write_video,
        "libero_revision": _LIBERO_REVISION,
        "implementation": implementation_fingerprint(policy_type),
        "runtime": runtime_provenance(
            {
                "accelerate": "accelerate",
                "bddl": "bddl",
                "daft": "daft",
                "huggingface_hub": "huggingface-hub",
                "hf_xet": "hf-xet",
                "imageio": "imageio",
                "libero": "libero",
                "modal": "modal",
                "mujoco": "mujoco",
                "numpy": "numpy",
                "pillow": "pillow",
                "robosuite": "robosuite",
                "scipy": "scipy",
                "timm": "timm",
                "tokenizers": "tokenizers",
                "torch": "torch",
                "transformers": "transformers",
            }
        ),
    }
    evaluation_id = evaluation_fingerprint(evaluation_config)
    out_dir = f"{OUTPUT_DIR}/rollouts/{policy_type}/{evaluation_id}"
    write_evaluation_manifest(out_dir, evaluation_id, evaluation_config)
    s, t, i, sd = enumerate_specs(
        suites,
        task_ids,
        episodes,
        seed,
        out_dir=out_dir,
        policy_type=policy_type,
        model_id=recorded_model_id,
    )
    if not s:
        return {"policy_type": policy_type, "episodes": 0, "successes": 0,
                "out_dir": out_dir, "summary": {}, "evaluation_id": evaluation_id,
                "model_id": recorded_model_id, "evaluation_config": evaluation_config,
                "note": "all episodes already on volume"}
    df = build_rollout_dataframe(
        s, t, i, sd, policy_type=policy_type, out_dir=out_dir,
        model_id=model_id, model_revision=model_revision, unnorm_key=unnorm_key,
        frames_dir=f"{OUTPUT_DIR}/frames/{policy_type}/{evaluation_id}",
        videos_dir=f"{OUTPUT_DIR}/videos/{policy_type}/{evaluation_id}",
        run_id=f"rollout-{policy_type}-{evaluation_id}", device="cuda", write_video=write_video,
    ).collect()
    MODEL_CACHE.commit()
    OUTPUTS.commit()
    summary = df.to_pydict()
    n = len(summary.get("episode_id", []))
    n_success = sum(summary.get("success", []))
    return {"policy_type": policy_type, "episodes": n, "successes": n_success,
            "out_dir": out_dir, "summary": summary, "evaluation_id": evaluation_id,
            "model_id": recorded_model_id, "evaluation_config": evaluation_config}


@app.function(**_fn_kwargs(openvla_image(), gpu=GPU_TYPE, memory=65536))
def run_sweep_openvla(suites: list[str], task_ids: list[int] | None = None, episodes: int = 10,
                      model_id: str = "", model_revision: str = "", seed: int = 7,
                      write_video: bool = True) -> dict:
    return _run_sweep(
        "openvla", suites, task_ids, episodes, model_id, model_revision, seed, write_video
    )


@app.local_entrypoint()
def modal_main(
    policy_type: str = "openvla",
    suites: str = "libero_spatial",
    task_ids: str = "",
    episodes: int = 10,
    model_id: str = "",
    model_revision: str = "",
    seed: int = 7,
    write_video: bool = True,
    download_only: bool = False,
    smoke_test: bool = False,
):
    if smoke_test:
        print(smoke.remote())
        return

    suite_list = [s.strip() for s in suites.split(",") if s.strip()]
    task_list = [int(t) for t in task_ids.split(",") if t.strip()] or None

    if policy_type != "openvla":
        raise SystemExit(
            f"policy_type={policy_type!r} is in physical_ai_evals/cloud/vla_jepa_app.py. "
            "Use this app for --policy-type openvla."
        )

    if download_only:
        print(
            download_openvla.remote(model_id, model_revision)
            if model_id
            else download_openvla.remote(model_revision=model_revision)
        )
        return

    result = run_sweep_openvla.remote(
        suites=suite_list, task_ids=task_list, episodes=episodes,
        model_id=model_id, model_revision=model_revision, seed=seed, write_video=write_video,
    )
    print(f"{result['successes']}/{result['episodes']} succeeded -> {result['out_dir']}")
