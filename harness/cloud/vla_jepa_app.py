"""Modal deployment for VLA-JEPA LIBERO rollouts — in-process via the lerobot port.

Run::

    modal run harness/cloud/vla_jepa_app.py --smoke-test
    modal run harness/cloud/vla_jepa_app.py --download-only
    modal run harness/cloud/vla_jepa_app.py --suites libero_spatial --task-ids 0 --episodes 2
"""

from __future__ import annotations

import modal

from harness.cloud.modal_infra import (
    APP_DIR,
    MODAL_LOCAL_DIR_IGNORE,
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    hf_cache_env,
)

GPU_TYPE = "A100-40GB"
MODAL_REGION = ["us-west"]
CUDA_BASE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
_PY = "3.12"

LEROBOT_PIN = "git+https://github.com/huggingface/lerobot@052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34"

CHECKPOINT_REPO = "lerobot/VLA-JEPA-LIBERO"
CHECKPOINT_REVISION = "735d9f692981e286ade093b5046627eda876e5d0"
QWEN3_VL_REPO = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_VL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
VJEPA2_REPO = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"

app = modal.App("daft-vlajepa-libero-rollout")

MODEL_CACHE = modal.Volume.from_name("daft-model-cache", create_if_missing=True)
OUTPUTS = modal.Volume.from_name("daft-model-outputs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("hf-token")
VOLUMES = {MODEL_CACHE_DIR: MODEL_CACHE, OUTPUT_DIR: OUTPUTS}

_GL_APT = (
    "git", "ffmpeg", "build-essential", "clang", "linux-libc-dev",
    "cmake",
    "libgl1", "libglib2.0-0", "libegl1", "libgles2",
    "libosmesa6", "libosmesa6-dev", "libsm6", "libxext6", "patchelf",
)


def vla_jepa_image() -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_BASE, add_python=_PY)
        .apt_install(*_GL_APT)
        .pip_install(
            f"lerobot[vla_jepa,libero] @ {LEROBOT_PIN}",
            "daft>=0.7.17",
            "huggingface_hub",
            "hf_xet",
            "imageio[ffmpeg]",
        )
        .run_commands("printf 'n\\n' | python -c 'import libero.libero'")
        .env({**hf_cache_env(), "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
        .add_local_dir(".", remote_path=APP_DIR, copy=True, ignore=MODAL_LOCAL_DIR_IGNORE)
        .add_local_python_source("harness")
    )


def _fn_kwargs(image: modal.Image, *, gpu: str | None = None, cpu: float = 8, memory: int = 65536,
               timeout: int = 14400) -> dict:
    kwargs: dict = {
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
        kwargs["gpu"] = gpu
    return kwargs


@app.function(**_fn_kwargs(vla_jepa_image(), cpu=2))
def smoke() -> dict:
    import os

    import daft
    import lerobot
    import numpy
    from lerobot.policies.factory import get_policy_class
    from libero.libero import benchmark, get_libero_path

    policy_cls = get_policy_class("vla_jepa")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return {
        "lerobot": getattr(lerobot, "__version__", "git"),
        "vla_jepa_policy_class": policy_cls.__name__,
        "numpy": numpy.__version__,
        "daft": daft.__version__,
        "lerobot_revision": LEROBOT_PIN.rsplit("@", 1)[-1],
        "checkpoint_revision": CHECKPOINT_REVISION,
        "qwen3_vl_revision": QWEN3_VL_REVISION,
        "vjepa2_revision": VJEPA2_REVISION,
        "libero_spatial_n_tasks": suite.get_num_tasks(),
        "task0_instruction": task.language,
        "bddl_exists": os.path.exists(bddl),
        "bddl_path": bddl,
    }


@app.function(**_fn_kwargs(vla_jepa_image(), cpu=4))
def download_vla_jepa() -> dict:
    from huggingface_hub import snapshot_download

    paths = {
        CHECKPOINT_REPO: snapshot_download(
            repo_id=CHECKPOINT_REPO, revision=CHECKPOINT_REVISION
        ),
        QWEN3_VL_REPO: snapshot_download(
            repo_id=QWEN3_VL_REPO, revision=QWEN3_VL_REVISION
        ),
        VJEPA2_REPO: snapshot_download(repo_id=VJEPA2_REPO, revision=VJEPA2_REVISION),
    }
    MODEL_CACHE.commit()
    return paths


@app.function(**_fn_kwargs(vla_jepa_image(), gpu=GPU_TYPE, memory=98304))
def run_sweep_vla_jepa(suites: list[str], task_ids: list[int] | None = None, episodes: int = 50,
                       model_id: str = CHECKPOINT_REPO,
                       model_revision: str = CHECKPOINT_REVISION, seed: int = 7,
                       write_video: bool = True) -> dict:
    from harness.cloud.rollout_udf import build_rollout_dataframe
    from harness.cloud.sweep import (
        enumerate_specs,
        evaluation_fingerprint,
        immutable_model_id,
        implementation_fingerprint,
        resolve_vla_jepa_config,
        runtime_provenance,
        write_evaluation_manifest,
    )
    from harness.core.config import SUITE_MAX_STEPS

    model_id, model_revision = resolve_vla_jepa_config(model_id, model_revision)
    recorded_model_id = immutable_model_id(model_id, model_revision)
    evaluation_config = {
        "policy_type": "vla_jepa",
        "model": recorded_model_id,
        "suites": sorted(set(suites)),
        "seed": seed,
        "camera_height": 224,
        "camera_width": 224,
        "num_steps_wait": 10,
        "max_steps": {suite: SUITE_MAX_STEPS.get(suite, 300) for suite in sorted(set(suites))},
        "control_mode": "relative",
        "write_frames": True,
        "write_video": write_video,
        "lerobot_revision": LEROBOT_PIN.rsplit("@", 1)[-1],
        "dependency_models": {
            QWEN3_VL_REPO: QWEN3_VL_REVISION,
            VJEPA2_REPO: VJEPA2_REVISION,
        },
        "implementation": implementation_fingerprint("vla_jepa"),
        "runtime": runtime_provenance(
            {
                "daft": "daft",
                "huggingface_hub": "huggingface-hub",
                "hf_xet": "hf-xet",
                "imageio": "imageio",
                "lerobot": "lerobot",
                "libero": "libero",
                "modal": "modal",
                "numpy": "numpy",
                "torch": "torch",
            }
        ),
    }
    evaluation_id = evaluation_fingerprint(evaluation_config)
    out_dir = f"{OUTPUT_DIR}/rollouts/vla_jepa/{evaluation_id}"
    write_evaluation_manifest(out_dir, evaluation_id, evaluation_config)
    s, t, i, sd = enumerate_specs(
        suites,
        task_ids,
        episodes,
        seed,
        out_dir=out_dir,
        policy_type="vla_jepa",
        model_id=recorded_model_id,
    )
    if not s:
        return {"policy_type": "vla_jepa", "episodes": 0, "successes": 0,
                "out_dir": out_dir, "summary": {}, "evaluation_id": evaluation_id,
                "model_id": recorded_model_id, "evaluation_config": evaluation_config,
                "note": "all episodes already on volume"}
    df = build_rollout_dataframe(
        s, t, i, sd,
        policy_type="vla_jepa",
        out_dir=out_dir,
        model_id=model_id,
        model_revision=model_revision,
        frames_dir=f"{OUTPUT_DIR}/frames/vla_jepa/{evaluation_id}",
        videos_dir=f"{OUTPUT_DIR}/videos/vla_jepa/{evaluation_id}",
        run_id=f"rollout-vla_jepa-{evaluation_id}",
        device="cuda",
        camera_height=224,
        camera_width=224,
        write_video=write_video,
    ).collect()
    MODEL_CACHE.commit()
    OUTPUTS.commit()
    summary = df.to_pydict()
    n = len(summary.get("episode_id", []))
    n_success = sum(summary.get("success", []))
    return {"policy_type": "vla_jepa", "episodes": n, "successes": n_success,
            "out_dir": out_dir, "summary": summary, "evaluation_id": evaluation_id,
            "model_id": recorded_model_id, "evaluation_config": evaluation_config}


@app.local_entrypoint()
def modal_main(
    suites: str = "libero_spatial",
    task_ids: str = "",
    episodes: int = 50,
    model_id: str = CHECKPOINT_REPO,
    model_revision: str = CHECKPOINT_REVISION,
    seed: int = 7,
    write_video: bool = True,
    download_only: bool = False,
    smoke_test: bool = False,
):
    if smoke_test:
        print(smoke.remote())
        return
    if download_only:
        print(download_vla_jepa.remote())
        return
    suite_list = [s.strip() for s in suites.split(",") if s.strip()]
    task_list = [int(t) for t in task_ids.split(",") if t.strip()] or None
    result = run_sweep_vla_jepa.remote(
        suites=suite_list,
        task_ids=task_list,
        episodes=episodes,
        model_id=model_id,
        model_revision=model_revision,
        seed=seed,
        write_video=write_video,
    )
    print(f"{result['successes']}/{result['episodes']} succeeded -> {result['out_dir']}")
