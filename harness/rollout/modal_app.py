"""Modal deployment shell for the LIBERO rollout UDF.

The rollout UDF lives in ``rollout_udf.py``; this file owns only the container image, Volumes,
and entrypoints — mirroring the daft-examples ``models/<name>/modal_app.py`` convention.

Run::

    modal run harness/rollout/modal_app.py --policy-type openvla --suites libero_goal --episodes 5
    modal run harness/rollout/modal_app.py --policy-type openvla --download-only

⚠️  NOT YET DEPLOY-VERIFIED. The container image below is the #1 open question (see NOTES.md):
LIBERO (robosuite 1.4 / numpy 1.22.4, historically Python 3.8) must co-exist IN ONE IMAGE with
a VLA policy stack (OpenVLA: torch 2.2 / transformers 4.40; VLA-JEPA: modern lerobot) for the
in-process closed loop. Modern LIBERO forks run on 3.10/3.11 (OpenPI does), so we build on 3.11
— but the exact numpy/torch pin compatibility must be validated on the first deploy. If they
cannot coexist, fall back to the OpenPI policy-server / env-client SPLIT (websocket between two
images). The two policy stacks conflict with each other, so they get SEPARATE images.
"""

from __future__ import annotations

import modal

from harness._modal import (
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

# Shared with the daft-examples model cache so VLA weights dedupe across runs.
MODEL_CACHE = modal.Volume.from_name("daft-model-cache", create_if_missing=True)
OUTPUTS = modal.Volume.from_name("daft-model-outputs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("hf-token")
VOLUMES = {MODEL_CACHE_DIR: MODEL_CACHE, OUTPUT_DIR: OUTPUTS}


def _libero_base() -> modal.Image:
    """LIBERO robosuite/MuJoCo sim layer (headless EGL). The fragile pins live here."""
    return (
        modal.Image.from_registry(CUDA_BASE, add_python="3.11")
        .apt_install("git", "ffmpeg", "libosmesa6-dev", "libgl1", "libglib2.0-0", "libegl1", "libgles2")
        .pip_install(
            "numpy==1.22.4",        # robosuite 1.4 needs <1.24 (np.float/np.bool); see NOTES.md
            "robosuite==1.4.0",
            "bddl==1.0.1",
            "gym==0.25.2",          # OLD gym 4-tuple step API
            "easydict",
            "imageio[ffmpeg]",
            "daft>=0.7.15",
            "huggingface_hub",
            "hf_xet",
        )
        .run_commands("pip install --no-deps git+https://github.com/Lifelong-Robot-Learning/LIBERO.git")
        .env({**hf_cache_env(), "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
    )


def openvla_image() -> modal.Image:
    return (
        _libero_base()
        # OpenVLA's transformers pin is load-bearing and CONFLICTS with VLA-JEPA -> own image.
        .pip_install("torch==2.2.0", "transformers==4.40.1", "tokenizers==0.19.1", "timm==0.9.10", "accelerate", "pillow")
        .add_local_dir(".", remote_path=APP_DIR, copy=True, ignore=MODAL_LOCAL_DIR_IGNORE)
        .add_local_python_source("harness")
    )


def vla_jepa_image() -> modal.Image:
    return (
        _libero_base()
        .pip_install("lerobot>=0.5.0")  # modern LeRobot/Qwen3-VL stack
        .add_local_dir(".", remote_path=APP_DIR, copy=True, ignore=MODAL_LOCAL_DIR_IGNORE)
        .add_local_python_source("harness")
    )


def _image_for(policy_type: str) -> modal.Image:
    return {"openvla": openvla_image, "vla_jepa": vla_jepa_image}[policy_type]()


def _fn_kwargs(image: modal.Image, *, gpu: str | None = None, cpu: float = 8, memory: int = 32768,
               timeout: int = 7200) -> dict:
    kwargs: dict = {
        "image": image, "cpu": cpu, "memory": memory, "timeout": timeout, "region": MODAL_REGION,
        "volumes": VOLUMES, "secrets": [HF_SECRET], "enable_memory_snapshot": False,
    }
    if gpu is not None:
        kwargs["gpu"] = gpu
    return kwargs


@app.function(**_fn_kwargs(openvla_image(), cpu=4))
def download_openvla(model_id: str = "openvla/openvla-7b-finetuned-libero-spatial") -> dict:
    return _download(model_id)


@app.function(**_fn_kwargs(vla_jepa_image(), cpu=4))
def download_vla_jepa(model_id: str = "lerobot/VLA-JEPA-LIBERO") -> dict:
    return _download(model_id)


def _download(model_id: str) -> dict:
    from harness._modal import resolve_hf_model_path

    path = resolve_hf_model_path(model_id, MODEL_CACHE_DIR)
    MODEL_CACHE.commit()
    return {"model_id": model_id, "model_path": str(path)}


def _enumerate_specs(suites: list[str], task_ids: list[int] | None, episodes: int, seed: int):
    """Expand (suites × tasks × episodes) into flat spec columns. Runs where LIBERO is importable."""
    from harness.rollout.libero_runner import _libero_num_tasks

    s_col, t_col, i_col, seed_col = [], [], [], []
    for suite in suites:
        tasks = task_ids if task_ids is not None else range(_libero_num_tasks(suite))
        for task_id in tasks:
            for init_state_id in range(episodes):
                s_col.append(suite); t_col.append(int(task_id)); i_col.append(init_state_id); seed_col.append(seed)
    return s_col, t_col, i_col, seed_col


def _run_sweep(policy_type: str, suites: list[str], task_ids: list[int] | None, episodes: int,
               model_id: str, seed: int, write_video: bool) -> dict:
    from harness.rollout.rollout_udf import build_rollout_dataframe

    s, t, i, sd = _enumerate_specs(suites, task_ids, episodes, seed)
    out_dir = f"{OUTPUT_DIR}/rollouts/{policy_type}"
    df = build_rollout_dataframe(
        s, t, i, sd, policy_type=policy_type, out_dir=out_dir, model_id=model_id,
        frames_dir=f"{OUTPUT_DIR}/frames/{policy_type}", videos_dir=f"{OUTPUT_DIR}/videos/{policy_type}",
        run_id=f"rollout-{policy_type}", device="cuda", write_video=write_video,
    ).collect()
    MODEL_CACHE.commit()
    OUTPUTS.commit()
    summary = df.to_pydict()
    n = len(summary.get("episode_id", []))
    n_success = sum(summary.get("success", []))
    return {"policy_type": policy_type, "episodes": n, "successes": n_success,
            "out_dir": out_dir, "summary": summary}


@app.function(**_fn_kwargs(openvla_image(), gpu=GPU_TYPE, memory=65536))
def run_sweep_openvla(suites: list[str], task_ids: list[int] | None = None, episodes: int = 10,
                      model_id: str = "", seed: int = 0, write_video: bool = True) -> dict:
    return _run_sweep("openvla", suites, task_ids, episodes, model_id, seed, write_video)


@app.function(**_fn_kwargs(vla_jepa_image(), gpu=GPU_TYPE, memory=65536))
def run_sweep_vla_jepa(suites: list[str], task_ids: list[int] | None = None, episodes: int = 10,
                       model_id: str = "", seed: int = 0, write_video: bool = True) -> dict:
    return _run_sweep("vla_jepa", suites, task_ids, episodes, model_id, seed, write_video)


@app.local_entrypoint()
def modal_main(
    policy_type: str = "openvla",
    suites: str = "libero_goal",          # comma-separated suite keys
    task_ids: str = "",                   # comma-separated; empty = all tasks in each suite
    episodes: int = 10,
    model_id: str = "",
    seed: int = 0,
    write_video: bool = True,
    download_only: bool = False,
):
    suite_list = [s.strip() for s in suites.split(",") if s.strip()]
    task_list = [int(t) for t in task_ids.split(",") if t.strip()] or None

    if download_only:
        dl = {"openvla": download_openvla, "vla_jepa": download_vla_jepa}[policy_type]
        print(dl.remote(model_id) if model_id else dl.remote())
        return

    fn = {"openvla": run_sweep_openvla, "vla_jepa": run_sweep_vla_jepa}[policy_type]
    result = fn.remote(suites=suite_list, task_ids=task_list, episodes=episodes,
                       model_id=model_id, seed=seed, write_video=write_video)
    print(f"{result['successes']}/{result['episodes']} succeeded -> {result['out_dir']}")
