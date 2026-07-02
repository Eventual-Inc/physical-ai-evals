"""Modal deployment shell for the LIBERO rollout UDF.

The rollout UDF lives in ``rollout_udf.py``; this file owns only the container image, Volumes,
and entrypoints — mirroring the daft-examples ``models/<name>/modal_app.py`` convention.

Run::

    modal run harness/rollout/modal_app.py --policy-type openvla --suites libero_goal --episodes 5
    modal run harness/rollout/modal_app.py --policy-type openvla --download-only

For VLA-JEPA, use ``harness/rollout/modal_vla_jepa_app.py``. It is split out so its
heavier StarVLA image does not block OpenVLA deploys.

⚠️  NOT YET DEPLOY-VERIFIED. The container image below is the #1 open question (see NOTES.md):
LIBERO (robosuite 1.4 / numpy 1.22.4, historically Python 3.8) must co-exist IN ONE IMAGE with
a VLA policy stack for the in-process closed loop. Modern LIBERO forks run on 3.10/3.11
(OpenVLA does), so we build on 3.10. VLA-JEPA is intentionally handled by
``modal_vla_jepa_app.py`` because its official path is a StarVLA WebSocket policy server.
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


# Headless GL / sim apt deps (robosuite + MuJoCo EGL rendering).
# build-essential + clang: the CUDA runtime base has NO compiler, and Modal's add_python ships a
#   clang-built standalone interpreter, so building C exts (evdev) needs clang present (NOTES.md).
# linux-libc-dev: robosuite -> pynput -> evdev needs linux/input.h to build.
_GL_APT = (
    "git", "ffmpeg", "build-essential", "clang", "linux-libc-dev",
    "libgl1", "libglib2.0-0", "libegl1", "libgles2",
    "libosmesa6", "libosmesa6-dev", "libsm6", "libxext6", "patchelf",
)

# Per OpenVLA's official LIBERO eval (experiments/robot/libero/libero_requirements.txt). Python
# 3.10 is the proven combined env. numpy is intentionally LEFT UNPINNED — LIBERO's setup.py is
# dep-free (its requirements.txt with numpy==1.22.4 / transformers==4.21.1 is training-only and
# NOT used by `pip install -e LIBERO`), so pip resolves one numpy for torch 2.2 + robosuite 1.4.1.
# opencv-python==4.9.0.80: robosuite pulls opencv unpinned -> 4.13 which declares numpy>=2 and
# warns against our numpy 1.26.4; 4.9.0.80 is the last numpy-1-clean line (NOTES.md).
# matplotlib + einops: LIBERO runtime deps that `--no-deps` drops but env construction needs
# (env_wrapper imports matplotlib.cm; einops is used pervasively) — the smoke missed them because
# it only imported libero.libero.benchmark, not libero.libero.envs (NOTES.md).
_LIBERO_SIM_PINS = (
    "robosuite==1.4.1", "bddl", "easydict", "cloudpickle", "gym",
    "imageio[ffmpeg]", "opencv-python==4.9.0.80", "matplotlib", "einops",
)
_LIBERO_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
_PY = "3.10"


_LIBERO_CFG = "/opt/LIBERO/.libero_config"
_LIBERO_ROOT = "/opt/LIBERO/libero/libero"   # dir of libero/libero/__init__.py = the benchmark root


def _with_libero(image: modal.Image) -> modal.Image:
    """Add the LIBERO sim layer: robosuite runtime + the package CLONED on disk + a pre-written
    config so its ``__init__`` never hits the interactive dataset-path prompt.

    LIBERO is git-cloned (not pip-from-git) so its ``bddl_files`` / ``init_states`` ship on disk
    where ``get_libero_path`` resolves them; ``--no-deps`` because setup.py installs nothing and
    we don't want LIBERO's training requirements.txt pulled in.
    """
    return (
        image.pip_install(*_LIBERO_SIM_PINS)
        .run_commands(
            f"git clone --depth 1 {_LIBERO_REPO} /opt/LIBERO",
            "pip install --no-deps -e /opt/LIBERO",
            # Pre-write ~/.libero config (echo per line — no yaml dep, no quoting pain) so LIBERO's
            # __init__ skips its `input()` prompt (EOFError in a container). Points at the clone.
            f"mkdir -p {_LIBERO_CFG}",
            f"echo 'benchmark_root: {_LIBERO_ROOT}' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'bddl_files: {_LIBERO_ROOT}/bddl_files' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'init_states: {_LIBERO_ROOT}/init_files' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'assets: {_LIBERO_ROOT}/assets' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'datasets: {_LIBERO_ROOT}/../datasets' >> {_LIBERO_CFG}/config.yaml",
        )
        # PYTHONPATH=/opt/LIBERO: editable install doesn't import in the Modal function runtime.
        # LIBERO_CONFIG_PATH: where LIBERO reads config.yaml (default ~/.libero). See NOTES.md.
        .env({
            "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": "/opt/LIBERO", "LIBERO_CONFIG_PATH": _LIBERO_CFG,
        })
    )


def _with_pipeline(image: modal.Image) -> modal.Image:
    return (
        # numpy==1.26.4 (the verified resolution): daft pulls pyarrow 24 which would drag in
        # numpy 2.x, breaking torch 2.2 ("_ARRAY_API not found") and robosuite. Pin it (NOTES.md).
        image.pip_install("daft>=0.7.15", "huggingface_hub", "hf_xet", "numpy==1.26.4")
        .env(hf_cache_env())
        .add_local_dir(".", remote_path=APP_DIR, copy=True, ignore=MODAL_LOCAL_DIR_IGNORE)
        .add_local_python_source("harness")
    )


def openvla_image() -> modal.Image:
    """LIBERO + OpenVLA HF inference stack. We do NOT pip-install the openvla package — inference
    is pure ``AutoModelForVision2Seq(trust_remote_code=True).predict_action``, so only the HF
    stack is needed. flash-attn is omitted (painful build); the policy uses sdpa on A100."""
    base = modal.Image.from_registry(CUDA_BASE, add_python=_PY).apt_install(*_GL_APT).pip_install(
        "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0",
        "transformers==4.40.1", "tokenizers==0.19.1", "timm==0.9.10",
        "accelerate>=0.25.0", "json-numpy", "pillow",
    )
    return _with_pipeline(_with_libero(base))


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


@app.function(**_fn_kwargs(openvla_image(), cpu=2))
def smoke() -> dict:
    """CPU image smoke test (no GPU, no model download): do the imports resolve, is the LIBERO
    package + its bddl_files on disk, and — key for reproducibility — what numpy did pip pick?"""
    import os

    import daft
    import numpy
    import robosuite
    import transformers
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return {
        "numpy": numpy.__version__,
        "transformers": transformers.__version__,
        "robosuite": robosuite.__version__,
        "daft": daft.__version__,
        "libero_goal_n_tasks": suite.get_num_tasks(),
        "task0_instruction": task.language,
        "bddl_exists": os.path.exists(bddl),
        "bddl_path": bddl,
    }


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
    smoke_test: bool = False,
):
    if smoke_test:  # cheap CPU build check: imports + LIBERO bddl_files + resolved versions
        print(smoke.remote())
        return

    suite_list = [s.strip() for s in suites.split(",") if s.strip()]
    task_list = [int(t) for t in task_ids.split(",") if t.strip()] or None

    if policy_type != "openvla":
        raise SystemExit(
            f"policy_type={policy_type!r} is in harness/rollout/modal_vla_jepa_app.py. "
            "Use this app for --policy-type openvla."
        )

    if download_only:
        print(download_openvla.remote(model_id) if model_id else download_openvla.remote())
        return

    fn = run_sweep_openvla
    result = fn.remote(suites=suite_list, task_ids=task_list, episodes=episodes,
                       model_id=model_id, seed=seed, write_video=write_video)
    print(f"{result['successes']}/{result['episodes']} succeeded -> {result['out_dir']}")
