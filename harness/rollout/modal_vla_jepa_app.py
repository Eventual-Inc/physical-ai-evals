"""Modal deployment for VLA-JEPA LIBERO rollouts — in-process via the lerobot port.

Separate from ``modal_app.py`` on purpose: the two policy stacks want different images
(OpenVLA = py3.10 / transformers 4.40; VLA-JEPA = py3.12 / transformers 5.4–5.6 via lerobot),
and Modal builds every image an app references.

No policy server. ``lerobot[vla_jepa,libero]`` gives us everything in ONE process:
  * the VLA-JEPA policy (``lerobot.policies.vla_jepa``, main-only until the next release —
    pinned at a git SHA the way openpi pins lerobot),
  * the official ``lerobot/VLA-JEPA-LIBERO`` safetensors checkpoint (unnorm stats baked in),
  * LIBERO itself via the ``hf-libero`` wheel — bddl files / assets / init states ship in the
    package with the same ``libero.libero`` import paths, so the git-clone + config-patch +
    PYTHONPATH machinery of the server route is gone (see git history for that version).

Run::

    modal run harness/rollout/modal_vla_jepa_app.py --smoke-test
    modal run harness/rollout/modal_vla_jepa_app.py --download-only
    modal run harness/rollout/modal_vla_jepa_app.py --suites libero_spatial --task-ids 0 --episodes 2
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
_PY = "3.12"  # lerobot requires >=3.12 (vs the OpenVLA image's 3.10)

#: lerobot main SHA carrying the vla_jepa policy (merged 2026-06-04; includes the 2026-06-29
#: GPU-roundtrip fix — the only two vla_jepa commits). Swap for a PyPI pin at the next release.
LEROBOT_PIN = "git+https://github.com/huggingface/lerobot@052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34"

CHECKPOINT_REPO = "lerobot/VLA-JEPA-LIBERO"
QWEN3_VL_REPO = "Qwen/Qwen3-VL-2B-Instruct"     # backbone, pulled at policy construction
VJEPA2_REPO = "facebook/vjepa2-vitl-fpc64-256"  # world-model encoder (training-only, still loaded)

app = modal.App("daft-vlajepa-libero-rollout")

MODEL_CACHE = modal.Volume.from_name("daft-model-cache", create_if_missing=True)
OUTPUTS = modal.Volume.from_name("daft-model-outputs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("hf-token")
VOLUMES = {MODEL_CACHE_DIR: MODEL_CACHE, OUTPUT_DIR: OUTPUTS}

_GL_APT = (
    "git", "ffmpeg", "build-essential", "clang", "linux-libc-dev",
    "cmake",  # hf-libero -> egl_probe is a CMake-built C ext; CUDA base has no cmake (NOTES.md)
    "libgl1", "libglib2.0-0", "libegl1", "libgles2",
    "libosmesa6", "libosmesa6-dev", "libsm6", "libxext6", "patchelf",
)


def vla_jepa_image() -> modal.Image:
    """One pip layer: lerobot@SHA with the vla_jepa + libero extras, plus our data plane.

    numpy resolves to 2.x here (lerobot core requires >=2.0) — the harness is source-mounted,
    not pip-installed, so its numpy<2 pin doesn't fight the resolver; daft/pyarrow under
    numpy 2 is a smoke-test watch item (NOTES.md).
    """
    return (
        modal.Image.from_registry(CUDA_BASE, add_python=_PY)
        .apt_install(*_GL_APT)
        .pip_install(
            f"lerobot[vla_jepa,libero] @ {LEROBOT_PIN}",
            "daft>=0.7.16",
            "huggingface_hub",
            "hf_xet",
            "imageio[ffmpeg]",
        )
        # hf-libero kept LIBERO's interactive first-import dataset-path prompt (EOFError in a
        # container). Import once at build with 'n' piped in so LIBERO writes its own default
        # config (pointing at the wheel's bddl/assets/init paths) into the image (NOTES.md).
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
    """CPU image check: lerobot vla_jepa registers, hf-libero's bddl assets resolve on disk."""
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
        "libero_spatial_n_tasks": suite.get_num_tasks(),
        "task0_instruction": task.language,
        "bddl_exists": os.path.exists(bddl),
        "bddl_path": bddl,
    }


@app.function(**_fn_kwargs(vla_jepa_image(), cpu=4))
def download_vla_jepa() -> dict:
    """Warm the HF cache volume with the checkpoint + both backbones (three repos)."""
    from huggingface_hub import snapshot_download

    paths = {repo: snapshot_download(repo_id=repo) for repo in (CHECKPOINT_REPO, QWEN3_VL_REPO, VJEPA2_REPO)}
    MODEL_CACHE.commit()
    return paths


def _enumerate_specs(suites: list[str], task_ids: list[int] | None, episodes: int, seed: int):
    from harness.rollout.libero_runner import _libero_num_tasks

    s_col, t_col, i_col, seed_col = [], [], [], []
    for suite in suites:
        tasks = task_ids if task_ids is not None else range(_libero_num_tasks(suite))
        for task_id in tasks:
            for init_state_id in range(episodes):
                s_col.append(suite)
                t_col.append(int(task_id))
                i_col.append(init_state_id)
                seed_col.append(seed)
    return s_col, t_col, i_col, seed_col


@app.function(**_fn_kwargs(vla_jepa_image(), gpu=GPU_TYPE, memory=98304))
def run_sweep_vla_jepa(suites: list[str], task_ids: list[int] | None = None, episodes: int = 50,
                       model_id: str = CHECKPOINT_REPO, seed: int = 7,
                       write_video: bool = True) -> dict:
    from harness.rollout.rollout_udf import build_rollout_dataframe

    s, t, i, sd = _enumerate_specs(suites, task_ids, episodes, seed)
    out_dir = f"{OUTPUT_DIR}/rollouts/vla_jepa"
    df = build_rollout_dataframe(
        s, t, i, sd,
        policy_type="vla_jepa",
        out_dir=out_dir,
        model_id=model_id,
        frames_dir=f"{OUTPUT_DIR}/frames/vla_jepa",
        videos_dir=f"{OUTPUT_DIR}/videos/vla_jepa",
        run_id="rollout-vla_jepa",
        device="cuda",
        # 224 matches the lerobot-validated eval (96.5% over 400 eps); the model
        # area-resizes anyway, so this just skips a wasted downscale.
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
            "out_dir": out_dir, "summary": summary}


@app.local_entrypoint()
def modal_main(
    suites: str = "libero_spatial",
    task_ids: str = "",
    episodes: int = 50,
    model_id: str = CHECKPOINT_REPO,
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
        seed=seed,
        write_video=write_video,
    )
    print(f"{result['successes']}/{result['episodes']} succeeded -> {result['out_dir']}")
