"""Modal deployment for VLA-JEPA LIBERO rollouts.

Separate from ``modal_app.py`` on purpose: Modal builds images referenced by the app being
run, and the VLA-JEPA/StarVLA image is much heavier than the OpenVLA image. Keeping this in
its own app lets OpenVLA deploys stay independent while still giving VLA-JEPA a real path.

Run::

    modal run harness/rollout/modal_vla_jepa_app.py --download-only
    modal run harness/rollout/modal_vla_jepa_app.py --suites libero_goal --episodes 5

The function starts VLA-JEPA's official WebSocket policy server as a subprocess, then the
standard Daft rollout UDF connects to ``127.0.0.1:<port>`` through ``VLAJEPAPolicy``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import modal

from harness._modal import (
    APP_DIR,
    MODAL_LOCAL_DIR_IGNORE,
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    hf_cache_env,
    resolve_hf_model_path,
)
from harness.policies.vla_jepa import DEFAULT_LIBERO_CHECKPOINT, VLA_JEPA_REPO_ID

GPU_TYPE = "A100-40GB"
MODAL_REGION = ["us-west"]
CUDA_BASE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
VLA_JEPA_DIR = "/opt/VLA-JEPA"
VLA_JEPA_REPO = "https://github.com/ginwind/VLA-JEPA.git"
QWEN3_VL_REPO = "Qwen/Qwen3-VL-2B-Instruct"
VJEPA2_REPO = "facebook/vjepa2-vitl-fpc64-256"
POLICY_SERVER_PORT = 10093
_PY = "3.10"

app = modal.App("daft-vlajepa-libero-rollout")

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
)
_LIBERO_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
_LIBERO_CFG = "/opt/LIBERO/.libero_config"
_LIBERO_ROOT = "/opt/LIBERO/libero/libero"


def _with_libero(image: modal.Image) -> modal.Image:
    return (
        image.pip_install(*_LIBERO_SIM_PINS)
        .run_commands(
            f"git clone --depth 1 {_LIBERO_REPO} /opt/LIBERO",
            "pip install --no-deps -e /opt/LIBERO",
            f"mkdir -p {_LIBERO_CFG}",
            f"echo 'benchmark_root: {_LIBERO_ROOT}' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'bddl_files: {_LIBERO_ROOT}/bddl_files' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'init_states: {_LIBERO_ROOT}/init_files' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'assets: {_LIBERO_ROOT}/assets' >> {_LIBERO_CFG}/config.yaml",
            f"echo 'datasets: {_LIBERO_ROOT}/../datasets' >> {_LIBERO_CFG}/config.yaml",
        )
        .env({
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": f"/opt/LIBERO:{VLA_JEPA_DIR}",
            "LIBERO_CONFIG_PATH": _LIBERO_CFG,
        })
    )


def _with_pipeline(image: modal.Image) -> modal.Image:
    return (
        image.pip_install("daft>=0.7.15", "huggingface_hub", "hf_xet", "numpy==1.26.4")
        .env(hf_cache_env())
        .add_local_dir(".", remote_path=APP_DIR, copy=True, ignore=MODAL_LOCAL_DIR_IGNORE)
        .add_local_python_source("harness")
    )


def vla_jepa_image() -> modal.Image:
    """LIBERO + VLA-JEPA's official StarVLA server stack.

    The HF checkpoint config is patched at runtime to point at downloaded Qwen3-VL/V-JEPA2
    snapshots and to use SDPA instead of FlashAttention, keeping the image build tractable.
    """
    base = (
        modal.Image.from_registry(CUDA_BASE, add_python=_PY)
        .apt_install(*_GL_APT)
        .run_commands(
            f"git clone --depth 1 {VLA_JEPA_REPO} {VLA_JEPA_DIR}",
            f"pip install -r {VLA_JEPA_DIR}/requirements.txt",
            # The README's eval extras are not all in requirements.txt.
            "pip install tyro mediapy websockets msgpack msgpack-numpy",
            f"pip install -e {VLA_JEPA_DIR}",
        )
        .env({"PYTHONPATH": VLA_JEPA_DIR})
    )
    return _with_pipeline(_with_libero(base))


def _fn_kwargs(image: modal.Image, *, gpu: str | None = None, cpu: float = 8, memory: int = 65536,
               timeout: int = 7200) -> dict:
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


def _patch_vla_jepa_config(snapshot_root: Path, qwen_path: Path, vjepa_path: Path) -> Path:
    """Make HF snapshot configs portable inside Modal."""
    model_root = snapshot_root / "LIBERO"
    config_json = model_root / "config.json"
    config_yaml = model_root / "config.yaml"

    cfg = json.loads(config_json.read_text())
    cfg["framework"]["qwenvl"]["base_vlm"] = str(qwen_path)
    cfg["framework"]["qwenvl"]["attn_implementation"] = "sdpa"
    cfg["framework"]["vj2_model"]["base_encoder"] = str(vjepa_path)
    config_json.write_text(json.dumps(cfg, indent=2))

    text = config_yaml.read_text()
    text = text.replace("/home/dataset-local/models/Qwen3-VL-2B-Instruct", str(qwen_path))
    text = text.replace("/home/dataset-local/models/vjepa2-vitl-fpc64-256", str(vjepa_path))
    text = text.replace("attn_implementation: flash_attention_2", "attn_implementation: sdpa")
    config_yaml.write_text(text)
    return snapshot_root / DEFAULT_LIBERO_CHECKPOINT


def _prepare_vla_jepa(model_id: str) -> Path:
    requested = Path(model_id).expanduser()
    if requested.suffix == ".pt" and requested.exists():
        snapshot = requested.parent.parent
    else:
        snapshot = resolve_hf_model_path(model_id or VLA_JEPA_REPO_ID, MODEL_CACHE_DIR)
    qwen = resolve_hf_model_path(QWEN3_VL_REPO, MODEL_CACHE_DIR)
    vjepa = resolve_hf_model_path(VJEPA2_REPO, MODEL_CACHE_DIR)
    ckpt = _patch_vla_jepa_config(snapshot, qwen, vjepa)
    MODEL_CACHE.commit()
    return ckpt


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout_s: int = 600) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"VLA-JEPA policy server exited early with code {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"VLA-JEPA policy server did not open {host}:{port} within {timeout_s}s")


def _start_policy_server(checkpoint_path: Path, *, port: int = POLICY_SERVER_PORT) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VLA_JEPA_DIR}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        [
            sys.executable,
            f"{VLA_JEPA_DIR}/deployment/model_server/server_policy.py",
            "--ckpt_path",
            str(checkpoint_path),
            "--port",
            str(port),
            "--use_bf16",
            "--cuda",
            "0",
        ],
        env=env,
    )
    _wait_for_port("127.0.0.1", port, proc)
    return proc


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


@app.function(**_fn_kwargs(vla_jepa_image(), cpu=4, memory=65536))
def download_vla_jepa(model_id: str = VLA_JEPA_REPO_ID) -> dict:
    ckpt = _prepare_vla_jepa(model_id)
    return {"model_id": model_id, "checkpoint_path": str(ckpt)}


@app.function(**_fn_kwargs(vla_jepa_image(), gpu=GPU_TYPE, memory=98304))
def run_sweep_vla_jepa(suites: list[str], task_ids: list[int] | None = None, episodes: int = 10,
                       model_id: str = VLA_JEPA_REPO_ID, seed: int = 0,
                       write_video: bool = True) -> dict:
    from harness.rollout.rollout_udf import build_rollout_dataframe

    ckpt = _prepare_vla_jepa(model_id)
    proc = _start_policy_server(ckpt)
    try:
        s, t, i, sd = _enumerate_specs(suites, task_ids, episodes, seed)
        out_dir = f"{OUTPUT_DIR}/rollouts/vla_jepa"
        df = build_rollout_dataframe(
            s, t, i, sd,
            policy_type="vla_jepa",
            out_dir=out_dir,
            model_id=str(ckpt),
            frames_dir=f"{OUTPUT_DIR}/frames/vla_jepa",
            videos_dir=f"{OUTPUT_DIR}/videos/vla_jepa",
            run_id="rollout-vla_jepa",
            device="cuda",
            unnorm_key="franka",
            vla_jepa_host="127.0.0.1",
            vla_jepa_port=POLICY_SERVER_PORT,
            write_video=write_video,
        ).collect()
        OUTPUTS.commit()
        summary = df.to_pydict()
        n = len(summary.get("episode_id", []))
        n_success = sum(summary.get("success", []))
        return {"policy_type": "vla_jepa", "episodes": n, "successes": n_success,
                "out_dir": out_dir, "summary": summary}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


@app.local_entrypoint()
def modal_main(
    suites: str = "libero_goal",
    task_ids: str = "",
    episodes: int = 10,
    model_id: str = VLA_JEPA_REPO_ID,
    seed: int = 0,
    write_video: bool = True,
    download_only: bool = False,
):
    suite_list = [s.strip() for s in suites.split(",") if s.strip()]
    task_list = [int(t) for t in task_ids.split(",") if t.strip()] or None
    if download_only:
        print(download_vla_jepa.remote(model_id))
        return
    result = run_sweep_vla_jepa.remote(
        suites=suite_list,
        task_ids=task_list,
        episodes=episodes,
        model_id=model_id,
        seed=seed,
        write_video=write_video,
    )
    print(f"{result['successes']}/{result['episodes']} succeeded -> {result['out_dir']}")
