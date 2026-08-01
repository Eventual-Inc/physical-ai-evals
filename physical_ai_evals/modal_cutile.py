"""Pinned H100 LIBERO lane for the daft-cuTile VLA-JEPA engine."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import modal

from physical_ai_evals.cutile_vla_jepa import (
    CUTILE_DAFT_REVISION,
    CUTILE_VLA_STATIC_BATCH_SIZE,
)
from physical_ai_evals.libero import (
    LIBERO_PRO_CODE_REPOSITORY,
    LIBERO_PRO_CODE_REVISION,
)
from physical_ai_evals.modal import (
    _APT,
    _SIM,
    MODEL_CACHE,
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    OUTPUTS,
    _benchmark,
    _cache_environment,
    _function_options,
)

app = modal.App("physical-ai-evals-cutile")


def _checked_cutile_source() -> Path:
    raw = os.environ.get("DAFT_CUTILE_SOURCE_ROOT")
    if not raw:
        raise RuntimeError(
            "DAFT_CUTILE_SOURCE_ROOT must name a clean daft-cuTile checkout at "
            f"{CUTILE_DAFT_REVISION}"
        )
    root = Path(raw).expanduser().resolve(strict=True)
    if not (root / "scripts" / "modal" / "image.py").is_file():
        raise RuntimeError("DAFT_CUTILE_SOURCE_ROOT is not a daft-cuTile checkout")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    revision = git("rev-parse", "HEAD")
    if revision != CUTILE_DAFT_REVISION:
        raise RuntimeError(
            f"daft-cuTile source revision changed: expected {CUTILE_DAFT_REVISION}, got {revision}"
        )
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError("DAFT_CUTILE_SOURCE_ROOT must be clean")
    return root


def cutile_vla_image() -> modal.Image:
    """Compose the attested CUDA 13.3 engine image with LIBERO-PRO."""

    root = _checked_cutile_source()
    sys.path.insert(0, str(root))
    try:
        from scripts.modal import IMAGE_DEFAULTS, make_image
    finally:
        sys.path.pop(0)

    config = replace(
        IMAGE_DEFAULTS,
        prebuild_release_test_binaries=False,
        runtime_env={
            "HF_HOME": f"{MODEL_CACHE_DIR}/huggingface",
            "HF_HUB_CACHE": f"{MODEL_CACHE_DIR}/huggingface/hub",
            "TRANSFORMERS_CACHE": f"{MODEL_CACHE_DIR}/huggingface/hub",
            "HF_HUB_OFFLINE": "1",
            "CUTILE_DAFT_TRANSFER_COUNTERS": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        },
    )
    dependencies = (
        "daft[huggingface,video]==0.7.21",
        "psutil>=7,<8",
        *_SIM,
    )
    install = "uv pip install --system " + " ".join(
        shlex.quote(dependency) for dependency in dependencies
    )
    image = make_image(config).apt_install(*_APT).run_commands(install)
    benchmark_root = "/opt/LIBERO-PRO/libero/libero"
    return (
        image.run_commands(
            "git init /opt/LIBERO-PRO",
            f"git -C /opt/LIBERO-PRO remote add origin {LIBERO_PRO_CODE_REPOSITORY}",
            f"git -C /opt/LIBERO-PRO fetch --depth 1 origin {LIBERO_PRO_CODE_REVISION}",
            "git -C /opt/LIBERO-PRO checkout --detach FETCH_HEAD",
            "uv pip install --system --no-deps -e /opt/LIBERO-PRO",
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


CUTILE_VLA_IMAGE = cutile_vla_image() if modal.is_local() else modal.Image.debian_slim()


def _cutile_function_options() -> dict[str, Any]:
    options = _function_options(
        CUTILE_VLA_IMAGE,
        gpu="H100",
        cpu=32,
        memory=98304,
    )
    # This lane is deliberately offline and resolves both snapshots from the
    # mounted model cache, so it must not depend on an HF credential secret.
    options.pop("secrets")
    return options


def _run_cutile(
    benchmark_name: str,
    suite: str,
    tasks: list[str] | None,
    perturbations: list[str] | None,
    episodes: int,
    seed: int,
    write_video: bool,
    env_batch_size: int,
    profile: bool,
) -> dict[str, Any]:
    from physical_ai_evals import evaluate, vla_jepa_cutile

    if not 1 <= env_batch_size <= CUTILE_VLA_STATIC_BATCH_SIZE:
        raise ValueError("daft-cuTile VLA-JEPA env_batch_size must be in [1, 4]")
    benchmark = _benchmark(
        benchmark_name,
        suite,
        tasks,
        perturbations,
        episodes,
        seed,
    )
    policy = vla_jepa_cutile()

    def checkpoint() -> None:
        MODEL_CACHE.commit()
        OUTPUTS.commit()

    evaluation = evaluate(
        policy,
        benchmark,
        out=f"{OUTPUT_DIR}/evaluations/vla_jepa_cutile/{benchmark_name}",
        write_video=write_video,
        checkpoint=checkpoint,
        env_batch_size=env_batch_size,
        profile=profile,
    )
    checkpoint()
    summary = evaluation.metrics().to_pydict()
    return {
        "policy": "vla_jepa_cutile",
        "engine_revision": CUTILE_DAFT_REVISION,
        "benchmark": benchmark_name,
        "evaluation_id": evaluation.evaluation_id,
        "episodes": int(summary["episodes"][0]),
        "successes": int(summary["successes"][0] or 0),
        "success_rate": evaluation.success_rate(),
        "out_dir": str(evaluation.path),
        "profile_path": str(evaluation.path / "profiles.jsonl") if profile else None,
    }


@app.function(**_cutile_function_options())
def run_vla_jepa_cutile(
    benchmark_name: str = "libero",
    suite: str = "libero_spatial",
    tasks: list[str] | None = None,
    perturbations: list[str] | None = None,
    episodes: int = 1,
    seed: int = 7,
    write_video: bool = True,
    env_batch_size: int = 4,
    profile: bool = True,
) -> dict[str, Any]:
    return _run_cutile(
        benchmark_name,
        suite,
        tasks,
        perturbations,
        episodes,
        seed,
        write_video,
        env_batch_size,
        profile,
    )


@app.local_entrypoint()
def modal_main(
    benchmark: str = "libero",
    suite: str = "libero_spatial",
    tasks: str = "0",
    perturbations: str = "",
    episodes: int = 1,
    seed: int = 7,
    write_video: bool = True,
    env_batch_size: int = 4,
    profile: bool = True,
) -> None:
    task_list = [item.strip() for item in tasks.split(",") if item.strip()] or None
    perturbation_list = [item.strip() for item in perturbations.split(",") if item.strip()] or None
    result = run_vla_jepa_cutile.remote(
        benchmark_name=benchmark,
        suite=suite,
        tasks=task_list,
        perturbations=perturbation_list,
        episodes=episodes,
        seed=seed,
        write_video=write_video,
        env_batch_size=env_batch_size,
        profile=profile,
    )
    print(
        f"{result['successes']}/{result['episodes']} succeeded "
        f"({result['success_rate']:.3f}) -> {result['out_dir']}"
    )


__all__ = ["app", "run_vla_jepa_cutile"]
