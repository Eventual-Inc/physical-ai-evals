"""Shared Modal sweep configuration, spec expansion, and safe resume logic."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EVALUATION_PROTOCOL_VERSION = "rollout-v2"
OPENVLA_REVISIONS: dict[str, str] = {
    "openvla/openvla-7b-finetuned-libero-spatial": (
        "962318cec55ac10993ff0f5f43eda9a270b4c873"
    ),
    "openvla/openvla-7b-finetuned-libero-object": (
        "287d6cfdf12d07b1449505f66d9bf3550257e9b3"
    ),
    "openvla/openvla-7b-finetuned-libero-goal": (
        "fa5ae1e7509348889295bba8e08621d8b55e9baf"
    ),
    "openvla/openvla-7b-finetuned-libero-10": (
        "80970322773f81baa2e22fe495d0487b93a05cfa"
    ),
}
VLA_JEPA_REVISION = "735d9f692981e286ade093b5046627eda876e5d0"
VLA_JEPA_MODEL_ID = "lerobot/VLA-JEPA-LIBERO"
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def resolve_openvla_config(
    suites: Sequence[str],
    model_id: str | None = None,
    unnorm_key: str | None = None,
    model_revision: str | None = None,
) -> tuple[str, str, str]:
    """Resolve one OpenVLA checkpoint and fail closed on ambiguous/incompatible runs.

    A worker loads one policy, so a run may contain exactly one LIBERO suite. Known
    OpenVLA checkpoints must match that suite. A custom checkpoint is accepted only
    when the caller explicitly declares its matching suite via ``unnorm_key``.
    """
    from harness.policy.openvla import LIBERO_CHECKPOINTS

    unique_suites = tuple(dict.fromkeys(suites))
    if len(unique_suites) != 1:
        raise ValueError(
            "OpenVLA requires exactly one suite per run so the checkpoint and action "
            f"normalization cannot be mixed; got {list(unique_suites)!r}"
        )
    suite = unique_suites[0]
    if suite not in LIBERO_CHECKPOINTS:
        raise ValueError(
            f"no registered OpenVLA checkpoint for suite {suite!r}; "
            f"choose one of {sorted(LIBERO_CHECKPOINTS)!r}"
        )

    expected_model, expected_key = LIBERO_CHECKPOINTS[suite]
    known_model_suites = {mid: registered_suite for registered_suite, (mid, _key) in LIBERO_CHECKPOINTS.items()}
    resolved_model = model_id or expected_model

    if resolved_model in known_model_suites:
        checkpoint_suite = known_model_suites[resolved_model]
        if checkpoint_suite != suite:
            raise ValueError(
                f"OpenVLA checkpoint {resolved_model!r} is registered for {checkpoint_suite!r}, "
                f"not requested suite {suite!r}"
            )
    elif unnorm_key != suite:
        raise ValueError(
            f"custom OpenVLA checkpoint {resolved_model!r} requires an explicit "
            f"unnorm_key={suite!r} compatibility declaration"
        )

    resolved_key = unnorm_key or expected_key
    if resolved_key != suite:
        raise ValueError(
            f"OpenVLA unnorm_key {resolved_key!r} does not match requested suite {suite!r}"
        )
    resolved_revision = model_revision or OPENVLA_REVISIONS.get(resolved_model)
    if resolved_revision is None:
        raise ValueError(
            f"custom OpenVLA checkpoint {resolved_model!r} requires --model-revision "
            "with an immutable 40-character commit SHA"
        )
    if not _COMMIT_SHA.fullmatch(resolved_revision):
        raise ValueError(
            f"OpenVLA model revision must be an immutable 40-character commit SHA, "
            f"got {resolved_revision!r}"
        )
    return resolved_model, resolved_revision, resolved_key


def immutable_model_id(model_id: str, revision: str) -> str:
    """Canonical model identity stored in output rows and evaluation manifests."""
    return f"{model_id}@{revision}"


def resolve_vla_jepa_config(
    model_id: str | None = None,
    model_revision: str | None = None,
) -> tuple[str, str]:
    """Resolve VLA-JEPA to an immutable Hugging Face snapshot."""
    resolved_model = model_id or VLA_JEPA_MODEL_ID
    resolved_revision = model_revision or (
        VLA_JEPA_REVISION if resolved_model == VLA_JEPA_MODEL_ID else None
    )
    if resolved_revision is None or not _COMMIT_SHA.fullmatch(resolved_revision):
        raise ValueError(
            f"VLA-JEPA model {resolved_model!r} requires an immutable 40-character "
            "model revision commit SHA"
        )
    return resolved_model, resolved_revision


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"evaluation config contains a non-serializable value: {value!r}")


def implementation_fingerprint(policy_type: str) -> str:
    """Hash rollout implementation sources so changed code gets a new namespace."""
    root = Path(__file__).resolve().parents[2]
    relative_paths = [
        "harness/bench/libero.py",
        "harness/cloud/rollout_udf.py",
        "harness/core/config.py",
        "harness/core/schema.py",
        "harness/core/writer.py",
        f"harness/policy/{policy_type}.py",
    ]
    digest = hashlib.sha256()
    digest.update(EVALUATION_PROTOCOL_VERSION.encode())
    for relative_path in relative_paths:
        path = root / relative_path
        digest.update(relative_path.encode())
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()[:16]


def runtime_provenance(distributions: Mapping[str, str]) -> dict[str, Any]:
    """Capture exact runtime/package/GPU provenance for an evaluation manifest."""
    versions: dict[str, str | None] = {}
    for label, distribution in sorted(distributions.items()):
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None

    gpu: dict[str, Any] = {"available": False}
    try:
        import torch

        gpu = {
            "available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except (ImportError, RuntimeError):
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "gpu": gpu,
    }


def evaluation_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a deterministic namespace ID for an evaluation-affecting config."""
    payload = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "config": _jsonable(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_evaluation_manifest(
    out_dir: str | Path,
    evaluation_id: str,
    config: Mapping[str, Any],
) -> Path:
    """Atomically persist the config behind a namespaced evaluation directory."""
    expected_id = evaluation_fingerprint(config)
    if evaluation_id != expected_id:
        raise ValueError(
            f"evaluation_id {evaluation_id!r} does not match config fingerprint {expected_id!r}"
        )
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "_evaluation.json"
    payload = {
        "evaluation_id": evaluation_id,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "config": _jsonable(config),
    }
    fd, temporary_name = tempfile.mkstemp(
        dir=directory,
        prefix="._evaluation.json.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as manifest_file:
            json.dump(payload, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary_name, manifest_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return manifest_path


def _part_is_valid(
    path: Path,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    seed: int,
    policy_type: str | None,
    model_id: str | None,
) -> bool:
    """Validate a completed episode part before treating it as resumable."""
    try:
        import pyarrow.parquet as pq

        from harness.core.writer import assert_emits_schema

        assert_emits_schema(path)
        table = pq.read_table(
            path,
            columns=[
                "episode_id",
                "suite",
                "task_id",
                "init_state_id",
                "seed",
                "policy_type",
                "model",
                "num_steps",
                "step_idx",
            ],
        )
        if table.num_rows == 0:
            return False

        expected: dict[str, object] = {
            "episode_id": f"{suite}/{task_id}/{init_state_id}/{seed}",
            "suite": suite,
            "task_id": task_id,
            "init_state_id": init_state_id,
            "seed": seed,
        }
        if policy_type is not None:
            expected["policy_type"] = policy_type
        if model_id is not None:
            expected["model"] = model_id
        for column, value in expected.items():
            if set(table.column(column).to_pylist()) != {value}:
                return False

        if set(table.column("num_steps").to_pylist()) != {table.num_rows}:
            return False
        return sorted(table.column("step_idx").to_pylist()) == list(range(table.num_rows))
    except (OSError, ValueError, TypeError, KeyError, AssertionError):
        return False


def enumerate_specs(
    suites: list[str],
    task_ids: list[int] | None,
    episodes: int,
    seed: int,
    out_dir: str | None = None,
    *,
    policy_type: str | None = None,
    model_id: str | None = None,
) -> tuple[list[str], list[int], list[int], list[int]]:
    """Expand specs, skipping only valid, matching Parquet episode parts."""
    from harness.bench.libero import libero_num_tasks

    s_col, t_col, i_col, seed_col, skipped = [], [], [], [], 0
    for suite in suites:
        tasks = task_ids if task_ids is not None else range(libero_num_tasks(suite))
        for task_id in tasks:
            for init_state_id in range(episodes):
                path = (
                    Path(out_dir) / f"{suite}__{task_id}__{init_state_id}__{seed}.parquet"
                    if out_dir
                    else None
                )
                if path is not None and path.is_file() and _part_is_valid(
                    path,
                    suite=suite,
                    task_id=int(task_id),
                    init_state_id=init_state_id,
                    seed=seed,
                    policy_type=policy_type,
                    model_id=model_id,
                ):
                    skipped += 1
                    continue
                s_col.append(suite)
                t_col.append(int(task_id))
                i_col.append(init_state_id)
                seed_col.append(seed)
    if skipped:
        print(f"[resume] skipping {skipped} validated episodes already on the volume")
    return s_col, t_col, i_col, seed_col
