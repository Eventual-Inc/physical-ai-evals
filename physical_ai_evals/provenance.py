"""Evaluation identity and atomic manifest persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from physical_ai_evals.schema import SCHEMA_VERSION


def runtime_provenance(distributions: Sequence[str]) -> dict[str, Any]:
    """Record the interpreter, platform, package versions, and visible GPU."""
    versions: dict[str, str | None] = {}
    for distribution in sorted(set(distributions)):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None

    gpu: dict[str, Any] = {"available": False}
    try:
        import torch
        from torch import version as torch_version

        available = torch.cuda.is_available()
        gpu = {
            "available": available,
            "cuda": getattr(torch_version, "cuda", None),
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0) if available else None,
        }
    except (ImportError, RuntimeError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "gpu": gpu,
    }


def implementation_fingerprint() -> str:
    """Hash every source file that can affect an evaluation trace."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode())
    for name in (
        "schema.py",
        "provenance.py",
        "policy.py",
        "rollout.py",
        "libero.py",
        "geometry.py",
    ):
        path = root / name
        if not path.is_file():
            # Missing source must fail closed. Silently hashing "<missing>"
            # changes evaluation IDs and can make an unsafe run look resumable.
            raise RuntimeError(f"implementation fingerprint source is missing: {path}")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluation_manifest(
    *,
    policy: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    rollouts_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Return the deterministic evaluation ID and its complete manifest config."""
    config = {
        "schema_version": SCHEMA_VERSION,
        "policy": dict(policy),
        "benchmark": {**dict(benchmark), "rollouts_sha256": rollouts_sha256},
        "implementation_sha256": implementation_fingerprint(),
        "runtime": runtime_provenance(
            ("daft", "numpy", "torch", "libero", "transformers", "lerobot")
        ),
    }
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16], config


def write_manifest(
    root: Path,
    evaluation_id: str,
    config: Mapping[str, Any],
) -> None:
    """Atomically create a manifest and reject namespace/config conflicts."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    payload = {
        "evaluation_id": evaluation_id,
        "config": dict(config),
    }
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError(
                f"evaluation manifest conflict at {path}; choose a new output root"
            )
        return

    fd, temporary_name = tempfile.mkstemp(
        dir=root,
        prefix=".manifest.",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "evaluation_manifest",
    "implementation_fingerprint",
    "runtime_provenance",
    "write_manifest",
]
