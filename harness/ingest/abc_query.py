"""Lightweight ABC-130K Hugging Face queries.

These helpers inspect repo metadata and file trees only; they do not download MCAP
payloads. They are intentionally stdlib-only so ``harness abc-query`` works before any
dataset-specific extras are installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ID = "XDOF/ABC-130k"
API_ROOT = "https://huggingface.co/api/datasets"


@dataclass(frozen=True)
class TreeEntry:
    path: str
    type: str
    size: int = 0

    @property
    def name(self) -> str:
        return self.path.rstrip("/").rsplit("/", 1)[-1]


def _token_from_env(token_env: str = "HF_TOKEN") -> str | None:
    return os.environ.get(token_env) or None


def _get_json(url: str, *, token: str | None = None) -> Any:
    headers = {"User-Agent": "vla-jepa-harness/abc-query"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HF API request failed ({err.code}) for {url}: {body}") from err


def dataset_info(*, token: str | None = None) -> dict:
    """Return Hugging Face dataset metadata for ABC-130K."""
    return _get_json(f"{API_ROOT}/{REPO_ID}", token=token)


def tree(path: str, *, token: str | None = None, recursive: bool = False) -> list[TreeEntry]:
    """List a Hugging Face dataset tree path without downloading files."""
    encoded = quote(path.strip("/"), safe="/")
    rows = _get_json(
        f"{API_ROOT}/{REPO_ID}/tree/main/{encoded}?recursive={str(recursive).lower()}",
        token=token,
    )
    return [TreeEntry(path=r["path"], type=r["type"], size=int(r.get("size") or 0)) for r in rows]


def list_tasks(
    *,
    split: str = "train",
    contains: str | None = None,
    token: str | None = None,
) -> list[str]:
    """List task directory names for a split."""
    entries = tree(f"data/{split}", token=token)
    tasks = [e.name for e in entries if e.type == "directory"]
    if contains:
        needle = contains.lower()
        tasks = [t for t in tasks if needle in t.lower()]
    return tasks


def list_episodes(
    task: str,
    *,
    split: str = "train",
    token: str | None = None,
) -> list[str]:
    """List episode directory names for one task."""
    task = task.removeprefix("task=")
    entries = tree(f"data/{split}/{task}", token=token)
    return [e.name for e in entries if e.type == "directory"]


def list_episode_files(
    task: str,
    episode: str,
    *,
    split: str = "train",
    token: str | None = None,
) -> list[TreeEntry]:
    """List files in one episode directory, including sizes when the API exposes them."""
    task = task.removeprefix("task=")
    return tree(f"data/{split}/{task}/{episode}", token=token)


def format_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024


def token_from_env(token_env: str = "HF_TOKEN") -> str | None:
    """Public wrapper used by the CLI."""
    return _token_from_env(token_env)
