"""Manifest catalog and Daft reader for the EgoDex LeRobot conversion."""

from __future__ import annotations

import re
from typing import Any

import daft

from physical_ai_evals.core.hub import list_repo_files
from physical_ai_evals.datasets._lerobot import verified_lerobot_uri

DEFAULT_REPO_ID = "griffinlabs/EgoDex-LeRobot-v3.0"
DEFAULT_REVISION = "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"

_DATASET_RE = re.compile(r"(?P<split>train|test)/(?P<task_name>[^/]+)/meta/info\.json")
_SEGMENT_RE = re.compile(r"[A-Za-z0-9_.-]+")


def catalog(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    split: str | None = None,
):
    """Return one manifest-only row per EgoDex activity dataset."""
    if split is not None and split not in {"train", "test"}:
        raise ValueError("split must be 'train', 'test', or None")

    rows: dict[str, Any] = {
        "dataset": [],
        "dataset_revision": [],
        "split": [],
        "task_name": [],
        "dataset_uri": [],
    }
    for repo_path in sorted(list_repo_files(repo_id, revision)):
        match = _DATASET_RE.fullmatch(repo_path)
        if match is None:
            continue
        values = match.groupdict()
        if split is not None and values["split"] != split:
            continue
        subpath = f"{values['split']}/{values['task_name']}"
        rows["dataset"].append("egodex")
        rows["dataset_revision"].append(revision)
        rows["split"].append(values["split"])
        rows["task_name"].append(values["task_name"])
        rows["dataset_uri"].append(f"hf://datasets/{repo_id}/{subpath}")
    return daft.from_pydict(rows)


def raw(
    task_name: str,
    *,
    split: str = "train",
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    io_config=None,
    include_stats: bool = False,
    load_video_frames: str | list[str] | bool = False,
):
    """Return a lazy frame-level DataFrame for one EgoDex activity."""
    from daft.datasets import lerobot

    uri = _activity_uri(repo_id, revision, split, task_name)
    return lerobot.read(
        uri,
        io_config=io_config,
        include_stats=include_stats,
        load_video_frames=load_video_frames,
    )


def episodes(
    task_name: str,
    *,
    split: str = "train",
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    io_config=None,
    include_stats: bool = False,
):
    """Return lazy episode metadata for one EgoDex activity."""
    from daft.datasets import lerobot

    uri = _activity_uri(repo_id, revision, split, task_name)
    return lerobot.read_episodes(uri, io_config=io_config, include_stats=include_stats)


def tasks(
    task_name: str,
    *,
    split: str = "train",
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    io_config=None,
):
    """Return LeRobot task metadata for one EgoDex activity."""
    from daft.datasets import lerobot

    uri = _activity_uri(repo_id, revision, split, task_name)
    return lerobot.read_tasks(uri, io_config=io_config)


def _activity_uri(repo_id: str, revision: str, split: str, task_name: str) -> str:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if _SEGMENT_RE.fullmatch(task_name) is None:
        raise ValueError("task_name must be one path-safe activity name from catalog()")
    return verified_lerobot_uri(repo_id, revision, subpath=f"{split}/{task_name}")
