"""Lazy catalog for the LIBERO-Para benchmark."""

from __future__ import annotations

import re
from typing import Any

import daft

from physical_ai_evals.datasets._hub import (
    add_instructions,
    hf_dataset_uri,
    list_repo_files,
    select_paths,
)

DEFAULT_REPO_ID = "HAI-Lab/LIBERO-Para"
DEFAULT_REVISION = "d306f66f8b441cad1155b21a3f69e440079c81c9"

_TASK_RE = re.compile(
    r"(?P<paraphrase_type>act|obj|comp)_(?P<paraphrase_key>.+)"
    r"_eval(?P<environment_task_id>\d+)_ver(?P<variant_id>\d+)\.bddl"
)


def raw(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
):
    """Return one lazy row per LIBERO-Para instruction.

    This queries the pinned Hub manifest only. BDDL contents are read by
    :func:`instructions` after callers apply filters or limits.
    """
    repo_files = list_repo_files(repo_id, revision)
    bddl_files = select_paths(repo_files, prefix="bddl_files/", suffix=".bddl")

    rows: dict[str, Any] = {
        "dataset": [],
        "dataset_revision": [],
        "suite": [],
        "environment_suite": [],
        "environment_task_id": [],
        "task_name": [],
        "paraphrase_type": [],
        "paraphrase_key": [],
        "variant_id": [],
        "bddl_path": [],
    }
    for repo_path in bddl_files:
        task_name = repo_path.rsplit("/", 1)[-1]
        match = _TASK_RE.fullmatch(task_name)
        if match is None:
            raise ValueError(f"Unexpected LIBERO-Para task filename: {task_name}")
        values = match.groupdict()
        rows["dataset"].append("libero_para")
        rows["dataset_revision"].append(revision)
        rows["suite"].append("libero_para")
        rows["environment_suite"].append("libero_goal")
        rows["environment_task_id"].append(int(values["environment_task_id"]))
        rows["task_name"].append(task_name.removesuffix(".bddl"))
        rows["paraphrase_type"].append(values["paraphrase_type"])
        rows["paraphrase_key"].append(values["paraphrase_key"])
        rows["variant_id"].append(int(values["variant_id"]))
        rows["bddl_path"].append(hf_dataset_uri(repo_id, revision, repo_path))
    return daft.from_pydict(rows)


def instructions(tasks, *, io_config=None):
    """Read the BDDL instruction for each selected catalog row."""
    return add_instructions(tasks, io_config=io_config)
