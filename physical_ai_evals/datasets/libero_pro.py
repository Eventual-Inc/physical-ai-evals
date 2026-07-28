"""Lazy catalog for the LIBERO-PRO perturbation benchmark."""

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

DEFAULT_REPO_ID = "zhouxueyang/LIBERO-Pro"
DEFAULT_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"

_CONFIG_RE = re.compile(
    r"bddl_files/(?P<configuration>\d+_[^/]+)/bddl/"
    r"(?P<base_suite>libero_(?:10|goal|object|spatial))/(?P<filename>[^/]+)\.bddl"
)
_SUITE_RE = re.compile(
    r"bddl_files/(?P<suite_variant>libero_(?:10|goal|object|spatial)"
    r"_(?P<perturbation>lan|object|swap|task))/(?P<filename>[^/]+)\.bddl"
)


def _parse_path(repo_path: str) -> tuple[str, str, str, str]:
    configured = _CONFIG_RE.fullmatch(repo_path)
    if configured is not None:
        values = configured.groupdict()
        return (
            values["base_suite"],
            values["configuration"],
            values["configuration"],
            values["filename"],
        )

    suite = _SUITE_RE.fullmatch(repo_path)
    if suite is not None:
        values = suite.groupdict()
        suffix = f"_{values['perturbation']}"
        return (
            values["suite_variant"].removesuffix(suffix),
            values["suite_variant"],
            values["perturbation"],
            values["filename"],
        )
    raise ValueError(f"Unexpected LIBERO-PRO BDDL path: {repo_path}")


def raw(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
):
    """Return one lazy row per published LIBERO-PRO BDDL task."""
    repo_files = list_repo_files(repo_id, revision)
    repo_file_set = set(repo_files)
    bddl_files = select_paths(repo_files, prefix="bddl_files/", suffix=".bddl")

    rows: dict[str, Any] = {
        "dataset": [],
        "dataset_revision": [],
        "suite": [],
        "suite_variant": [],
        "perturbation": [],
        "task_name": [],
        "bddl_path": [],
        "init_path": [],
    }
    for repo_path in bddl_files:
        base_suite, suite_variant, perturbation, task_name = _parse_path(repo_path)
        init_repo_path = f"init_files/{suite_variant}/{task_name}.pruned_init"
        if init_repo_path not in repo_file_set:
            init_repo_path = f"init_files/{base_suite}/{task_name}.pruned_init"

        rows["dataset"].append("libero_pro")
        rows["dataset_revision"].append(revision)
        rows["suite"].append(base_suite)
        rows["suite_variant"].append(suite_variant)
        rows["perturbation"].append(perturbation)
        rows["task_name"].append(task_name)
        rows["bddl_path"].append(hf_dataset_uri(repo_id, revision, repo_path))
        rows["init_path"].append(
            hf_dataset_uri(repo_id, revision, init_repo_path)
            if init_repo_path in repo_file_set
            else None
        )
    return daft.from_pydict(rows)


def instructions(tasks, *, io_config=None):
    """Read the BDDL instruction for each selected catalog row."""
    return add_instructions(tasks, io_config=io_config)
