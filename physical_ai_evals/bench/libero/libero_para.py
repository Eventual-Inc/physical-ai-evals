"""Lazy catalog for the LIBERO-Para benchmark."""

from __future__ import annotations

from daft import DataFrame, DataType, col, lit
from daft.functions import regexp_extract

from physical_ai_evals.bench.libero._bddl import add_instructions
from physical_ai_evals.core.hub import glob_repo_files, hf_uri

DEFAULT_REPO_ID = "HAI-Lab/LIBERO-Para"
DEFAULT_REVISION = "d306f66f8b441cad1155b21a3f69e440079c81c9"

_TASK = r"bddl_files/((act|obj|comp)_(.+)_eval(\d+)_ver(\d+))\.bddl"


def raw(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    io_config=None,
) -> DataFrame:
    """Return one lazy row per LIBERO-Para instruction.

    Path globbing and filename parsing are both plan nodes, so a caller's
    ``where``/``limit`` narrows the work before anything executes.
    """
    files = glob_repo_files(repo_id, revision, ("bddl_files/*.bddl",), io_config=io_config)
    path = col("path")
    return files.select(
        lit("libero_para").alias("dataset"),
        lit(revision).alias("dataset_revision"),
        lit("libero_para").alias("suite"),
        lit("libero_goal").alias("environment_suite"),
        regexp_extract(path, _TASK, 4).cast(DataType.int64()).alias("environment_task_id"),
        regexp_extract(path, _TASK, 1).alias("task_name"),
        regexp_extract(path, _TASK, 2).alias("paraphrase_type"),
        regexp_extract(path, _TASK, 3).alias("paraphrase_key"),
        regexp_extract(path, _TASK, 5).cast(DataType.int64()).alias("variant_id"),
        hf_uri(repo_id, revision, path).alias("bddl_path"),
    )


def instructions(tasks: DataFrame, *, io_config=None) -> DataFrame:
    """Read the BDDL instruction for each selected catalog row."""
    return add_instructions(tasks, io_config=io_config)
