"""Lazy catalog for the LIBERO-PRO perturbation benchmark."""

from __future__ import annotations

from daft import DataFrame, col, lit
from daft.functions import coalesce, format, regexp, regexp_extract, when

from physical_ai_evals.bench.libero._bddl import add_instructions
from physical_ai_evals.core.hub import glob_repo_files, hf_uri

DEFAULT_REPO_ID = "zhouxueyang/LIBERO-Pro"
DEFAULT_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"

_CONFIGURED = (
    r"bddl_files/(\d+_[^/]+)/bddl/(libero_(?:10|goal|object|spatial))/([^/]+)\.bddl"
)
_PERTURBED = (
    r"bddl_files/(libero_(?:10|goal|object|spatial))_(lan|object|swap|task)/([^/]+)\.bddl"
)


def raw(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    io_config=None,
) -> DataFrame:
    """Return one lazy row per published LIBERO-PRO BDDL task.

    The whole catalog is a Daft plan: nothing is listed, parsed, or paired until
    the caller collects. BDDL and init payloads stay unread file references.
    """
    files = glob_repo_files(
        repo_id,
        revision,
        ("bddl_files/**/*.bddl", "init_files/**/*.pruned_init"),
        io_config=io_config,
    )
    path = col("path")
    configured = regexp(path, _CONFIGURED)

    tasks = files.where(path.startswith("bddl_files/")).select(
        lit("libero_pro").alias("dataset"),
        lit(revision).alias("dataset_revision"),
        when(configured, regexp_extract(path, _CONFIGURED, 2))
        .otherwise(regexp_extract(path, _PERTURBED, 1))
        .alias("suite"),
        when(configured, regexp_extract(path, _CONFIGURED, 1))
        .otherwise(
            format(
                "{}_{}",
                regexp_extract(path, _PERTURBED, 1),
                regexp_extract(path, _PERTURBED, 2),
            )
        )
        .alias("suite_variant"),
        when(configured, regexp_extract(path, _CONFIGURED, 1))
        .otherwise(regexp_extract(path, _PERTURBED, 2))
        .alias("perturbation"),
        when(configured, regexp_extract(path, _CONFIGURED, 3))
        .otherwise(regexp_extract(path, _PERTURBED, 3))
        .alias("task_name"),
        hf_uri(repo_id, revision, path).alias("bddl_path"),
    )

    # An init file is preferred per suite_variant and falls back to the base suite.
    inits = files.where(path.startswith("init_files/"))
    keyed = tasks.with_columns(
        {
            "_variant_key": format(
                "init_files/{}/{}.pruned_init", col("suite_variant"), col("task_name")
            ),
            "_suite_key": format(
                "init_files/{}/{}.pruned_init", col("suite"), col("task_name")
            ),
        }
    )
    for key, hit in (("_variant_key", "_variant_hit"), ("_suite_key", "_suite_hit")):
        keyed = keyed.join(
            inits.select(path.alias(key), path.alias(hit)), on=key, how="left"
        )

    return keyed.select(
        *tasks.column_names,
        hf_uri(
            repo_id, revision, coalesce(col("_variant_hit"), col("_suite_hit"))
        ).alias("init_path"),
    )


def instructions(tasks: DataFrame, *, io_config=None) -> DataFrame:
    """Read the BDDL instruction for each selected catalog row."""
    return add_instructions(tasks, io_config=io_config)
