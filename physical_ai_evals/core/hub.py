"""Shared Hugging Face Hub access: revision pinning and Daft-native discovery."""

from __future__ import annotations

from collections.abc import Iterable

import daft
from daft import DataFrame, Expression, col
from daft.functions import format


def current_repo_revision(repo_id: str) -> str:
    """Return the current commit for a public Hugging Face dataset."""
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(repo_id=repo_id).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face did not return a revision for {repo_id}")
    return revision


def list_repo_files(repo_id: str, revision: str) -> list[str]:
    """Query a pinned Hugging Face dataset manifest without fetching payloads."""
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )


def check_repo_revision(repo_id: str, revision: str) -> None:
    """Fail closed when a pinned repository has moved upstream."""
    current = current_repo_revision(repo_id)
    if current != revision:
        raise RuntimeError(
            f"{repo_id} moved from expected revision {revision} to {current}; "
            "review the upstream change and update the recorded revision"
        )


def glob_repo_files(
    repo_id: str,
    revision: str,
    patterns: Iterable[str],
    *,
    io_config=None,
) -> DataFrame:
    """Discover repository files with Daft: one revision-relative ``path`` column.

    Daft 0.7.21 drops ``@revision`` from paths returned by Hugging Face glob
    listings, which prevents revision-pinned wildcard patterns from matching.
    Verify that the repository is still at the recorded revision on both sides
    of the unpinned glob, then let callers pin file URIs with :func:`hf_uri`.

    The result stays a lazy plan, so callers filter, join, and limit before
    anything is listed or read.
    """
    check_repo_revision(repo_id, revision)
    repo_root = f"hf://datasets/{repo_id}"
    glob_paths = [f"{repo_root}/{pattern.lstrip('/')}" for pattern in patterns]
    listing = daft.from_glob_path(glob_paths, io_config=io_config)
    check_repo_revision(repo_id, revision)
    # from_glob_path roots every result at the pattern prefix, so this is a substring.
    return listing.select(col("path").substr(len(repo_root) + 1).alias("path"))


def hf_uri(repo_id: str, revision: str, repo_path: Expression | str) -> Expression:
    """Pin a revision-relative repo path column to an immutable Hugging Face URI."""
    return format(f"hf://datasets/{repo_id}@{revision}/{{}}", repo_path)
