"""Shared helpers for revision-checked LeRobot datasets."""

from __future__ import annotations


def current_repo_revision(repo_id: str) -> str:
    """Return the current commit for a public Hugging Face dataset."""
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(repo_id=repo_id).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face did not return a revision for {repo_id}")
    return revision


def verified_lerobot_uri(
    repo_id: str,
    revision: str,
    *,
    subpath: str | None = None,
) -> str:
    """Return a Daft-compatible LeRobot URI after checking the Hub revision.

    Daft's LeRobot reader uses recursive globs. Hugging Face supports immutable
    revisions for direct ``hf://`` files, but not for those globs, so verify the
    repository head before constructing the read plan.
    """
    current = current_repo_revision(repo_id)
    if current != revision:
        raise RuntimeError(
            f"{repo_id} moved from expected revision {revision} to {current}; "
            "review the upstream change and update the recorded revision"
        )

    uri = f"hf://datasets/{repo_id}"
    if subpath is not None:
        uri = f"{uri}/{subpath.strip('/')}"
    return uri
