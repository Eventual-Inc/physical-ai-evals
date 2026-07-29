"""Shared helpers for revision-checked LeRobot datasets."""

from __future__ import annotations

from physical_ai_evals.core.hub import current_repo_revision


def verified_lerobot_uri(
    repo_id: str,
    revision: str,
    *,
    subpath: str | None = None,
) -> str:
    """Return a Daft-compatible LeRobot URI after checking the Hub revision.

    Daft's LeRobot reader uses recursive globs, whose listed Hugging Face paths
    drop ``@revision`` in Daft 0.7.21. Verify the repository head before
    constructing the unpinned read plan.
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
