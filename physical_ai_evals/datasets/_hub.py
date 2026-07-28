from __future__ import annotations

from collections.abc import Iterable


def list_repo_files(repo_id: str, revision: str) -> list[str]:
    """Query a pinned Hugging Face dataset manifest without fetching payloads."""
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )


def hf_dataset_uri(repo_id: str, revision: str, repo_path: str) -> str:
    return f"hf://datasets/{repo_id}@{revision}/{repo_path}"


def select_paths(files: Iterable[str], *, prefix: str, suffix: str) -> list[str]:
    return sorted(path for path in files if path.startswith(prefix) and path.endswith(suffix))


def add_instructions(tasks, *, io_config=None):
    """Add BDDL language instructions, reading only rows retained by the query plan."""
    from daft import col
    from daft.datatype import DataType
    from daft.functions import regexp_extract

    content = col("bddl_path").download(io_config=io_config).cast(DataType.string)
    instruction = regexp_extract(content, r"\(:language\s+([^\r\n)]+)", 1)
    return tasks.with_column("instruction", instruction)
