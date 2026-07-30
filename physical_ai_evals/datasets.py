"""Revision-checked Daft pipelines for LeRobot v3 datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from daft import DataFrame


@dataclass(frozen=True)
class LeRobotSource:
    """One immutable Hugging Face LeRobot dataset or nested dataset root."""

    repo_id: str
    revision: str
    subpath: str | None = None

    def uri(self) -> str:
        _check_revision(self.repo_id, self.revision)
        uri = f"hf://datasets/{self.repo_id}"
        if self.subpath:
            uri = f"{uri}/{self.subpath.strip('/')}"
        return uri


ALOHA = LeRobotSource(
    "lerobot/aloha_mobile_shrimp",
    "6e828202059d2cc204b61ff968c232d202127a34",
)
ABC_130K = LeRobotSource(
    "lerobot/abc_130k_v3_train",
    "68651e4929d9fb00f798937b2d62617cab5c771d",
)
ABC_130K_SMOKE = LeRobotSource(
    "lerobot/abc_130k_v3_smoke",
    "b342a0ff262195d49bae3eece6e3f40c6e1dbe15",
)


def _current_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(repo_id=repo_id).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face did not return a revision for {repo_id}")
    return revision


def _check_revision(repo_id: str, revision: str) -> None:
    current = _current_revision(repo_id)
    if current != revision:
        raise RuntimeError(
            f"{repo_id} moved from expected revision {revision} to {current}; "
            "review the upstream change and update the recorded revision"
        )


def _uri(source: str | LeRobotSource, revision: str | None) -> str:
    if isinstance(source, LeRobotSource):
        if revision is not None and revision != source.revision:
            raise ValueError("revision conflicts with the LeRobotSource revision")
        return source.uri()
    if revision is None:
        return source
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", source):
        raise ValueError("revision can be used only with a Hugging Face org/repo source")
    return LeRobotSource(source, revision).uri()


def lerobot(
    source: str | LeRobotSource,
    *,
    revision: str | None = None,
    io_config=None,
    include_stats: bool = False,
    load_video_frames: str | list[str] | bool = False,
) -> DataFrame:
    """Return a lazy frame table for a local, remote, or pinned LeRobot v3 source."""
    from daft.datasets import lerobot as reader

    return reader.read(
        _uri(source, revision),
        io_config=io_config,
        include_stats=include_stats,
        load_video_frames=load_video_frames,
    )


def lerobot_episodes(
    source: str | LeRobotSource,
    *,
    revision: str | None = None,
    io_config=None,
    include_stats: bool = False,
) -> DataFrame:
    """Return a lazy episode table without decoding video."""
    from daft.datasets import lerobot as reader

    return reader.read_episodes(
        _uri(source, revision),
        io_config=io_config,
        include_stats=include_stats,
    )


def lerobot_tasks(
    source: str | LeRobotSource,
    *,
    revision: str | None = None,
    io_config=None,
) -> DataFrame:
    """Return a lazy task table."""
    from daft.datasets import lerobot as reader

    return reader.read_tasks(_uri(source, revision), io_config=io_config)


__all__ = [
    "ABC_130K",
    "ABC_130K_SMOKE",
    "ALOHA",
    "LeRobotSource",
    "lerobot",
    "lerobot_episodes",
    "lerobot_tasks",
]
