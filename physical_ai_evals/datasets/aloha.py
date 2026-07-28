"""Revision-checked Daft reader for ALOHA LeRobot datasets."""

from __future__ import annotations

from physical_ai_evals.datasets._lerobot import verified_lerobot_uri

DEFAULT_REPO_ID = "lerobot/aloha_mobile_shrimp"
DEFAULT_REVISION = "6e828202059d2cc204b61ff968c232d202127a34"


def raw(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    io_config=None,
    include_stats: bool = False,
    load_video_frames: str | list[str] | bool = False,
):
    """Return a lazy frame-level DataFrame for an ALOHA LeRobot v3 dataset."""
    from daft.datasets import lerobot

    uri = verified_lerobot_uri(repo_id, revision)
    return lerobot.read(
        uri,
        io_config=io_config,
        include_stats=include_stats,
        load_video_frames=load_video_frames,
    )


def episodes(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    io_config=None,
    include_stats: bool = False,
):
    """Return a lazy episode-level DataFrame without decoding video."""
    from daft.datasets import lerobot

    uri = verified_lerobot_uri(repo_id, revision)
    return lerobot.read_episodes(uri, io_config=io_config, include_stats=include_stats)


def tasks(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    io_config=None,
):
    """Return the dataset's task metadata."""
    from daft.datasets import lerobot

    uri = verified_lerobot_uri(repo_id, revision)
    return lerobot.read_tasks(uri, io_config=io_config)
