"""Revision-checked Daft reader for the public ABC-130K LeRobot conversion."""

from __future__ import annotations

from physical_ai_evals.datasets._lerobot import verified_lerobot_uri

DEFAULT_REPO_ID = "lerobot/abc_130k_v3_train"
DEFAULT_REVISION = "68651e4929d9fb00f798937b2d62617cab5c771d"

SMOKE_REPO_ID = "lerobot/abc_130k_v3_smoke"
SMOKE_REVISION = "b342a0ff262195d49bae3eece6e3f40c6e1dbe15"

ORIGINAL_REPO_ID = "XDOF/ABC-130k"
ORIGINAL_REVISION = "29136bc9b9e38d320b00ffcddbbe4cd0e3278c58"


def raw(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    io_config=None,
    include_stats: bool = False,
    load_video_frames: str | list[str] | bool = False,
):
    """Return a lazy frame-level DataFrame for the ABC-130K LeRobot v3 data."""
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
