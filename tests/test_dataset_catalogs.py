from __future__ import annotations

import pytest

from physical_ai_evals.datasets import _lerobot, abc, aloha, egodex


def test_revision_check_rejects_upstream_drift(monkeypatch):
    monkeypatch.setattr(_lerobot, "current_repo_revision", lambda _repo_id: "new-head")

    with pytest.raises(RuntimeError, match="moved from expected revision"):
        _lerobot.verified_lerobot_uri("org/data", "recorded-head")


def test_aloha_raw_uses_revision_checked_native_reader(monkeypatch):
    from daft.datasets import lerobot

    calls = {}
    monkeypatch.setattr(
        aloha,
        "verified_lerobot_uri",
        lambda repo_id, revision: f"verified://{repo_id}@{revision}",
    )

    def fake_read(uri, **kwargs):
        calls["uri"] = uri
        calls["kwargs"] = kwargs
        return "aloha-frame-plan"

    monkeypatch.setattr(lerobot, "read", fake_read)

    result = aloha.raw(load_video_frames="observation.images.cam_high")

    assert result == "aloha-frame-plan"
    assert calls["uri"].endswith(f"@{aloha.DEFAULT_REVISION}")
    assert calls["kwargs"]["load_video_frames"] == "observation.images.cam_high"


def test_abc_episode_reader_can_target_smoke_dataset(monkeypatch):
    from daft.datasets import lerobot

    calls = {}
    monkeypatch.setattr(
        abc,
        "verified_lerobot_uri",
        lambda repo_id, revision: f"verified://{repo_id}@{revision}",
    )

    def fake_read_episodes(uri, **kwargs):
        calls["uri"] = uri
        calls["kwargs"] = kwargs
        return "abc-episode-plan"

    monkeypatch.setattr(lerobot, "read_episodes", fake_read_episodes)

    result = abc.episodes(
        repo_id=abc.SMOKE_REPO_ID,
        revision=abc.SMOKE_REVISION,
        include_stats=True,
    )

    assert result == "abc-episode-plan"
    assert abc.SMOKE_REPO_ID in calls["uri"]
    assert calls["kwargs"]["include_stats"] is True


def test_egodex_catalog_lists_nested_activity_datasets(monkeypatch):
    monkeypatch.setattr(
        egodex,
        "list_repo_files",
        lambda _repo_id, _revision: [
            "README.md",
            "test/add_remove_lid/meta/info.json",
            "test/add_remove_lid/meta/tasks.parquet",
            "train/write/meta/info.json",
        ],
    )

    data = egodex.catalog("org/egodex", "abc123", split="test").to_pydict()

    assert data == {
        "dataset": ["egodex"],
        "dataset_revision": ["abc123"],
        "split": ["test"],
        "task_name": ["add_remove_lid"],
        "dataset_uri": ["hf://datasets/org/egodex/test/add_remove_lid"],
    }


def test_egodex_reader_uses_selected_activity(monkeypatch):
    from daft.datasets import lerobot

    calls = {}
    monkeypatch.setattr(
        egodex,
        "verified_lerobot_uri",
        lambda repo_id, revision, subpath: f"verified://{repo_id}@{revision}/{subpath}",
    )

    def fake_read_episodes(uri, **kwargs):
        calls["uri"] = uri
        return "egodex-episode-plan"

    monkeypatch.setattr(lerobot, "read_episodes", fake_read_episodes)

    result = egodex.episodes("add_remove_lid", split="test")

    assert result == "egodex-episode-plan"
    assert calls["uri"].endswith("/test/add_remove_lid")
