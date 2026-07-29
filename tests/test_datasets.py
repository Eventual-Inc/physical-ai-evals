"""Daft LeRobot reader routing and immutable source tests."""

from __future__ import annotations

import importlib

import daft
import pytest

from physical_ai_evals.datasets import (
    ALOHA,
    LeRobotSource,
    egodex,
    egodex_catalog,
    lerobot,
    lerobot_episodes,
    lerobot_tasks,
)

datasets = importlib.import_module("physical_ai_evals.datasets")


def test_source_verifies_revision_and_builds_nested_uri(monkeypatch):
    monkeypatch.setattr(datasets, "_current_revision", lambda _repo: "abc123")
    source = LeRobotSource("org/repo", "abc123", "train/task")

    assert source.uri() == "hf://datasets/org/repo/train/task"

    monkeypatch.setattr(datasets, "_current_revision", lambda _repo: "moved")
    with pytest.raises(RuntimeError, match="moved from expected"):
        source.uri()


def test_readers_are_thin_daft_pipelines(monkeypatch):
    monkeypatch.setattr(datasets, "_current_revision", lambda _repo: ALOHA.revision)
    calls = []

    class Reader:
        @staticmethod
        def read(uri, **kwargs):
            calls.append(("read", uri, kwargs))
            return daft.from_pydict({"frame_index": [0]})

        @staticmethod
        def read_episodes(uri, **kwargs):
            calls.append(("episodes", uri, kwargs))
            return daft.from_pydict({"episode_index": [0]})

        @staticmethod
        def read_tasks(uri, **kwargs):
            calls.append(("tasks", uri, kwargs))
            return daft.from_pydict({"task_index": [0]})

    monkeypatch.setattr(daft.datasets, "lerobot", Reader)

    assert lerobot(ALOHA).column_names == ["frame_index"]
    assert lerobot_episodes(ALOHA).column_names == ["episode_index"]
    assert lerobot_tasks(ALOHA).column_names == ["task_index"]
    assert [call[0] for call in calls] == ["read", "episodes", "tasks"]
    assert {call[1] for call in calls} == {"hf://datasets/lerobot/aloha_mobile_shrimp"}


def test_egodex_source_validates_path_segments():
    source = egodex("add_remove_lid", split="test")
    assert source.subpath == "test/add_remove_lid"

    with pytest.raises(ValueError, match="split"):
        egodex("task", split="validation")
    with pytest.raises(ValueError, match="path-safe"):
        egodex("../escape")


def test_egodex_catalog_is_expression_based(monkeypatch):
    monkeypatch.setattr(datasets, "_check_revision", lambda *_args: None)
    root = "hf://datasets/griffinlabs/EgoDex-LeRobot-v3.0"
    monkeypatch.setattr(
        daft,
        "from_glob_path",
        lambda *_args, **_kwargs: daft.from_pydict(
            {
                "path": [
                    f"{root}/test/add_remove_lid/meta/info.json",
                    f"{root}/train/write/meta/info.json",
                ]
            }
        ),
    )

    data = egodex_catalog().sort(["split", "task_name"]).to_pydict()

    assert data["split"] == ["test", "train"]
    assert data["task_name"] == ["add_remove_lid", "write"]
    assert data["dataset_uri"] == [
        f"{root}/test/add_remove_lid",
        f"{root}/train/write",
    ]
