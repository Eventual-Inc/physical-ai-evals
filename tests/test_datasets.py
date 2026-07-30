"""Daft LeRobot reader routing and immutable source tests."""

from __future__ import annotations

import importlib

import daft
import pytest

from physical_ai_evals.datasets import (
    ALOHA,
    LeRobotSource,
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
