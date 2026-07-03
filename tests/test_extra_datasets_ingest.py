"""Additional dataset ingest tests: ALOHA, EgoDex, and ABC exports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from harness.ingest.abc import ABCIngestor, parse_abc_episode_dir
from harness.ingest.aloha import AlohaIngestor, parse_aloha_hdf5
from harness.ingest.egodex import EgoDexIngestor, parse_egodex_hdf5
from harness.writer import assert_emits_schema, write_rows


def _write_aloha(path, *, n=4, mobile=False):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        f.attrs["language_instruction"] = "fold the cloth"
        f.attrs["task_name"] = "fold_cloth"
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.arange(n * 14, dtype=np.float64).reshape(n, 14))
        obs.create_dataset("qvel", data=np.zeros((n, 14), dtype=np.float64))
        images = obs.create_group("images")
        images.create_dataset("cam_high", data=np.zeros((n, 8, 8, 3), dtype=np.uint8))
        images.create_dataset("cam_left_wrist", data=np.zeros((n, 8, 8, 3), dtype=np.uint8))
        f.create_dataset("action", data=np.ones((n, 14), dtype=np.float64))
        if mobile:
            f.create_dataset("base_action", data=np.full((n, 2), 0.25, dtype=np.float64))


def _write_egodex(path, *, n=3):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        f.attrs["llm_type"] = "reversible"
        f.attrs["which_llm_description"] = "2"
        f.attrs["llm_description"] = "open the container"
        f.attrs["llm_description2"] = "close the container"
        cam = f.create_group("camera")
        cam.create_dataset("intrinsic", data=np.eye(3, dtype=np.float32))
        transforms = f.create_group("transforms")
        for name, offset in (("camera", 0.0), ("leftHand", 1.0), ("rightHand", 2.0)):
            arr = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
            arr[:, :3, 3] = np.array([offset, offset + 0.1, offset + 0.2], dtype=np.float32)
            transforms.create_dataset(name, data=arr)


def _write_abc_episode(root, name="episode_abc", *, n=5):
    ep_dir = root / name
    ep_dir.mkdir(parents=True)
    states = np.arange(n * 14, dtype=np.float64).reshape(n, 14)
    actions = -states
    np.concatenate([states, actions], axis=1).tofile(ep_dir / "states_actions.bin")
    (ep_dir / "combined_camera-images-rgb.mp4").write_bytes(b"fake")
    (ep_dir / "episode_metadata.json").write_text(json.dumps({
        "task_name": "organize_the_condiment_bottles",
        "cameras": ["top", "left", "right"],
        "alignment": "fixed_clock_30hz_causal",
        "num_steps": n,
    }))
    return ep_dir


def test_aloha_hdf5_joint_actions(tmp_path):
    _write_aloha(tmp_path / "episode_0.hdf5", mobile=True)
    ep = parse_aloha_hdf5(str(tmp_path / "episode_0.hdf5"))
    assert ep.episode_id == "aloha/fold_cloth/episode_0"
    assert ep.instruction == "fold the cloth"
    assert ep.success is True

    rows = ep.to_step_rows(run_id="test")
    assert rows[0]["source"] == "aloha"
    assert rows[0]["control_mode"] == "joint"
    assert len(rows[0]["state"]) == 14
    assert len(rows[0]["action"]) == 16
    assert_emits_schema(write_rows(rows, tmp_path / "aloha.parquet"))


def test_aloha_ingestor_discovers_directory(tmp_path):
    task = tmp_path / "zip_tie"
    task.mkdir()
    _write_aloha(task / "episode_0.hdf5")
    eps = list(AlohaIngestor().load(str(tmp_path), limit=1))
    assert len(eps) == 1
    assert eps[0].task_name == "fold_cloth"


def test_egodex_hdf5_video_metadata(tmp_path):
    task = tmp_path / "close_lid"
    task.mkdir()
    h5_path = task / "0.hdf5"
    _write_egodex(h5_path)
    (task / "0.mp4").write_bytes(b"fake")

    ep = parse_egodex_hdf5(str(h5_path), task_name="close_lid")
    assert ep.episode_id == "egodex/close_lid/0"
    assert ep.instruction == "close the container"
    assert ep.success is False and ep.terminal_failure == "unlabeled"

    rows = ep.to_step_rows(run_id="test")
    assert rows[0]["source"] == "egodex"
    assert rows[0]["control_mode"] == "human_pose"
    assert rows[0]["video_path"] == str(task / "0.mp4")
    assert len(rows[0]["state"]) == 6
    assert rows[0]["action"] is None
    assert_emits_schema(write_rows(rows, tmp_path / "egodex.parquet"))


def test_egodex_ingestor_discovers_directory(tmp_path):
    task = tmp_path / "pick"
    task.mkdir()
    _write_egodex(task / "0.hdf5")
    eps = list(EgoDexIngestor().load(str(tmp_path)))
    assert len(eps) == 1
    assert eps[0].task_name == "pick"


def test_abc_exported_episode(tmp_path):
    ep_dir = _write_abc_episode(tmp_path / "train_real")
    ep = parse_abc_episode_dir(ep_dir, root=tmp_path)
    assert ep.episode_id == "abc/train_real/episode_abc"
    assert ep.instruction == "organize_the_condiment_bottles"
    assert ep.success is True

    rows = ep.to_step_rows(run_id="test")
    assert rows[0]["source"] == "abc"
    assert rows[0]["control_mode"] == "joint"
    assert rows[0]["video_path"] == str(ep_dir / "combined_camera-images-rgb.mp4")
    assert len(rows[0]["state"]) == 14
    assert len(rows[0]["action"]) == 14
    assert_emits_schema(write_rows(rows, tmp_path / "abc.parquet"))


def test_abc_ingestor_discovers_nested_export(tmp_path):
    _write_abc_episode(tmp_path / "train_real", "episode_1", n=2)
    _write_abc_episode(tmp_path / "val_real", "episode_2", n=3)
    eps = list(ABCIngestor().load(str(tmp_path), limit=1))
    assert len(eps) == 1
    assert eps[0].num_steps == 2
