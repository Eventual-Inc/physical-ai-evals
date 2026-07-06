"""Daft-native ingest tests — LeRobot and DROID via ``daft.datasets``.

Exercises Daft's native ``daft.datasets.{lerobot,droid}`` readers (nightly Daft; PyPI from the
release after 0.7.16) against synthetic fixtures, then asserts our adapters normalize their
output onto the canonical ``ROLLOUT_SCHEMA``. No network/GPU; the LeRobot fixture has no video
features so no MP4 decode is needed.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("daft")
h5py = pytest.importorskip("h5py")

from harness.ingest.droid import DroidIngestor
from harness.ingest.lerobot import LeRobotIngestor
from harness.writer import assert_emits_schema, write_rows

# --------------------------------------------------------------------- fixtures

def _write_lerobot_v3(root, *, codebase_version="v3.0"):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({
        "codebase_version": codebase_version, "fps": 30,
        "features": {"action": {"dtype": "float32"}, "observation.state": {"dtype": "float32"}},
        "data_path": "data", "video_path": "videos",
    }))
    pq.write_table(
        pa.table({"episode_index": [0, 1], "length": [3, 2],
                  "dataset_from_index": [0, 3], "dataset_to_index": [3, 5]}),
        root / "meta/episodes/chunk-000/file-000.parquet")
    fi = [0, 1, 2, 0, 1]
    pq.write_table(pa.table({
        "episode_index": [0, 0, 0, 1, 1], "frame_index": fi, "index": [0, 1, 2, 3, 4],
        "timestamp": [float(x) for x in fi],
        "action": [[0.1 * i] * 7 for i in range(5)],
        "observation.state": [[0.0] * 8 for _ in range(5)],
        "task_index": [0, 0, 0, 1, 1],
    }), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0, 1],
                             "task": ["put the bowl on the plate", "open the drawer"]}),
                   root / "meta/tasks.parquet")


def _write_droid_dir(root, episodes):
    for name, T, ok, task in episodes:
        d = root / name
        d.mkdir(parents=True)
        (d / f"metadata_{name}.json").write_text(json.dumps({
            "uuid": name, "success": ok, "current_task": task, "trajectory_length": T,
            "wrist_cam_serial": "", "ext1_cam_serial": "", "ext2_cam_serial": "",
        }))
        with h5py.File(d / "trajectory.h5", "w") as f:
            ag = f.create_group("action_dict")
            ag.create_dataset("cartesian_position", data=np.arange(T * 6, dtype=np.float64).reshape(T, 6))
            ag.create_dataset("gripper_position", data=np.linspace(0, 1, T, dtype=np.float64).reshape(T, 1))
            rs = f.create_group("observation").create_group("robot_state")
            rs.create_dataset("cartesian_position", data=np.zeros((T, 6), np.float64))
            rs.create_dataset("gripper_position", data=np.linspace(0, 1, T, dtype=np.float64).reshape(T, 1))
            rs.create_dataset("joint_positions", data=np.zeros((T, 7), np.float64))


# ------------------------------------------------------------------ native daft.datasets

def test_daft_ships_native_readers():
    import daft.datasets

    assert hasattr(daft.datasets.lerobot, "read")
    assert hasattr(daft.datasets.droid, "raw")


# --------------------------------------------------------------------------- LeRobot

def test_lerobot_read_via_daft(tmp_path):
    _write_lerobot_v3(tmp_path)
    eps = {e.episode_id: e for e in LeRobotIngestor().load(str(tmp_path))}
    assert set(eps) == {"lerobot/0", "lerobot/1"}
    assert eps["lerobot/0"].num_steps == 3 and eps["lerobot/1"].num_steps == 2
    assert eps["lerobot/0"].instruction == "put the bowl on the plate"   # joined via task_index
    assert eps["lerobot/1"].instruction == "open the drawer"
    assert eps["lerobot/0"].source == "lerobot"

    rows = eps["lerobot/0"].to_step_rows(run_id="t")
    assert_emits_schema(write_rows(rows, tmp_path / "out.parquet"))  # canonical schema parity
    assert [r["step_idx"] for r in rows] == [0, 1, 2]
    assert len(rows[0]["action"]) == 7
    assert rows[0]["terminal_failure"] == "unlabeled"  # LeRobot data has no success label


def test_lerobot_v3_only_guard(tmp_path):
    _write_lerobot_v3(tmp_path, codebase_version="v2.1")
    with pytest.raises(ValueError, match="v3"):
        list(LeRobotIngestor().load(str(tmp_path)))  # vendored reader rejects v2.1


def test_lerobot_limit(tmp_path):
    _write_lerobot_v3(tmp_path)
    assert len(list(LeRobotIngestor().load(str(tmp_path), limit=1))) == 1


# ----------------------------------------------------------------------------- DROID

def test_droid_raw_compose(tmp_path):
    _write_droid_dir(tmp_path, [("ep_aaa", 4, True, "pick up the cup"),
                                ("ep_bbb", 3, False, "stack the blocks")])
    eps = {e.episode_id: e for e in DroidIngestor().load(str(tmp_path))}  # use_daft -> raw()
    assert set(eps) == {"droid/ep_aaa", "droid/ep_bbb"}
    assert eps["droid/ep_aaa"].num_steps == 4 and eps["droid/ep_aaa"].success is True
    assert eps["droid/ep_bbb"].success is False
    assert eps["droid/ep_aaa"].instruction == "pick up the cup"

    rows = eps["droid/ep_aaa"].to_step_rows(run_id="t")
    assert_emits_schema(write_rows(rows, tmp_path / "out.parquet"))
    assert rows[0]["control_mode"] == "absolute"  # DROID stores pose targets, not deltas
    assert len(rows[0]["action"]) == 7


def test_droid_glob_fallback(tmp_path):
    _write_droid_dir(tmp_path, [("ep_x", 3, True, "do x")])
    eps = list(DroidIngestor().load(str(tmp_path), use_daft=False))  # no-Daft path
    assert len(eps) == 1 and eps[0].episode_id == "droid/ep_x"
