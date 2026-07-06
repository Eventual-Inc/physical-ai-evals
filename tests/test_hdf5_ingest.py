"""HDF5 ingest tests — robomimic/LIBERO demos and raw-DROID trajectory.h5.

Self-contained: synthetic h5py fixtures, no real datasets / GCS / GPU / MP4 decode. Every
adapter routes through ``Episode.to_step_rows`` -> ``ROLLOUT_SCHEMA``, so these assert both
the per-format parsing AND that ingested data lands on the SAME canonical schema the rollout
path emits (the property the strict schema buys vs. Daft's per-dataset passthrough scans).
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pytest

h5py = pytest.importorskip("h5py")

from harness.ingest.droid import DroidIngestor, parse_trajectory_h5
from harness.ingest.hdf5 import Hdf5Ingestor
from harness.writer import assert_emits_schema, write_episode, write_rows

# --------------------------------------------------------------------------- fixtures

def _write_demo(parent, name, *, n, gripper=0.5, native=False,
                with_obs=True, with_rewards=True, with_dones=True, success=True):
    g = parent.create_group(name)
    actions = np.zeros((n, 7), dtype=np.float32)
    actions[:, -1] = gripper
    for i in range(n):
        actions[i, :6] = 0.01 * i
    g.create_dataset("actions", data=actions)
    if with_rewards:
        rew = np.zeros(n, dtype=np.float32)
        if success:
            rew[-1] = 1.0
        g.create_dataset("rewards", data=rew)
    if with_dones:
        dn = np.zeros(n, dtype=np.int64)
        if success:
            dn[-1] = 1
        g.create_dataset("dones", data=dn)
    if with_obs:
        obs = g.create_group("obs")
        eef = np.tile(np.array([0.1, 0.2, 0.3], np.float32), (n, 1))
        gqpos = np.tile(np.array([0.04, -0.04], np.float32), (n, 1))  # diff -> 0.08
        img = np.zeros((n, 8, 8, 3), np.uint8)
        if native:
            obs.create_dataset("agentview_rgb", data=img)
            obs.create_dataset("eye_in_hand_rgb", data=img)
            obs.create_dataset("ee_pos", data=eef)
            obs.create_dataset("ee_ori", data=np.zeros((n, 3), np.float32))  # axis-angle
            obs.create_dataset("gripper_states", data=gqpos)
        else:
            obs.create_dataset("agentview_image", data=img)
            obs.create_dataset("robot0_eye_in_hand_image", data=img)
            obs.create_dataset("robot0_eef_pos", data=eef)
            obs.create_dataset("robot0_eef_quat",
                               data=np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1)))
            obs.create_dataset("robot0_gripper_qpos", data=gqpos)


def _write_robomimic(path, *, native=False, with_obs=True, with_rewards=True,
                     with_dones=True, problem_info=True, demos=None):
    if demos is None:
        demos = [("demo_0", 3, True), ("demo_10", 5, True), ("demo_2", 4, True)]
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        if problem_info:
            data.attrs["problem_info"] = json.dumps({
                "problem_name": "put_bowl",
                "domain_name": "libero_goal",
                "language_instruction": "put the bowl on the plate",
            })
            data.attrs["bddl_file_name"] = "/x/put_bowl.bddl"
        for name, n, success in demos:
            _write_demo(data, name, n=n, native=native, with_obs=with_obs,
                        with_rewards=with_rewards, with_dones=with_dones, success=success)


def _write_droid_trajectory(path, *, T=4, action_group="action_dict", with_joints=True):
    with h5py.File(path, "w") as f:
        ag = f.create_group(action_group)
        cart = np.arange(T * 6, dtype=np.float64).reshape(T, 6)
        grip = np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(T, 1)
        ag.create_dataset("cartesian_position", data=cart)
        ag.create_dataset("gripper_position", data=grip)
        rs = f.create_group("observation").create_group("robot_state")
        rs.create_dataset("cartesian_position", data=cart + 0.001)  # measured != commanded
        rs.create_dataset("gripper_position", data=grip)
        if with_joints:
            rs.create_dataset("joint_positions", data=np.zeros((T, 7), np.float64))


# ----------------------------------------------------------------------- robomimic/LIBERO

def test_robomimic_integer_sort(tmp_path):
    _write_robomimic(tmp_path / "demos.hdf5")  # demo_0, demo_10, demo_2 on disk
    eps = list(Hdf5Ingestor().load(str(tmp_path / "demos.hdf5")))
    assert [int(e.episode_id.rsplit("/", 1)[1]) for e in eps] == [0, 2, 10]


def test_robomimic_success_derivation(tmp_path):
    _write_robomimic(tmp_path / "d.hdf5",
                     demos=[("demo_0", 3, True), ("demo_1", 3, False)])
    eps = {int(e.episode_id.rsplit("/", 1)[1]): e for e in Hdf5Ingestor().load(str(tmp_path / "d.hdf5"))}
    assert eps[0].success is True
    assert eps[1].success is False


def test_robomimic_spine_projection(tmp_path):
    _write_robomimic(tmp_path / "d.hdf5", demos=[("demo_0", 3, True)])
    ep = next(iter(Hdf5Ingestor().load(str(tmp_path / "d.hdf5"))))
    assert ep.task_name == "put_bowl" and ep.suite == "libero_goal"

    rows = ep.to_step_rows(run_id="test")
    out = write_rows(rows, tmp_path / "ep.parquet")
    assert_emits_schema(out)  # schema parity with ROLLOUT_SCHEMA

    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert row["step_idx"] == i
        assert row["num_steps"] == 3
        assert row["instruction"] == "put the bowl on the plate"
        assert row["bddl_file"] == "/x/put_bowl.bddl"   # proves to_step_rows metadata wiring
        assert row["control_mode"] == "relative"
        assert row["source"] == "hdf5"
        assert len(row["state"]) == 8
        assert abs(row["gripper_state"] - 0.08) < 1e-5     # qpos[0]-qpos[1]
        assert abs(row["gripper_action"] - row["action"][-1]) < 1e-6  # derived == action[-1]


def test_robomimic_native_keys(tmp_path):
    _write_robomimic(tmp_path / "n.hdf5", native=True, demos=[("demo_0", 3, True)])
    ep = next(iter(Hdf5Ingestor().load(str(tmp_path / "n.hdf5"))))
    rows = ep.to_step_rows(run_id="test")
    assert len(rows[0]["state"]) == 8               # ee_ori already axis-angle, no quat conv
    assert rows[0]["state"][3:6] == [0.0, 0.0, 0.0]
    assert_emits_schema(write_rows(rows, tmp_path / "n.parquet"))


def test_robomimic_missing_obs(tmp_path):
    _write_robomimic(tmp_path / "raw.hdf5", with_obs=False, with_rewards=False,
                     with_dones=False, demos=[("demo_0", 4, True)])
    ep = next(iter(Hdf5Ingestor().load(str(tmp_path / "raw.hdf5"))))
    assert ep.success is False and ep.num_steps == 4
    rows = ep.to_step_rows(run_id="test")
    assert rows[0]["state"] is None
    assert rows[0]["eef_pos"] is None
    assert rows[0]["gripper_state"] is None
    assert rows[0]["action"] is not None
    assert_emits_schema(write_rows(rows, tmp_path / "raw.parquet"))


# ----------------------------------------------------------------------------- raw DROID

def test_droid_action_assembly(tmp_path):
    _write_droid_trajectory(tmp_path / "trajectory.h5", T=4, action_group="action_dict")
    meta = {"uuid": "abc123", "success": True, "current_task": "pick up the cup",
            "trajectory_length": 4}
    ep = parse_trajectory_h5(str(tmp_path / "trajectory.h5"), meta)
    assert ep.episode_id == "droid/abc123"
    assert ep.instruction == "pick up the cup"
    assert ep.success is True and ep.num_steps == 4

    rows = ep.to_step_rows(run_id="test")
    assert rows[0]["control_mode"] == "absolute"      # DROID = absolute, not relative
    assert rows[0]["source"] == "droid"
    assert len(rows[0]["action"]) == 7                # concat(cart6, grip1)
    np.testing.assert_allclose(rows[0]["action"], [0, 1, 2, 3, 4, 5, 0.0], rtol=1e-5)
    np.testing.assert_allclose(rows[0]["eef_pos"], [0.001, 1.001, 2.001], rtol=1e-4)
    assert abs(rows[0]["gripper_state"] - 0.0) < 1e-6  # measured gripper[0]
    assert_emits_schema(write_rows(rows, tmp_path / "d.parquet"))


def test_droid_action_group_fallback(tmp_path):
    _write_droid_trajectory(tmp_path / "trajectory.h5", T=4, action_group="action")
    ep = parse_trajectory_h5(str(tmp_path / "trajectory.h5"),
                             {"uuid": "x", "success": False, "current_task": "t"})
    assert ep.num_steps == 4 and ep.success is False


def test_droid_schema_parity_media_null(tmp_path):
    _write_droid_trajectory(tmp_path / "trajectory.h5", T=3)
    ep = parse_trajectory_h5(str(tmp_path / "trajectory.h5"),
                             {"uuid": "y", "success": True, "current_task": "c"})
    rows = ep.to_step_rows(run_id="test")
    assert_emits_schema(write_rows(rows, tmp_path / "d.parquet"))
    for i, row in enumerate(rows):
        assert row["frame_path"] is None and row["wrist_path"] is None
        assert row["video_path"] is None
        assert row["reward"] is None
        assert row["done"] == (i == len(rows) - 1)


def test_droid_ingestor_dir_glob(tmp_path):
    ep_dir = tmp_path / "ep_0001"
    ep_dir.mkdir()
    _write_droid_trajectory(ep_dir / "trajectory.h5", T=3)
    (ep_dir / "metadata_ep0001.json").write_text(
        json.dumps({"uuid": "u1", "success": True, "current_task": "do x"})
    )
    eps = list(DroidIngestor().load(str(tmp_path)))
    assert len(eps) == 1
    assert eps[0].episode_id == "droid/u1" and eps[0].instruction == "do x"


# --------------------------------------------------------- cross-source canonical glob

def test_cross_source_failure_filter(tmp_path):
    """One failing HDF5 demo + one successful DROID episode written to one dir; the wedge
    query filter(success==False) selects exactly the HDF5 steps — no source-specific path."""
    _write_robomimic(tmp_path / "demos.hdf5", demos=[("demo_0", 3, False)])
    h_ep = next(iter(Hdf5Ingestor().load(str(tmp_path / "demos.hdf5"))))
    _write_droid_trajectory(tmp_path / "trajectory.h5", T=2)
    d_ep = parse_trajectory_h5(str(tmp_path / "trajectory.h5"),
                               {"uuid": "d", "success": True, "current_task": "t"})

    out_dir = tmp_path / "rollouts"
    write_episode(h_ep, out_dir, run_id="t")
    write_episode(d_ep, out_dir, run_id="t")

    table = pads.dataset(out_dir, format="parquet").to_table()   # one glob, unified schema
    failures = table.filter(pc.equal(table.column("success"), False))
    assert failures.num_rows == 3
    assert set(failures.column("source").to_pylist()) == {"hdf5"}
