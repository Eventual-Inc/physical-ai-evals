"""HDF5 ingest tests — robomimic/LIBERO demos.

Self-contained: synthetic h5py fixtures, no real datasets / GPU / MP4 decode. The adapter
routes through ``Episode.to_step_rows`` -> ``ROLLOUT_SCHEMA``, so these assert both the
parsing and that ingested data lands on the same canonical schema the rollout path emits.
"""

from __future__ import annotations

import json

import daft
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from physical_ai_evals.core.writer import assert_emits_schema, write_episode, write_rows
from physical_ai_evals.ingest.hdf5 import Hdf5Ingestor

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


# ----------------------------------------------------------------------- robomimic/LIBERO

def test_robomimic_integer_sort(tmp_path):
    _write_robomimic(tmp_path / "demos.hdf5")  # demo_0, demo_10, demo_2 on disk
    eps = list(Hdf5Ingestor().load(str(tmp_path / "demos.hdf5")))
    assert [int(e.episode_id.rsplit("/", 1)[1]) for e in eps] == [0, 2, 10]


def test_robomimic_uses_daft_hdf5_reader(tmp_path, monkeypatch):
    from daft.file import Hdf5File as RealHdf5File

    calls = []

    class TrackingHdf5File:
        def __init__(self, path, io_config=None):
            calls.append(("init", path))
            self._inner = RealHdf5File(path, io_config=io_config)

        def attrs(self, path):
            calls.append(("attrs", path))
            return self._inner.attrs(path)

        def keys(self, path):
            calls.append(("keys", path))
            return self._inner.keys(path)

        def metadata(self, path):
            calls.append(("metadata", path))
            return self._inner.metadata(path)

        def read(self, paths):
            calls.append(("read", paths))
            return self._inner.read(paths)

    _write_robomimic(
        tmp_path / "demos.hdf5",
        demos=[("demo_0", 3, True)],
    )
    monkeypatch.setattr("daft.file.Hdf5File", TrackingHdf5File)

    episodes = list(Hdf5Ingestor().load(str(tmp_path / "demos.hdf5")))

    assert len(episodes) == 1
    assert ("metadata", "data/demo_0") in calls
    reads = [paths for method, paths in calls if method == "read"]
    assert len(reads) == 1
    assert "data/demo_0/actions" in reads[0]
    assert not any("image" in path or "rgb" in path for path in reads[0])


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
    np.testing.assert_array_equal(rows[0]["state"][3:6], [0.0, 0.0, 0.0])
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
    out = write_rows(rows, tmp_path / "raw.parquet")
    assert_emits_schema(out)
    data = daft.read_parquet(str(out)).select("state", "eef_pos").to_pydict()
    assert data["state"] == [None] * 4
    assert data["eef_pos"] == [None] * 4


# --------------------------------------------------------- cross-file canonical glob

def test_failure_filter_across_files(tmp_path):
    """A failing demo and a successful demo from different files written to one dir; the
    wedge query filter(success==False) selects exactly the failing episode's steps."""
    _write_robomimic(tmp_path / "fail.hdf5", demos=[("demo_0", 3, False)])
    fail_ep = next(iter(Hdf5Ingestor().load(str(tmp_path / "fail.hdf5"))))
    _write_robomimic(tmp_path / "pass.hdf5", demos=[("demo_0", 2, True)])
    pass_ep = next(iter(Hdf5Ingestor().load(str(tmp_path / "pass.hdf5"))))

    out_dir = tmp_path / "rollouts"
    write_episode(fail_ep, out_dir, run_id="t")
    write_episode(pass_ep, out_dir, run_id="t")

    failures = (
        daft.read_parquet(str(out_dir / "*.parquet"))
        .where(daft.col("success") == False)
        .select("episode_id")
        .to_pydict()
    )
    assert len(failures["episode_id"]) == 3
    assert set(failures["episode_id"]) == {fail_ep.episode_id}
