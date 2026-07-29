"""Rollout loop + RolloutWriter tests — closed loop driven by a fake env + fake policy.

No LIBERO sim, no GPU: ``run_episode``/``run_sweep`` only touch the ``Policy`` ABC and a
gym-style env, so a fake of each exercises the full loop -> ``RolloutWriter`` -> ROLLOUT_SCHEMA
parquet. Frame/video capture (PNG/mp4) needs pillow/ffmpeg and is left off here (asserted null).
"""

from __future__ import annotations

import daft
import numpy as np
import pytest

from physical_ai_evals.bench.libero import run_episode, run_sweep
from physical_ai_evals.core.config import RolloutConfig
from physical_ai_evals.core.writer import RolloutWriter, assert_emits_schema
from physical_ai_evals.policy.base import Policy


class _FakeEnv:
    """Succeeds after ``n_steps`` policy steps. Mirrors the LIBERO env surface the runner
    touches: reset / set_init_state / step / check_success / close."""

    def __init__(self, n_steps=2):
        self.n_steps = n_steps
        self.t = 0
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1   # load-bearing per episode (robosuite horizon/done clearing)
        self.t = 0
        return self._obs()

    def set_init_state(self, _state):
        self.t = 0
        return self._obs()

    def step(self, _action):
        self.t += 1
        done = self.t >= self.n_steps
        return self._obs(), (1.0 if done else 0.0), done, {}

    def check_success(self):
        return self.t >= self.n_steps

    def close(self):
        pass

    def _obs(self):
        return {
            "agentview_image": np.zeros((6, 6, 3), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((6, 6, 3), np.uint8),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3 + 0.01 * self.t], np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
            "robot0_gripper_qpos": np.array([0.04, -0.04], np.float32),
        }


class _FakeTask:
    language = "put the bowl on the plate"
    name = "put_bowl"
    bddl_file = "/x/put_bowl.bddl"


class _FakePolicy(Policy):
    control_mode = "relative"

    def reset(self, instruction):
        self._instruction = instruction

    def act(self, observation):
        return np.full(7, 0.2, np.float32)


def _cfg(tmp_path, **kw):
    return RolloutConfig(
        policy_type="openvla",
        suites=("libero_goal",),
        task_ids=(0,),
        n_episodes_per_task=2,
        seed=0,                      # pin: episode-id assertions below embed the seed
        num_steps_wait=0,            # fake env counts every step; skip settle for the test
        write_frames=False,
        write_video=False,
        out_dir=tmp_path / "rollouts",
        frames_dir=tmp_path / "frames",
        videos_dir=tmp_path / "videos",
        run_id="test-run",
        **kw,
    )


def test_run_episode_closed_loop(tmp_path):
    writer = RolloutWriter(tmp_path / "rollouts", run_id="t", write_frames=False, write_video=False)
    res = run_episode(
        _FakeEnv(n_steps=3), _FakePolicy(), init_state=np.zeros(4), instruction="put the bowl on the plate",
        max_steps=50, episode_id="libero_goal/0/7/0", writer=writer, num_steps_wait=0,
        suite="libero_goal", task_id=0, task_name="put_bowl", init_state_id=7, seed=0,
        bddl_file="/x/put_bowl.bddl", model="openvla/openvla-7b-finetuned-libero-goal", policy_type="openvla",
    )
    assert res.success is True
    assert res.num_steps == 3            # stopped on done, not max_steps
    assert res.terminal_failure is None

    data = daft.read_parquet(res.parquet_path).to_pydict()
    assert_emits_schema(res.parquet_path)
    assert len(data["episode_id"]) == 3
    assert data["episode_id"][0] == "libero_goal/0/7/0"
    assert data["source"][0] == "libero"
    assert data["policy_type"][0] == "openvla"
    assert data["control_mode"][0] == "relative"
    assert data["bddl_file"][0] == "/x/put_bowl.bddl"
    assert data["success"] == [True, True, True]
    # per-step signal the wedge needs is populated:
    assert data["step_idx"] == [0, 1, 2]
    np.testing.assert_allclose(data["action"][0], [0.2] * 7, rtol=1e-6)
    assert abs(data["gripper_action"][0] - 0.2) < 1e-6
    assert abs(data["gripper_state"][0] - 0.08) < 1e-5  # qpos[0]-qpos[1]
    assert data["state"][0].shape == (8,)  # eef(3)+axisangle(3)+gripper(2)
    assert data["eef_pos"][0].shape == (3,)
    # media off -> null
    assert data["frame_path"][0] is None
    assert data["video_path"][0] is None


def test_run_episode_failure_labels_unlabeled(tmp_path):
    writer = RolloutWriter(tmp_path / "rollouts", run_id="t", write_frames=False, write_video=False)
    res = run_episode(
        _FakeEnv(n_steps=999), _FakePolicy(), init_state=np.zeros(4), instruction="x",
        max_steps=5, episode_id="libero_goal/0/0/0", writer=writer, num_steps_wait=0,
    )
    assert res.success is False
    assert res.num_steps == 5                       # ran to max_steps without done
    assert res.terminal_failure == "unlabeled"
    data = daft.read_parquet(res.parquet_path).to_pydict()
    assert data["success"] == [False] * 5
    assert data["terminal_failure"][0] == "unlabeled"


def test_run_sweep_iteration_and_episode_ids(tmp_path):
    cfg = _cfg(tmp_path)
    env = _FakeEnv(n_steps=2)
    results = run_sweep(
        cfg, _FakePolicy(),
        env_factory=lambda s, t: (env, _FakeTask()),
        init_states_provider=lambda s, t: [np.zeros(4), np.zeros(4), np.zeros(4)],
        num_tasks_provider=lambda s: 1,
    )
    assert len(results) == 2                         # n_episodes_per_task=2
    assert [r.episode_id for r in results] == ["libero_goal/0/0/0", "libero_goal/0/1/0"]
    assert all(r.success for r in results)
    assert env.reset_calls == 2                      # one env.reset() per episode (horizon bug)

    # both parts land in the rollouts dir and form one canonical glob
    data = daft.read_parquet(str(cfg.out_dir / "*.parquet")).to_pydict()
    assert set(data["episode_id"]) == {"libero_goal/0/0/0", "libero_goal/0/1/0"}
    assert set(data["policy_type"]) == {"openvla"}


def test_frame_capture_writes_png(tmp_path):
    pytest.importorskip("PIL")  # imageio PNG writer uses pillow
    writer = RolloutWriter(tmp_path / "rollouts", frames_dir=tmp_path / "frames",
                           run_id="t", write_frames=True, write_video=False)
    res = run_episode(
        _FakeEnv(n_steps=2), _FakePolicy(), init_state=np.zeros(4), instruction="x",
        max_steps=10, episode_id="libero_goal/0/0/0", writer=writer, num_steps_wait=0,
    )
    data = daft.read_parquet(res.parquet_path).to_pydict()
    fp = data["frame_path"][0]
    assert fp is not None and fp.endswith("_primary.png")
    from pathlib import Path
    assert Path(fp).exists()
