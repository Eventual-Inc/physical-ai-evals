"""LIBERO rollout runner — drives a policy through LIBERO and captures to parquet.

Wraps LIBERO faithfully per the OpenPI ``examples/libero/main.py`` reference. The env stack
is version-fragile (see NOTES.md): robosuite==1.4.0, bddl==1.0.1, numpy==1.22.4, gym==0.25.2,
and ``MUJOCO_GL`` (egl on Linux / cgl on macOS) MUST be set BEFORE importing robosuite/mujoco.

``run_episode`` is the policy/env-agnostic closed loop — it only touches the ``Policy`` ABC and
a gym-style env (``set_init_state`` / ``step`` / ``check_success`` / ``close``), so it is
unit-tested with a fake env + fake policy. ``make_env`` / the LIBERO suite enumeration use real
LIBERO (lazy import); ``run_sweep`` takes injectable seams so its iteration is testable too.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from harness.config import RolloutConfig
from harness.rollout.policy import Policy
from harness.writer import RolloutWriter


@dataclass
class RolloutResult:
    """Outcome of one episode."""

    episode_id: str
    success: bool
    reward: float
    num_steps: int
    terminal_failure: str | None
    parquet_path: str


def _set_mujoco_gl() -> None:
    """Force a headless GL backend BEFORE importing robosuite/mujoco (egl Linux / cgl macOS)."""
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


# --- observation helpers (robosuite obs dict -> our spine fields) ---

def _derotate(img) -> np.ndarray:
    """LIBERO agentview renders 180deg rotated vs released VLA checkpoints — undo it (NOTES.md)."""
    return np.asarray(img)[::-1, ::-1]


def _eef_pos(obs):
    v = obs.get("robot0_eef_pos")
    return None if v is None else np.asarray(v, np.float32).ravel()[:3]


def _gripper(obs):
    q = obs.get("robot0_gripper_qpos")
    if q is None:
        return None
    q = np.asarray(q, np.float32).ravel()
    return float(q[0] - q[1]) if q.size >= 2 else float(q[0])


def _proprio(obs):
    """LeRobot-style 8-dim state: eef pos (3) + eef axis-angle (3) + gripper qpos (2)."""
    eef, quat, grip = obs.get("robot0_eef_pos"), obs.get("robot0_eef_quat"), obs.get("robot0_gripper_qpos")
    if eef is None or grip is None:
        return None
    parts = [np.asarray(eef, np.float32).ravel()[:3]]
    if quat is not None:
        from harness.ingest.hdf5 import _quat_xyzw_to_axis_angle
        parts.append(_quat_xyzw_to_axis_angle(np.asarray(quat).reshape(1, 4))[0])
    grip = np.asarray(grip, np.float32).ravel()
    parts.append(grip[:2] if grip.size >= 2 else grip)
    return np.concatenate(parts).astype(np.float32)


def run_episode(
    env,
    policy: Policy,
    *,
    init_state,
    instruction: str,
    max_steps: int,
    episode_id: str,
    writer: RolloutWriter,
    num_steps_wait: int = 10,
    suite: str | None = None,
    task_id: int | None = None,
    task_name: str | None = None,
    init_state_id: int | None = None,
    seed: int | None = None,
    bddl_file: str | None = None,
    model: str = "",
    policy_type: str = "",
) -> RolloutResult:
    """Run one deterministic episode and stream it to ``writer`` -> one parquet part.

    Loop (faithful to OpenPI): ``env.reset()`` -> ``set_init_state`` (selects object layout,
    NOT just seed) -> settle ``num_steps_wait`` zero-action steps -> ``policy.reset`` -> for
    each step: de-rotate agentview, ``policy.act`` -> clip -> ``env.step`` (OLD gym 4-tuple)
    -> append. Success is the BDDL predicate ``env.check_success()``, independent of the
    ``done`` flag. ``terminal_failure`` is left ``'unlabeled'`` on failure; the notebook's
    pass assigns the real class.

    ``env.reset()`` per episode is LOAD-BEARING for cached envs: ``set_init_state`` alone does
    NOT clear robosuite's internal ``timestep``/``done``, so reuse accumulates toward the
    horizon (1000) and, once tripped, every later ``step`` raises "executing action in
    terminated episode" (NOTES.md — surfaced on the first multi-episode sweep, invisible in
    short runs).
    """
    writer.begin_episode(
        episode_id, suite=suite, task_id=task_id, task_name=task_name, instruction=instruction,
        model=model, policy_type=policy_type or policy.__class__.__name__, init_state_id=init_state_id,
        seed=seed, bddl_file=bddl_file, control_mode=getattr(policy, "control_mode", "relative"),
    )
    env.reset()
    obs = env.set_init_state(init_state)
    dummy = [0.0] * policy.action_dim
    for _ in range(num_steps_wait):  # let objects settle before policy control
        obs = env.step(dummy)[0]

    policy.reset(instruction)
    reward, steps_run = 0.0, 0
    for t in range(max_steps):
        primary = _derotate(obs["agentview_image"])
        wrist = _derotate(obs["robot0_eye_in_hand_image"]) if "robot0_eye_in_hand_image" in obs else None
        state = _proprio(obs)
        action = policy.act({"image": primary, "wrist_image": wrist, "state": state, "instruction": instruction})
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        obs, reward, done, _info = env.step(action)  # gym==0.25.2 4-tuple, NOT gymnasium 5-tuple
        writer.append_step(
            t, action=action, reward=float(reward), done=bool(done), state=state,
            eef_pos=_eef_pos(obs), gripper_state=_gripper(obs),
            primary_frame=primary, wrist_frame=wrist,
        )
        steps_run = t + 1
        if done:
            break

    success = bool(env.check_success())
    terminal_failure = None if success else "unlabeled"
    parquet = writer.end_episode(success, terminal_failure=terminal_failure)
    return RolloutResult(episode_id, success, float(reward), steps_run, terminal_failure, str(parquet))


def make_env(suite_name: str, task_id: int, *, camera_height: int = 256, camera_width: int = 256, seed: int = 0):
    """Build a LIBERO ``OffScreenRenderEnv`` for one task. Mirrors OpenPI ``_get_libero_env``.

    Returns ``(env, task)``; ``task.language`` is the instruction. ``env.step`` is the OLD gym
    4-tuple (gym==0.25.2). Sets ``MUJOCO_GL`` before the robosuite import.
    """
    _set_mujoco_gl()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env.seed(seed)
    return env, task


def _libero_num_tasks(suite_name: str) -> int:
    _set_mujoco_gl()
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite_name]().get_num_tasks()


def _libero_init_states(suite_name: str, task_id: int):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite_name]().get_task_init_states(task_id)


def run_sweep(
    cfg: RolloutConfig,
    policy: Policy,
    *,
    env_factory=None,
    init_states_provider=None,
    num_tasks_provider=None,
    writer_factory=None,
) -> list[RolloutResult]:
    """Run the full suite/task/episode sweep. ``episode_id`` = ``f"{suite}/{task}/{init}/{seed}"``.

    The four ``*_factory``/``*_provider`` seams default to real LIBERO; tests inject fakes to
    exercise the iteration without sim. Standard protocol: 10 episodes/task across the 4 core
    suites = 400 episodes. ``env.close()`` between tasks avoids leaking MuJoCo GL contexts.
    """
    env_factory = env_factory or (
        lambda s, t: make_env(s, t, camera_height=cfg.camera_height, camera_width=cfg.camera_width, seed=cfg.seed)
    )
    init_states_provider = init_states_provider or _libero_init_states
    num_tasks_provider = num_tasks_provider or _libero_num_tasks
    writer_factory = writer_factory or (
        lambda: RolloutWriter(
            cfg.out_dir, cfg.frames_dir, cfg.videos_dir, run_id=cfg.run_id or "rollout",
            write_frames=cfg.write_frames, write_video=cfg.write_video,
        )
    )
    writer = writer_factory()
    results: list[RolloutResult] = []
    for suite in cfg.suites:
        task_ids = cfg.task_ids if cfg.task_ids is not None else range(num_tasks_provider(suite))
        for task_id in task_ids:
            env, task = env_factory(suite, task_id)
            try:
                init_states = init_states_provider(suite, task_id)
                instruction = getattr(task, "language", "")
                max_steps = cfg.resolved_max_steps(suite)
                for init_state_id in range(min(cfg.n_episodes_per_task, len(init_states))):
                    episode_id = f"{suite}/{task_id}/{init_state_id}/{cfg.seed}"
                    results.append(
                        run_episode(
                            env, policy, init_state=init_states[init_state_id], instruction=instruction,
                            max_steps=max_steps, episode_id=episode_id, writer=writer,
                            num_steps_wait=cfg.num_steps_wait, suite=suite, task_id=task_id,
                            task_name=getattr(task, "name", None), init_state_id=init_state_id,
                            seed=cfg.seed, bddl_file=getattr(task, "bddl_file", None),
                            model=cfg.model_id or "", policy_type=cfg.policy_type,
                        )
                    )
            finally:
                env.close()
    return results
