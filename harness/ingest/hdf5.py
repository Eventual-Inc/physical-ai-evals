"""robomimic / LIBERO HDF5 adapter — the format LIBERO demos ship in."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import numpy as np

from harness.core.episode import Episode, Step
from harness.core.geometry import quat_xyzw_to_axis_angle
from harness.ingest.base import Ingestor

_PRIMARY_IMG_KEYS = ("agentview_image", "agentview_rgb")
_WRIST_IMG_KEYS = ("robot0_eye_in_hand_image", "eye_in_hand_rgb")
_EEF_POS_KEYS = ("robot0_eef_pos", "ee_pos")
_EEF_QUAT_KEYS = ("robot0_eef_quat",)
_EEF_AXISANGLE_KEYS = ("ee_ori",)
_GRIPPER_QPOS_KEYS = ("robot0_gripper_qpos", "gripper_states")

_KNOWN_SUITES = (
    "libero_spatial", "libero_object", "libero_goal",
    "libero_10", "libero_90", "libero_100",
)


def _attr_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return _attr_str(v.reshape(-1)[0])
    return str(v)


def _first_present(group, names):
    for n in names:
        if n in group:
            return n
    return None


def _read_file_meta(data_group, path):
    attrs = data_group.attrs
    instruction, task_name, suite, bddl_file = "", None, None, None
    if "problem_info" in attrs:
        info = json.loads(_attr_str(attrs["problem_info"]))
        instruction = info.get("language_instruction", "") or ""
        task_name = info.get("problem_name")
        suite = info.get("domain_name")
    elif "env_args" in attrs:
        env_args = json.loads(_attr_str(attrs["env_args"]))
        task_name = env_args.get("env_name")
    if "bddl_file_name" in attrs:
        bddl_file = _attr_str(attrs["bddl_file_name"])
    if suite is None:
        stem = os.path.basename(path).lower()
        suite = next((s for s in _KNOWN_SUITES if s in stem), None)
    return instruction, task_name, suite, bddl_file


def _episode_from_demo(g, demo_index, file_stem, instruction, task_name, suite, bddl_file):
    actions = np.asarray(g["actions"][:], dtype=np.float32)
    n = len(actions)
    rewards = np.asarray(g["rewards"][:]) if "rewards" in g else None
    dones = np.asarray(g["dones"][:]) if "dones" in g else None

    eef_pos_arr = state_arr = gripper_scalar = None
    obs = g["obs"] if "obs" in g else None
    if obs is not None:
        pk = _first_present(obs, _EEF_POS_KEYS)
        eef_pos_arr = np.asarray(obs[pk][:], dtype=np.float32) if pk else None

        qk = _first_present(obs, _EEF_QUAT_KEYS)
        ak = _first_present(obs, _EEF_AXISANGLE_KEYS)
        if qk:
            axis_angle = quat_xyzw_to_axis_angle(obs[qk][:])
        elif ak:
            axis_angle = np.asarray(obs[ak][:], dtype=np.float32)
        else:
            axis_angle = None

        gk = _first_present(obs, _GRIPPER_QPOS_KEYS)
        gqpos = np.asarray(obs[gk][:], dtype=np.float32).reshape(n, -1) if gk else None
        if gqpos is not None:
            gripper_scalar = (
                gqpos[:, 0] - gqpos[:, 1] if gqpos.shape[1] >= 2 else gqpos[:, 0]
            )

        if eef_pos_arr is not None and axis_angle is not None and gqpos is not None and gqpos.shape[1] >= 2:
            state_arr = np.concatenate(
                [eef_pos_arr[:, :3], axis_angle[:, :3], gqpos[:, :2]], axis=1
            ).astype(np.float32)

    success = bool(
        (rewards is not None and rewards.size and float(np.max(rewards)) == 1.0)
        or (dones is not None and dones.size and int(dones[-1]) == 1)
    )

    steps = tuple(
        Step(
            timestep=i,
            state=None if state_arr is None else state_arr[i],
            action=actions[i],
            reward=None if rewards is None else float(rewards[i]),
            done=bool(dones[i]) if dones is not None else (i == n - 1),
            is_terminal=(i == n - 1),
            eef_pos=None if eef_pos_arr is None else eef_pos_arr[i],
            gripper_state=None if gripper_scalar is None else float(gripper_scalar[i]),
        )
        for i in range(n)
    )

    return Episode(
        episode_id=f"hdf5/{file_stem}/{demo_index}",
        source="hdf5",
        instruction=instruction or "",
        steps=steps,
        success=success,
        terminal_failure=None,
        model="libero_demo",
        policy_type="hdf5",
        suite=suite,
        task_id=None,
        task_name=task_name,
        metadata={"control_mode": "relative", "bddl_file": bddl_file},
    )


class Hdf5Ingestor(Ingestor):
    """Adapter for robomimic/LIBERO HDF5 demo files."""

    source = "hdf5"

    def load(
        self,
        path: str,
        *,
        limit: int | None = None,
        split: str | None = None,
    ) -> Iterator[Episode]:
        import h5py

        file_stem = os.path.splitext(os.path.basename(path))[0]
        f = h5py.File(path, "r")
        try:
            data = f["data"]
            instruction, task_name, suite, bddl_file = _read_file_meta(data, path)

            demo_keys = sorted(
                (k for k in data.keys() if k.startswith("demo")),
                key=lambda k: int(k.split("_")[1]),
            )
            if split is not None and "mask" in f:
                allowed = {_attr_str(s) for s in f["mask"][split][:]}
                demo_keys = [k for k in demo_keys if k in allowed]
            if limit is not None:
                demo_keys = demo_keys[:limit]

            for k in demo_keys:
                yield _episode_from_demo(
                    data[k], int(k.split("_")[1]), file_stem,
                    instruction, task_name, suite, bddl_file,
                )
        finally:
            f.close()
