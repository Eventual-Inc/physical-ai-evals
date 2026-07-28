"""Robomimic / LIBERO HDF5 adapter backed by Daft file IO."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import numpy as np

from physical_ai_evals.core.episode import Episode, Step
from physical_ai_evals.core.geometry import quat_xyzw_to_axis_angle
from physical_ai_evals.ingest.base import Ingestor

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


def _read_file_meta(attrs, path):
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


def _episode_from_demo(
    arrays,
    base_path,
    demo_index,
    file_stem,
    instruction,
    task_name,
    suite,
    bddl_file,
):
    def value(relative_path):
        return arrays.get(f"{base_path}/{relative_path}")

    actions = np.asarray(value("actions"), dtype=np.float32)
    n = len(actions)
    rewards_value = value("rewards")
    dones_value = value("dones")
    rewards = np.asarray(rewards_value) if rewards_value is not None else None
    dones = np.asarray(dones_value) if dones_value is not None else None

    eef_pos_arr = state_arr = gripper_scalar = None
    obs_paths = {
        path.removeprefix(f"{base_path}/obs/")
        for path in arrays
        if path.startswith(f"{base_path}/obs/")
    }
    if obs_paths:
        pk = _first_present(obs_paths, _EEF_POS_KEYS)
        eef_pos_arr = np.asarray(value(f"obs/{pk}"), dtype=np.float32) if pk else None

        qk = _first_present(obs_paths, _EEF_QUAT_KEYS)
        ak = _first_present(obs_paths, _EEF_AXISANGLE_KEYS)
        if qk:
            axis_angle = quat_xyzw_to_axis_angle(value(f"obs/{qk}"))
        elif ak:
            axis_angle = np.asarray(value(f"obs/{ak}"), dtype=np.float32)
        else:
            axis_angle = None

        gk = _first_present(obs_paths, _GRIPPER_QPOS_KEYS)
        gqpos = (
            np.asarray(value(f"obs/{gk}"), dtype=np.float32).reshape(n, -1)
            if gk
            else None
        )
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
    """Adapter for local or remote robomimic/LIBERO HDF5 demo files."""

    source = "hdf5"

    def __init__(self, camera_role_map=None, *, io_config=None) -> None:
        super().__init__(camera_role_map=camera_role_map)
        self.io_config = io_config

    def load(
        self,
        path: str,
        *,
        limit: int | None = None,
        split: str | None = None,
    ) -> Iterator[Episode]:
        from daft.file import Hdf5File

        file_stem = os.path.splitext(os.path.basename(path))[0]
        hdf5 = Hdf5File(path, io_config=self.io_config)
        instruction, task_name, suite, bddl_file = _read_file_meta(
            hdf5.attrs("/data"), path
        )

        demo_keys = sorted(
            (key for key in hdf5.keys("/data") if key.startswith("demo")),
            key=lambda key: int(key.split("_")[1]),
        )
        if split is not None and "mask" in hdf5.keys("/"):
            mask_path = f"/mask/{split}"
            allowed = {_attr_str(value) for value in hdf5.read(mask_path)}
            demo_keys = [key for key in demo_keys if key in allowed]
        if limit is not None:
            demo_keys = demo_keys[:limit]

        for key in demo_keys:
            base_path = f"data/{key}"
            selected_paths = {
                f"{base_path}/actions",
                f"{base_path}/rewards",
                f"{base_path}/dones",
                *(
                    f"{base_path}/obs/{name}"
                    for name in (
                        *_EEF_POS_KEYS,
                        *_EEF_QUAT_KEYS,
                        *_EEF_AXISANGLE_KEYS,
                        *_GRIPPER_QPOS_KEYS,
                    )
                ),
            }
            dataset_paths = [
                item["h5path"]
                for item in hdf5.metadata(base_path)
                if item["kind"] == "dataset" and item["h5path"] in selected_paths
            ]
            arrays = hdf5.read(dataset_paths)
            yield _episode_from_demo(
                arrays,
                base_path,
                int(key.split("_")[1]),
                file_stem,
                instruction,
                task_name,
                suite,
                bddl_file,
            )
