"""robomimic / LIBERO HDF5 adapter — the format LIBERO demos ship in.

CONCRETE (implemented). Normalizes a robomimic-style HDF5 demo file into the common
``Episode``/``Step`` representation (-> ``ROLLOUT_SCHEMA`` via ``Episode.to_step_rows``).
This is the one ingest format Daft has NO native reader for (see NOTES.md), so it's ours.

On-disk tree (robosuite/robomimic, what LIBERO writes)::

    data                                 (group)  attrs: problem_info / env_args (JSON),
      |                                            bddl_file_name, total, ...
      |-- demo_<i>                        (group)  attrs: num_samples
      |     |-- actions      (N, 7) f32   7-DoF EEF delta, gripper at [-1]
      |     |-- rewards      (N,)         sparse {0,1} (optional)
      |     |-- dones        (N,)         1 at terminal (optional)
      |     |-- states       (N, D)       flat MuJoCo replay state (NOT proprio; ignored)
      |     +-- obs          (group)      robot0_eef_pos (N,3), robot0_eef_quat (N,4 xyzw),
      |                                   robot0_gripper_qpos (N,2), agentview_image, ...
      +-- mask                            (group)  train/valid demo-id splits (optional)

Key traps handled (see NOTES.md): demo keys integer-sorted (``demo_10`` after ``demo_2``);
success derived from sparse reward / dones; instruction hoisted from file attrs (constant
per file); native-LIBERO obs aliases (``agentview_rgb``/``ee_pos``/``ee_ori``/``gripper_states``)
resolved alongside the robosuite-canonical names; missing obs/rewards/dones tolerated.

Frames are NOT loaded here (the adapter stays image-bytes-free for the pure-parquet path);
the 180deg de-rotation + PNG materialization belong in the writer's ``frame_path_for``
callback, so ``frame_path``/``wrist_path`` are null on this path.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import numpy as np

from harness.ingest.base import Episode, Ingestor, Step

# obs-key aliases: robosuite-canonical first, native-LIBERO second. First present wins.
_PRIMARY_IMG_KEYS = ("agentview_image", "agentview_rgb")
_WRIST_IMG_KEYS = ("robot0_eye_in_hand_image", "eye_in_hand_rgb")
_EEF_POS_KEYS = ("robot0_eef_pos", "ee_pos")
_EEF_QUAT_KEYS = ("robot0_eef_quat",)        # (N,4) xyzw -> convert to axis-angle
_EEF_AXISANGLE_KEYS = ("ee_ori",)            # (N,3) already axis-angle
_GRIPPER_QPOS_KEYS = ("robot0_gripper_qpos", "gripper_states")

_KNOWN_SUITES = (
    "libero_spatial", "libero_object", "libero_goal",
    "libero_10", "libero_90", "libero_100",
)


def _attr_str(v) -> str:
    """HDF5 attrs come back as bytes or str depending on writer; normalize to str."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):  # some writers store a 0-d/1-elem array
        return _attr_str(v.reshape(-1)[0])
    return str(v)


def _first_present(group, names):
    for n in names:
        if n in group:
            return n
    return None


def _quat_xyzw_to_axis_angle(quat) -> np.ndarray:
    """(N,4) xyzw quaternion -> (N,3) axis-angle (rotation vector). Robust near identity."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)                       # [0, 2pi]
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, None))     # sin(angle/2)
    small = s < 1e-6
    safe_s = np.where(small, 1.0, s)
    axis = np.stack([x, y, z], axis=-1) / safe_s[..., None]
    rotvec = axis * angle[..., None]
    rotvec = np.where(small[..., None], 0.0, rotvec)  # near-identity -> zero rotation
    return rotvec.astype(np.float32)


def _read_file_meta(data_group, path):
    """Instruction / task_name / suite / bddl are constant per file — read once."""
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
    if suite is None:  # fall back to the filename
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
            axis_angle = _quat_xyzw_to_axis_angle(obs[qk][:])
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

        # 8-dim proprio state only when all three pieces are present.
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
        episode_id=f"hdf5/{file_stem}/{demo_index}",  # file_stem namespaces across files
        source="hdf5",
        instruction=instruction or "",
        steps=steps,
        success=success,
        terminal_failure=None,  # expert demos; the notebook labels failure classes
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
        """Yield normalized Episodes from a robomimic/LIBERO HDF5 file.

        ``path``: .hdf5 file path. ``limit``: cap demos (smoke tests). ``split``: when the
        file has a ``mask`` group, restrict to that split's demo ids ('train'/'valid').
        """
        import h5py  # lazy: h5py is the ingest_hdf5 extra

        file_stem = os.path.splitext(os.path.basename(path))[0]
        f = h5py.File(path, "r")
        try:
            data = f["data"]
            instruction, task_name, suite, bddl_file = _read_file_meta(data, path)

            demo_keys = sorted(
                (k for k in data.keys() if k.startswith("demo")),
                key=lambda k: int(k.split("_")[1]),  # demo_10 must follow demo_2
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
