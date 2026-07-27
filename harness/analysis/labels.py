"""Failure-mode labeling from per-step rollout signals."""

from __future__ import annotations


def detect_regrasp(
    obj_z: list[float],
    grip_closed: list[bool],
    *,
    lift: float = 0.05,
    drop: float = 0.03,
) -> tuple[str, list[tuple[int, str]], int]:
    """Label an episode and return its event timeline from the per-step signal.

    grasp = gripper closes; lift = object rises above ``lift``; drop = a lifted object falls
    below ``drop``; re-grasp = the gripper closes again after a drop. >=1 re-grasp -> ``re_grasp``.
    """
    events: list[tuple[int, str]] = []
    lifted, drops, regrasps, prev = False, 0, 0, False
    for t, (z, c) in enumerate(zip(obj_z, grip_closed, strict=True)):
        if c and not prev:
            events.append((t, "re-grasp" if drops > 0 else "grasp"))
            regrasps += drops > 0
        if z > lift and not lifted:
            lifted = True
            events.append((t, "lift"))
        if lifted and z < drop:
            lifted, drops = False, drops + 1
            events.append((t, "drop"))
        prev = c
    if regrasps >= 1:
        label = "re_grasp"
    elif drops >= 1:
        label = "drop_no_recover"
    elif any(grip_closed):
        label = "missed_target"
    else:
        label = "no_grasp"
    return label, events, regrasps
