"""Shared geometry helpers for proprio and ingest adapters."""

from __future__ import annotations

import numpy as np


def quat_xyzw_to_axis_angle(quat) -> np.ndarray:
    """(N,4) xyzw quaternion -> (N,3) axis-angle (rotation vector). Robust near identity."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    small = s < 1e-6
    safe_s = np.where(small, 1.0, s)
    axis = np.stack([x, y, z], axis=-1) / safe_s[..., None]
    rotvec = axis * angle[..., None]
    rotvec = np.where(small[..., None], 0.0, rotvec)
    return rotvec.astype(np.float32)
