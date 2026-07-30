"""Geometry helpers shared by rollout and dataset adapters."""

from __future__ import annotations

import numpy as np


def quat_xyzw_to_axis_angle(quaternion) -> np.ndarray:
    """Convert ``(..., 4)`` xyzw quaternions to axis-angle rotation vectors."""
    value = np.asarray(quaternion, dtype=np.float64)
    value = value / np.clip(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8, None)
    x, y, z, w = (
        value[..., 0],
        value[..., 1],
        value[..., 2],
        value[..., 3],
    )
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    scale = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    small = scale < 1e-6
    safe_scale = np.where(small, 1.0, scale)
    axis = np.stack([x, y, z], axis=-1) / safe_scale[..., None]
    rotation = axis * angle[..., None]
    return np.where(small[..., None], 0.0, rotation).astype(np.float32)
