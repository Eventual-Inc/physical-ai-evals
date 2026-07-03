"""Rollout subpackage: the Policy contract (concrete) + the LIBERO runner (stub).

``policy`` (Policy ABC, Observation) is import-light. ``libero_runner`` lazy-imports the
robosuite/mujoco/LIBERO stack and forces MUJOCO_GL before those imports — do not import
it at package init.
"""

from __future__ import annotations

from harness.rollout.policy import Observation, Policy

__all__ = ["Policy", "Observation"]
