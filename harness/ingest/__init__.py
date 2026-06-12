"""Ingest adapters: normalize LeRobot / DROID / robomimic-HDF5 -> common Episode/Step.

``base`` (Episode, Step, Ingestor) is concrete and import-light. The three adapters
lazy-import their heavyweight deps (lerobot / tensorflow / h5py) inside ``load`` so
importing this package never requires all three stacks. Import an adapter class only
when you actually use it.
"""

from __future__ import annotations

from harness.ingest.base import (
    DEFAULT_CAMERA_ROLE_MAPS,
    PRIMARY,
    WRIST,
    Episode,
    Ingestor,
    Step,
)
from harness.ingest.droid import DroidIngestor, parse_trajectory_h5
from harness.ingest.hdf5 import Hdf5Ingestor

__all__ = [
    "Episode",
    "Step",
    "Ingestor",
    "PRIMARY",
    "WRIST",
    "DEFAULT_CAMERA_ROLE_MAPS",
    "Hdf5Ingestor",
    "DroidIngestor",
    "parse_trajectory_h5",
]
