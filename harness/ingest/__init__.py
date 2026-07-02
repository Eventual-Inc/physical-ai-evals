"""Ingest adapters: normalize robotics datasets -> common Episode/Step.

``base`` (Episode, Step, Ingestor) is concrete and import-light. The adapters
lazy-import their heavyweight deps (lerobot / h5py / video stacks) inside ``load`` so
importing this package never requires all stacks. Import an adapter class only
when you actually use it.
"""

from __future__ import annotations

from harness.ingest.abc import ABCIngestor, parse_abc_episode_dir
from harness.ingest.aloha import AlohaIngestor, parse_aloha_hdf5
from harness.ingest.base import (
    DEFAULT_CAMERA_ROLE_MAPS,
    PRIMARY,
    WRIST,
    Episode,
    Ingestor,
    Step,
)
from harness.ingest.droid import DroidIngestor, parse_trajectory_h5
from harness.ingest.egodex import EgoDexIngestor, parse_egodex_hdf5
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
    "AlohaIngestor",
    "EgoDexIngestor",
    "ABCIngestor",
    "parse_trajectory_h5",
    "parse_aloha_hdf5",
    "parse_egodex_hdf5",
    "parse_abc_episode_dir",
]
