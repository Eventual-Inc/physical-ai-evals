"""Ingestor adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from harness.core.episode import PRIMARY, WRIST, Episode

DEFAULT_CAMERA_ROLE_MAPS: dict[str, dict[str, str]] = {
    "hdf5": {
        "agentview_image": PRIMARY,
        "robot0_eye_in_hand_image": WRIST,
        "agentview_rgb": PRIMARY,
        "eye_in_hand_rgb": WRIST,
    },
}


class Ingestor(ABC):
    source: str = "base"

    def __init__(self, camera_role_map: dict[str, str] | None = None) -> None:
        self.camera_role_map = (
            camera_role_map or DEFAULT_CAMERA_ROLE_MAPS.get(self.source, {})
        )

    @abstractmethod
    def load(self, path: str, *, limit: int | None = None) -> Iterator[Episode]:
        raise NotImplementedError
