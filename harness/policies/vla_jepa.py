"""VLA-JEPA policy adapter — the comparative HEADLINER.

The public VLA-JEPA repo (``ginwind/VLA-JEPA``) does not expose a LeRobot
``PreTrainedPolicy`` factory. Its LIBERO evaluation starts a StarVLA policy server
(``deployment/model_server/server_policy.py``) and talks to it over a WebSocket client.

This adapter follows that real surface:

* ``reset(instruction)`` clears the local action-chunk cache.
* ``act(observation)`` sends de-rotated LIBERO images + state to the server when a new
  chunk is needed, unnormalizes the returned ``normalized_actions``, and returns one
  7-DoF LIBERO action.
* The gripper is converted from VLA-JEPA's open-probability convention (0/1) to the
  LIBERO command convention (-1=open, +1=close), matching the official eval script.

Heavy deps are still lazy. Unit tests inject a fake client and action stats, so no
server, GPU, or VLA-JEPA checkout is needed for CPU verification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from harness.rollout.policy import Observation, Policy

VLA_JEPA_REPO_ID = "ginwind/VLA-JEPA"
DEFAULT_LIBERO_CHECKPOINT = "LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
DEFAULT_UNNORM_KEY = "franka"
DEFAULT_IMAGE_SIZE = (224, 224)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _resize_uint8_hwc(image, size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resize an HWC uint8 image using Pillow, matching the official 224x224 VLA-JEPA eval."""
    arr = np.asarray(image)
    if arr.shape[:2] == size:
        return arr.astype(np.uint8, copy=False)
    try:
        from PIL import Image
    except ImportError as err:  # pragma: no cover - env-dependent
        raise ImportError("VLA-JEPA image resize requires Pillow.") from err
    pil = Image.fromarray(arr.astype(np.uint8, copy=False))
    return np.asarray(pil.resize(size[::-1], Image.Resampling.BILINEAR), dtype=np.uint8)


def _resolve_checkpoint_path(policy_path: str | os.PathLike[str] | None) -> Path:
    """Accept either a VLA-JEPA repo snapshot root or the concrete LIBERO ``.pt`` file."""
    raw = policy_path or os.environ.get("VLA_JEPA_CHECKPOINT_PATH") or VLA_JEPA_REPO_ID
    path = Path(str(raw)).expanduser()
    if path.suffix == ".pt":
        return path
    return path / DEFAULT_LIBERO_CHECKPOINT


def _checkpoint_root(checkpoint_path: Path) -> Path:
    # .../LIBERO/checkpoints/VLA-JEPA-LIBERO.pt -> .../LIBERO
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_action_metadata(
    checkpoint_path: str | os.PathLike[str],
    *,
    unnorm_key: str | None = None,
) -> tuple[dict[str, Any], str, int]:
    """Load VLA-JEPA action stats + chunk size from the HF snapshot next to the checkpoint."""
    ckpt = Path(checkpoint_path)
    root = _checkpoint_root(ckpt)
    stats_path = root / "dataset_statistics.json"
    config_path = root / "config.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing VLA-JEPA action stats at {stats_path}. Pass a local snapshot root "
            f"for {VLA_JEPA_REPO_ID} or the concrete {DEFAULT_LIBERO_CHECKPOINT} file."
        )

    stats_blob = json.loads(stats_path.read_text())
    key = unnorm_key or DEFAULT_UNNORM_KEY
    if key not in stats_blob:
        if len(stats_blob) == 1:
            key = next(iter(stats_blob))
        else:
            raise KeyError(f"VLA-JEPA unnorm_key={key!r} not found. Available: {sorted(stats_blob)}")

    action_stats = stats_blob[key]["action"]
    chunk_size = 7
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        future = cfg.get("framework", {}).get("action_model", {}).get("future_action_window_size")
        if future is not None:
            chunk_size = int(future) + 1
    return action_stats, key, chunk_size


def _unnormalize_actions(normalized_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    """Mirror VLA-JEPA's ``M1Inference.unnormalize_actions`` helper."""
    arr = np.asarray(normalized_actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected normalized actions with shape (chunk, action_dim), got {arr.shape}")
    arr = np.clip(arr.copy(), -1.0, 1.0)
    if arr.shape[1] >= 7:
        arr[:, 6] = np.where(arr[:, 6] < 0.5, 0.0, 1.0)

    low = np.asarray(action_stats.get("min", action_stats.get("q01")), dtype=np.float32)
    high = np.asarray(action_stats.get("max", action_stats.get("q99")), dtype=np.float32)
    if low.shape[0] != arr.shape[1] or high.shape[0] != arr.shape[1]:
        raise ValueError(f"action stats dim mismatch: actions={arr.shape[1]} low={low.shape} high={high.shape}")

    mask = np.asarray(action_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    return np.where(mask, 0.5 * (arr + 1.0) * (high - low) + low, arr).astype(np.float32)


def _libero_gripper_from_open(open_value: float | np.ndarray) -> float:
    """VLA-JEPA open gripper value (0/1) -> LIBERO command (-1=open, +1=close)."""
    v = float(np.asarray(open_value, dtype=np.float32).reshape(-1)[0])
    return float(1.0 - 2.0 * (v > 0.5))


class VLAJEPAPolicy(Policy):
    """VLA-JEPA via its official WebSocket policy server.

    ``policy_path`` is either a local HF snapshot root for ``ginwind/VLA-JEPA`` or the
    concrete ``LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`` file. The policy server must already
    be running unless a fake ``_client`` is injected by tests.
    """

    control_mode = "relative"

    def __init__(
        self,
        policy_path: str | os.PathLike[str] | None = None,
        device: str = "cuda",
        *,
        host: str = "127.0.0.1",
        port: int = 10093,
        unnorm_key: str | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        _client=None,
        _action_stats: dict[str, Any] | None = None,
        _action_chunk_size: int | None = None,
    ) -> None:
        self.policy_path = _resolve_checkpoint_path(policy_path)
        self.device = device
        self.host = host
        self.port = port
        self.image_size = image_size
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self._instruction = ""
        self._step = 0
        self._raw_actions: np.ndarray | None = None

        if _action_stats is None:
            self.action_stats, self.unnorm_key, self.action_chunk_size = load_action_metadata(
                self.policy_path, unnorm_key=unnorm_key
            )
        else:
            self.action_stats = _action_stats
            self.unnorm_key = unnorm_key or DEFAULT_UNNORM_KEY
            self.action_chunk_size = int(_action_chunk_size or 7)

        self.client = _client if _client is not None else self._connect_client()

    def _connect_client(self):
        try:
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        except ImportError as err:  # pragma: no cover - env-dependent
            raise ImportError(
                "VLA-JEPA rollouts need the ginwind/VLA-JEPA checkout on PYTHONPATH. "
                "Start deployment/model_server/server_policy.py in that env, then pass "
                "--model-id /path/to/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt."
            ) from err

        client = WebsocketClientPolicy(self.host, self.port)
        # Official client exposes this as a ping; it does not mutate model device on the server.
        if hasattr(client, "init_device"):
            client.init_device(self.device)
        return client

    def reset(self, instruction: str) -> None:
        """Start a new episode and clear the cached action chunk."""
        self._instruction = instruction
        self._step = 0
        self._raw_actions = None

    def act(self, observation: Observation) -> np.ndarray:
        """Return one unnormalized LIBERO action from the current or newly fetched chunk."""
        if self._raw_actions is None or self._step % self.action_chunk_size == 0:
            self._raw_actions = self._request_action_chunk(observation)

        idx = self._step % self.action_chunk_size
        action = np.asarray(self._raw_actions[idx], dtype=np.float32).reshape(-1)[: self.action_dim].copy()
        action[6] = _libero_gripper_from_open(action[6])
        self._step += 1
        return action.astype(np.float32, copy=False)

    def _request_action_chunk(self, observation: Observation) -> np.ndarray:
        images = [_resize_uint8_hwc(observation["image"], self.image_size)]
        if observation.get("wrist_image") is not None:
            images.append(_resize_uint8_hwc(observation["wrist_image"], self.image_size))

        payload: dict[str, Any] = {
            "batch_images": [images],
            "instructions": [self._instruction],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        if observation.get("state") is not None:
            payload["state"] = [np.asarray(observation["state"], dtype=np.float32)[None, :]]

        response = self.client.infer(payload)
        if response.get("ok") is False:
            raise RuntimeError(f"VLA-JEPA server inference failed: {response.get('error')}")

        data = response.get("data", response)
        normalized = _to_numpy(data["normalized_actions"])
        if normalized.ndim == 3:
            normalized = normalized[0]
        raw = _unnormalize_actions(normalized, self.action_stats)
        if raw.shape[0] < self.action_chunk_size:
            raise ValueError(f"VLA-JEPA returned {raw.shape[0]} actions, expected {self.action_chunk_size}")
        return raw

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()
