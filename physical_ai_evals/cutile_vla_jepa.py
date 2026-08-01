"""Persistent daft-cuTile VLA-JEPA policy for fixed LIBERO cohorts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_evals.policy import Observation
from physical_ai_evals.schema import ACTION_DIM, STATE_DIM

CUTILE_DAFT_REVISION = "6de66fa2274ef9efd730f6a35f2f0f26006c0475"
CUTILE_VLA_STATIC_BATCH_SIZE = 4
CUTILE_VLA_ACTION_HORIZON = 7
_IMAGE_SHAPE = (224, 224, 3)
_BATCH_IMAGE_SHAPE = (
    CUTILE_VLA_STATIC_BATCH_SIZE,
    2,
    *_IMAGE_SHAPE,
)
_BATCH_STATE_SHAPE = (CUTILE_VLA_STATIC_BATCH_SIZE, STATE_DIM)
_BATCH_NOISE_SHAPE = (
    CUTILE_VLA_STATIC_BATCH_SIZE,
    CUTILE_VLA_ACTION_HORIZON,
    ACTION_DIM,
)
_IMAGE_H2D_BYTES_PER_ROW = 2 * 224 * 224 * 3
_METADATA_H2D_BYTES_PER_ROW = 4_908
_OUTPUT_D2H_BYTES_PER_ROW = CUTILE_VLA_ACTION_HORIZON * ACTION_DIM * 4
_ALLOWED_TRANSFER_COUNTERS = frozenset(
    {
        "counters_enabled",
        "vla_action_input_h2d_bytes",
        "vla_action_input_d2d_bytes",
        "vla_action_output_d2h_bytes",
        "vla_action_host_visible_sync_count",
    }
)


def _normal_noise(
    *,
    episode_key: str,
    seed: int,
    chunk_index: int,
) -> np.ndarray:
    """Return stable per-episode noise independent of cohort row ordering."""

    identity = (
        f"physical-ai-evals/cutile-vla-jepa-noise-v1\0{episode_key}\0{seed}\0{chunk_index}"
    ).encode()
    generator_seed = int.from_bytes(hashlib.sha256(identity).digest()[:16], "little")
    return np.random.default_rng(generator_seed).standard_normal(
        (CUTILE_VLA_ACTION_HORIZON, ACTION_DIM),
        dtype=np.float32,
    )


def _image(value: Any, name: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value))
    if result.dtype != np.uint8 or result.shape != _IMAGE_SHAPE:
        raise ValueError(f"{name} must have dtype uint8 and shape {_IMAGE_SHAPE!r}")
    return result


def _state(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if result.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape {(STATE_DIM,)!r}")
    return result


class CutileVLAJEPAPolicy:
    """One persistent B4 session with deterministic seven-step chunk reuse."""

    action_dim = ACTION_DIM
    control_mode = "relative"

    def __init__(
        self,
        policy_path: str,
        qwen_path: str,
        *,
        device_id: int = 0,
        _engine: Any | None = None,
    ) -> None:
        self.policy_path = str(Path(policy_path))
        self.qwen_path = str(Path(qwen_path))
        if _engine is None:
            from daft_cutile import VLAJEPA
            from daft_cutile._internal.transfer_counters import drain

            _engine = VLAJEPA.from_host(
                self.policy_path,
                self.qwen_path,
                static_batch_size=CUTILE_VLA_STATIC_BATCH_SIZE,
                device_id=device_id,
            )
            self._drain_counters: Callable[[], Any] | None = drain
        else:
            self._drain_counters = None
        self._engine = _engine
        self._instructions: list[str] = []
        self._chunk: np.ndarray | None = None
        self._chunk_offset = CUTILE_VLA_ACTION_HORIZON
        self._chunk_index = 0
        self._warmed_active_rows: set[int] = set()
        self._prewarm_calls = 0
        self._action_chunk_calls = 0

    def reset(self, instruction: str) -> None:
        self.reset_batch([instruction])

    def reset_batch(self, instructions: Sequence[str]) -> None:
        if isinstance(instructions, (str, bytes)):
            raise TypeError("instructions must be a sequence of strings")
        resolved = list(instructions)
        if not 1 <= len(resolved) <= CUTILE_VLA_STATIC_BATCH_SIZE:
            raise ValueError("cuTile VLA-JEPA requires 1 to 4 fixed cohort rows")
        if any(not isinstance(value, str) for value in resolved):
            raise TypeError("instructions must contain only strings")
        self._instructions = resolved
        self._chunk = None
        self._chunk_offset = CUTILE_VLA_ACTION_HORIZON
        self._chunk_index = 0
        self._prewarm_calls = 0
        self._action_chunk_calls = 0
        if self._drain_counters is not None:
            self._drain_counters()
        self._engine.reset()

    def _inputs(
        self,
        observations: Sequence[Observation],
        *,
        chunk_index: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, int]:
        rows = list(observations)
        active_rows = len(rows)
        if active_rows != len(self._instructions):
            raise ValueError("cuTile VLA-JEPA observation batch does not match the reset cohort")

        images = np.zeros(_BATCH_IMAGE_SHAPE, dtype=np.uint8)
        states = np.zeros(_BATCH_STATE_SHAPE, dtype=np.float32)
        noise = np.zeros(_BATCH_NOISE_SHAPE, dtype=np.float32)
        instructions = [*self._instructions, *([""] * (CUTILE_VLA_STATIC_BATCH_SIZE - active_rows))]
        for row_index, (observation, instruction) in enumerate(
            zip(rows, self._instructions, strict=True)
        ):
            wrist = observation.get("wrist_image")
            if wrist is None:
                raise ValueError("cuTile VLA-JEPA requires the LIBERO wrist camera")
            state = observation.get("state")
            if state is None:
                raise ValueError("cuTile VLA-JEPA requires eight-value LIBERO telemetry")
            episode_key = observation.get("episode_key")
            seed = observation.get("seed")
            if not isinstance(episode_key, str) or not episode_key:
                raise ValueError("cuTile VLA-JEPA requires a non-empty episode_key")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("cuTile VLA-JEPA requires an integer episode seed")
            images[row_index, 0] = _image(observation["image"], "image")
            images[row_index, 1] = _image(wrist, "wrist_image")
            states[row_index] = _state(state)
            instructions[row_index] = observation.get("instruction") or instruction
            noise[row_index] = _normal_noise(
                episode_key=episode_key,
                seed=seed,
                chunk_index=chunk_index,
            )
        return images, states, instructions, noise, active_rows

    def prepare_batch(self, observations: Sequence[Observation]) -> None:
        """Compile and capture the exact active-row graph before profiling."""

        images, states, instructions, noise, active_rows = self._inputs(
            observations,
            chunk_index=0,
        )
        if active_rows in self._warmed_active_rows:
            return
        warmup_noise = np.zeros_like(noise)
        for _ in range(2):
            self._engine.predict_action_host(
                images,
                states,
                instructions,
                noise=warmup_noise,
                active_rows=active_rows,
            )
            self._prewarm_calls += 1
        self._warmed_active_rows.add(active_rows)

    def act(self, observation: Observation) -> np.ndarray:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: Sequence[Observation]) -> np.ndarray:
        rows = list(observations)
        if self._chunk_offset >= CUTILE_VLA_ACTION_HORIZON:
            self.prepare_batch(rows)
            images, states, instructions, noise, active_rows = self._inputs(
                rows,
                chunk_index=self._chunk_index,
            )
            output = np.asarray(
                self._engine.predict_action_host(
                    images,
                    states,
                    instructions,
                    noise=noise,
                    active_rows=active_rows,
                ),
                dtype=np.float32,
            )
            self._action_chunk_calls += 1
            expected = (active_rows, CUTILE_VLA_ACTION_HORIZON, ACTION_DIM)
            if output.shape != expected or not output.flags.c_contiguous:
                raise RuntimeError(
                    "cuTile VLA-JEPA action chunk must be contiguous float32 "
                    f"with shape {expected!r}"
                )
            self._chunk = output
            self._chunk_offset = 0
            self._chunk_index += 1
        if self._chunk is None:
            raise RuntimeError("cuTile VLA-JEPA action chunk is unavailable")
        action = self._chunk[:, self._chunk_offset].copy()
        self._chunk_offset += 1
        return action

    def batch_profile(self) -> dict[str, int | bool]:
        """Return engine counts and fail closed on a residency regression."""

        metrics: dict[str, int | bool] = {
            "cutile_prewarm_calls": self._prewarm_calls,
            "cutile_action_chunk_calls": self._action_chunk_calls,
            "cutile_actions_reused_per_chunk": CUTILE_VLA_ACTION_HORIZON,
        }
        if self._drain_counters is None:
            return metrics
        counters = self._drain_counters()
        payload = counters.as_dict()
        forbidden = sum(
            abs(float(value))
            for name, value in payload.items()
            if name not in _ALLOWED_TRANSFER_COUNTERS
        )
        expected_calls = self._prewarm_calls + self._action_chunk_calls
        active_rows = len(self._instructions)
        expected_input_bytes = (
            expected_calls
            * active_rows
            * (_IMAGE_H2D_BYTES_PER_ROW + _METADATA_H2D_BYTES_PER_ROW)
        )
        expected_output_bytes = (
            expected_calls * active_rows * _OUTPUT_D2H_BYTES_PER_ROW
        )
        resident = bool(
            counters.counters_enabled is True
            and counters.vla_action_input_h2d_bytes == expected_input_bytes
            and counters.vla_action_input_d2d_bytes == 0
            and counters.vla_action_output_d2h_bytes == expected_output_bytes
            and counters.vla_action_host_visible_sync_count == expected_calls
            and forbidden == 0
        )
        if not resident:
            raise RuntimeError(
                "cuTile VLA-JEPA device-residency counters changed: "
                f"expected_calls={expected_calls}, counters={payload!r}"
            )
        metrics.update(
            {
                "cutile_device_resident": True,
                "cutile_input_h2d_bytes": counters.vla_action_input_h2d_bytes,
                "cutile_output_d2h_bytes": counters.vla_action_output_d2h_bytes,
                "cutile_host_visible_sync_count": (counters.vla_action_host_visible_sync_count),
            }
        )
        return metrics

    def close(self) -> None:
        self._engine.close()
        self._chunk = None


class CutileVLAJEPAFactory:
    """Pickle-safe factory resolving only the action policy and Qwen weights."""

    def __init__(
        self,
        policy_id: str,
        policy_revision: str,
        qwen_id: str,
        qwen_revision: str,
        device_id: int,
        snapshot: Callable[[str, str], str],
    ) -> None:
        self.policy_id = policy_id
        self.policy_revision = policy_revision
        self.qwen_id = qwen_id
        self.qwen_revision = qwen_revision
        self.device_id = device_id
        self.snapshot = snapshot

    def __call__(self) -> CutileVLAJEPAPolicy:
        return CutileVLAJEPAPolicy(
            self.snapshot(self.policy_id, self.policy_revision),
            self.snapshot(self.qwen_id, self.qwen_revision),
            device_id=self.device_id,
        )


__all__ = [
    "CUTILE_DAFT_REVISION",
    "CUTILE_VLA_ACTION_HORIZON",
    "CUTILE_VLA_STATIC_BATCH_SIZE",
    "CutileVLAJEPAFactory",
    "CutileVLAJEPAPolicy",
]
