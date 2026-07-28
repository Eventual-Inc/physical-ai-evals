"""Daft UDF: one row per episode spec -> parquet + episode summary."""

from __future__ import annotations

import random

import daft
import numpy as np
from daft import DataType, Series, col
from daft.functions import unnest

RolloutSummary = DataType.struct(
    {
        "episode_id": DataType.string(),
        "suite": DataType.string(),
        "task_id": DataType.int64(),
        "init_state_id": DataType.int64(),
        "seed": DataType.int64(),
        "success": DataType.bool(),
        "num_steps": DataType.int64(),
        "reward": DataType.float64(),
        "terminal_failure": DataType.string(),
        "parquet_path": DataType.string(),
    }
)


def _build_policy(
    policy_type: str,
    model_id: str,
    model_revision: str,
    device: str,
    unnorm_key: str | None,
    attn_impl: str,
):
    if policy_type in {"openvla", "vla_jepa"} and not model_revision:
        raise ValueError(
            f"{policy_type} requires an immutable model_revision before policy loading"
        )
    if model_revision and (
        len(model_revision) != 40
        or any(character not in "0123456789abcdef" for character in model_revision)
    ):
        raise ValueError("model_revision must be an immutable 40-character commit SHA")
    load_path = model_id
    if model_revision:
        from physical_ai_evals.cloud.modal_infra import MODEL_CACHE_DIR, resolve_hf_model_path

        load_path = str(
            resolve_hf_model_path(model_id, MODEL_CACHE_DIR, revision=model_revision)
        )
    if policy_type == "openvla":
        from physical_ai_evals.policy.openvla import OpenVLAPolicy

        if not model_id:
            raise ValueError("OpenVLA requires a resolved suite-compatible model_id")
        return OpenVLAPolicy(
            model_id=load_path,
            unnorm_key=unnorm_key, device=device, attn_impl=attn_impl,
        )
    if policy_type == "vla_jepa":
        from physical_ai_evals.policy.vla_jepa import VLAJEPAPolicy

        return VLAJEPAPolicy(policy_path=load_path or None, device=device)
    raise ValueError(f"unknown policy_type: {policy_type!r} (expected 'openvla' | 'vla_jepa')")


def _seed_process(seed: int) -> None:
    """Seed process RNGs used by the simulator/policy for one episode."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@daft.cls(gpus=1.0, max_concurrency=1)
class LiberoRollout:
    """Loads ONE VLA policy per worker and rolls out one LIBERO episode per input row."""

    def __init__(
        self,
        *,
        policy_type: str,
        out_dir: str,
        model_id: str = "",
        model_revision: str = "",
        frames_dir: str | None = None,
        videos_dir: str | None = None,
        run_id: str = "rollout",
        device: str = "cuda",
        unnorm_key: str | None = None,
        attn_impl: str = "sdpa",
        camera_height: int = 256,
        camera_width: int = 256,
        num_steps_wait: int = 10,
        max_steps: int | None = None,
        env_seed: int | None = None,
        write_frames: bool = True,
        write_video: bool = True,
    ):
        from physical_ai_evals.core.writer import RolloutWriter

        self.policy_type = policy_type
        self.model_id = (
            f"{model_id}@{model_revision}" if model_id and model_revision else model_id
        )
        self.model_revision = model_revision
        self.device = device
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.num_steps_wait = num_steps_wait
        self.max_steps = max_steps
        self.env_seed = env_seed
        self.policy = _build_policy(
            policy_type, model_id, model_revision, device, unnorm_key, attn_impl
        )
        self.writer = RolloutWriter(
            out_dir, frames_dir, videos_dir, run_id=run_id,
            write_frames=write_frames, write_video=write_video,
        )
        self._env_cache: dict = {}

    def _env(self, suite: str, task_id: int, seed: int):
        if self.env_seed is not None and self.env_seed != seed:
            raise ValueError(
                f"legacy env_seed={self.env_seed} conflicts with row seed={seed}; "
                "the row seed now controls simulator and process RNGs"
            )
        key = (suite, task_id, seed)
        if key not in self._env_cache:
            from physical_ai_evals.bench.libero import make_env

            _seed_process(seed)
            for env, _task in self._env_cache.values():
                env.close()
            self._env_cache = {
                key: make_env(suite, task_id, camera_height=self.camera_height,
                              camera_width=self.camera_width, seed=seed)
            }
        return self._env_cache[key]

    @daft.method.batch(return_dtype=RolloutSummary, batch_size=1)
    def rollout(self, suites: Series, task_ids: Series, init_state_ids: Series, seeds: Series) -> list[dict]:
        from physical_ai_evals.bench.libero import libero_init_states, run_episode
        from physical_ai_evals.core.config import SUITE_MAX_STEPS

        out: list[dict] = []
        for suite, task_id, isid, seed in zip(
            suites.to_pylist(), task_ids.to_pylist(), init_state_ids.to_pylist(),
            seeds.to_pylist(), strict=True,
        ):
            row_seed = int(seed)
            _seed_process(row_seed)
            env, task = self._env(suite, int(task_id), row_seed)
            seed_env = getattr(env, "seed", None)
            if callable(seed_env):
                seed_env(row_seed)
            init_states = libero_init_states(suite, int(task_id))
            max_steps = self.max_steps or SUITE_MAX_STEPS.get(suite, 300)
            res = run_episode(
                env, self.policy, init_state=init_states[int(isid)],
                instruction=getattr(task, "language", ""), max_steps=max_steps,
                episode_id=f"{suite}/{task_id}/{isid}/{row_seed}", writer=self.writer,
                num_steps_wait=self.num_steps_wait, suite=suite, task_id=int(task_id),
                task_name=getattr(task, "name", None), init_state_id=int(isid), seed=row_seed,
                bddl_file=getattr(task, "bddl_file", None), model=self.model_id,
                policy_type=self.policy_type,
            )
            out.append({
                "episode_id": res.episode_id, "suite": suite, "task_id": int(task_id),
                "init_state_id": int(isid), "seed": row_seed, "success": res.success,
                "num_steps": res.num_steps, "reward": res.reward,
                "terminal_failure": res.terminal_failure, "parquet_path": res.parquet_path,
            })
        return out


def build_rollout_dataframe(
    suites: list[str],
    task_ids: list[int],
    init_state_ids: list[int],
    seeds: list[int],
    *,
    policy_type: str,
    out_dir: str,
    model_id: str = "",
    model_revision: str = "",
    **runner_kwargs,
) -> daft.DataFrame:
    """Build a Daft frame of episode summaries from rollout specs."""
    if policy_type == "openvla":
        from physical_ai_evals.cloud.sweep import resolve_openvla_config

        model_id, model_revision, resolved_key = resolve_openvla_config(
            suites,
            model_id=model_id or None,
            unnorm_key=runner_kwargs.get("unnorm_key"),
            model_revision=model_revision or None,
        )
        runner_kwargs["unnorm_key"] = resolved_key
    elif policy_type == "vla_jepa":
        from physical_ai_evals.cloud.sweep import resolve_vla_jepa_config

        model_id, model_revision = resolve_vla_jepa_config(
            model_id or None, model_revision or None
        )
    runner = LiberoRollout(
        policy_type=policy_type,
        out_dir=out_dir,
        model_id=model_id,
        model_revision=model_revision,
        **runner_kwargs,
    )
    return (
        daft.from_pydict(
            {"suite": suites, "task_id": task_ids, "init_state_id": init_state_ids, "seed": seeds}
        )
        .with_column("result", runner.rollout(col("suite"), col("task_id"), col("init_state_id"), col("seed")))
        .select(unnest(col("result")))
    )
