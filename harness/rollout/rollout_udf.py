"""LIBERO rollout as a Daft class UDF — the Daft-native sweep.

Backend: PyTorch VLA policies (OpenVLA / VLA-JEPA) driving the LIBERO robosuite/MuJoCo sim.
This is the ``model.py`` equivalent of the daft-examples convention: a ``@daft.cls`` that loads
the policy once per worker and runs one episode per input row. This module NEVER imports
``modal`` — see ``modal_app.py`` for deployment.

Input: one row per episode spec ``(suite, task_id, init_state_id, seed)``. Each row runs a full
LIBERO episode; the per-STEP trajectory is streamed to a ``RolloutWriter`` (one parquet part +
frames + mp4 per episode, on the OUTPUT Volume), and the UDF returns an episode-level summary
struct. The queryable ROLLOUT_SCHEMA DataFrame is the parquet-part glob the writer produces
(``daft.read_parquet(f"{out_dir}/*.parquet")``); this DataFrame is the run manifest.

NOTE: this requires real LIBERO + a GPU policy, so it is exercised on Modal, not in CPU CI. The
closed loop it calls (``run_episode``) and the writer are unit-tested with fakes in
``tests/test_rollout.py``.
"""

from __future__ import annotations

import daft
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


def _build_policy(policy_type: str, model_id: str, device: str, unnorm_key: str | None):
    if policy_type == "openvla":
        from harness.policies.openvla import OpenVLAPolicy
        return OpenVLAPolicy(
            model_id=model_id or "openvla/openvla-7b-finetuned-libero-spatial",
            unnorm_key=unnorm_key, device=device,
        )
    if policy_type == "vla_jepa":
        from harness.policies.vla_jepa import VLAJEPAPolicy
        return VLAJEPAPolicy(policy_path=model_id or "lerobot/VLA-JEPA-LIBERO", device=device)
    raise ValueError(f"unknown policy_type: {policy_type!r} (expected 'openvla' | 'vla_jepa')")


@daft.cls(gpus=1.0, max_concurrency=1)
class LiberoRollout:
    """Loads ONE VLA policy per worker and rolls out one LIBERO episode per input row."""

    def __init__(
        self,
        *,
        policy_type: str,
        out_dir: str,
        model_id: str = "",
        frames_dir: str | None = None,
        videos_dir: str | None = None,
        run_id: str = "rollout",
        device: str = "cuda",
        unnorm_key: str | None = None,
        camera_height: int = 256,
        camera_width: int = 256,
        num_steps_wait: int = 10,
        max_steps: int | None = None,
        env_seed: int = 0,
        write_frames: bool = True,
        write_video: bool = True,
    ):
        from harness.writer import RolloutWriter

        self.policy_type = policy_type
        self.model_id = model_id
        self.device = device
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.num_steps_wait = num_steps_wait
        self.max_steps = max_steps
        self.env_seed = env_seed
        self.policy = _build_policy(policy_type, model_id, device, unnorm_key)  # once per worker
        self.writer = RolloutWriter(
            out_dir, frames_dir, videos_dir, run_id=run_id,
            write_frames=write_frames, write_video=write_video,
        )
        self._env_cache: dict = {}  # (suite, task_id) -> (env, task); one live env at a time

    def _env(self, suite: str, task_id: int):
        key = (suite, task_id)
        if key not in self._env_cache:
            from harness.rollout.libero_runner import make_env
            for env, _task in self._env_cache.values():
                env.close()  # close the previous task's env to avoid MuJoCo GL leaks
            self._env_cache = {
                key: make_env(suite, task_id, camera_height=self.camera_height,
                              camera_width=self.camera_width, seed=self.env_seed)
            }
        return self._env_cache[key]

    @daft.method.batch(return_dtype=RolloutSummary, batch_size=1)
    def rollout(self, suites: Series, task_ids: Series, init_state_ids: Series, seeds: Series) -> list[dict]:
        from harness.config import SUITE_MAX_STEPS
        from harness.rollout.libero_runner import _libero_init_states, run_episode

        out: list[dict] = []
        for suite, task_id, isid, seed in zip(
            suites.to_pylist(), task_ids.to_pylist(), init_state_ids.to_pylist(), seeds.to_pylist()
        ):
            env, task = self._env(suite, int(task_id))
            init_states = _libero_init_states(suite, int(task_id))
            max_steps = self.max_steps or SUITE_MAX_STEPS.get(suite, 300)
            res = run_episode(
                env, self.policy, init_state=init_states[int(isid)],
                instruction=getattr(task, "language", ""), max_steps=max_steps,
                episode_id=f"{suite}/{task_id}/{isid}/{seed}", writer=self.writer,
                num_steps_wait=self.num_steps_wait, suite=suite, task_id=int(task_id),
                task_name=getattr(task, "name", None), init_state_id=int(isid), seed=int(seed),
                bddl_file=getattr(task, "bddl_file", None), model=self.model_id,
                policy_type=self.policy_type,
            )
            out.append({
                "episode_id": res.episode_id, "suite": suite, "task_id": int(task_id),
                "init_state_id": int(isid), "seed": int(seed), "success": res.success,
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
    **runner_kwargs,
) -> daft.DataFrame:
    """One row per episode spec -> per-step parquet on disk + an episode-level summary frame."""
    runner = LiberoRollout(policy_type=policy_type, out_dir=out_dir, model_id=model_id, **runner_kwargs)
    return (
        daft.from_pydict(
            {"suite": suites, "task_id": task_ids, "init_state_id": init_state_ids, "seed": seeds}
        )
        .with_column("result", runner.rollout(col("suite"), col("task_id"), col("init_state_id"), col("seed")))
        .select(unnest(col("result")))
    )
