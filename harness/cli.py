
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

from harness.core.config import SUITE_MAX_STEPS, IngestConfig, RolloutConfig


def _auto_run_id(prefix: str) -> str:
    return f"{prefix}-{_dt.datetime.now():%Y%m%d-%H%M%S}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description="VLA rollout -> parquet harness")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("rollout", help="run a policy through LIBERO -> rollout parquet")
    r.add_argument("--policy", required=True, choices=["openvla", "vla_jepa"],
                   help="which VLA backend to roll out")
    r.add_argument("--suite", default="libero_spatial",
                   help="one LIBERO suite key (default: libero_spatial)")
    r.add_argument("--episodes", type=int, default=50,
                   help="trials per task (canonical protocol = 50)")
    r.add_argument("--task-ids", default=None,
                   help="comma-separated task ids; default = all tasks in each suite")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--control-mode", default="relative", choices=["relative"],
                   help="action convention (only the implemented relative mode is accepted)")
    r.add_argument("--model-id", default=None, help="override the policy checkpoint id")
    r.add_argument("--model-revision", default=None,
                   help="immutable 40-character Hugging Face commit SHA")
    r.add_argument("--unnorm-key", default=None,
                   help="OpenVLA action unnormalization key (the SUITE name, e.g. libero_goal)")
    r.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    r.add_argument("--out", type=Path, default=Path("data/rollouts"))
    r.add_argument("--run-id", default=None)
    r.add_argument("--dry-run", action="store_true",
                   help="print the resolved plan; do not import policy/env stacks")
    r.set_defaults(func=_cmd_rollout)

    i = sub.add_parser("ingest", help="normalize a dataset -> rollout parquet")
    i.add_argument("--source", required=True, choices=["hdf5"])
    i.add_argument("--input", required=True, help="path to a robomimic/LIBERO .hdf5 demo file")
    i.add_argument("--out", type=Path, default=Path("data/rollouts"))
    i.add_argument("--limit", type=int, default=None, help="cap episodes (smoke tests)")
    i.add_argument("--run-id", default=None)
    i.add_argument("--dry-run", action="store_true")
    i.set_defaults(func=_cmd_ingest)

    return p


def _cmd_rollout(args: argparse.Namespace) -> int:
    suites = tuple(s.strip() for s in args.suite.split(",") if s.strip())
    task_ids = (
        tuple(int(t) for t in args.task_ids.split(",")) if args.task_ids else None
    )
    from harness.cloud.sweep import (
        evaluation_fingerprint,
        immutable_model_id,
        implementation_fingerprint,
        resolve_openvla_config,
        resolve_vla_jepa_config,
        runtime_provenance,
        write_evaluation_manifest,
    )

    if args.policy == "openvla":
        load_model_id, model_revision, unnorm_key = resolve_openvla_config(
            suites,
            model_id=args.model_id,
            unnorm_key=args.unnorm_key,
            model_revision=args.model_revision,
        )
    else:
        load_model_id, model_revision = resolve_vla_jepa_config(
            args.model_id, args.model_revision
        )
        unnorm_key = args.unnorm_key

    recorded_model_id = immutable_model_id(load_model_id, model_revision)
    evaluation_config = {
        "policy_type": args.policy,
        "model": recorded_model_id,
        "suites": sorted(set(suites)),
        "seed": args.seed,
        "unnorm_key": unnorm_key if args.policy == "openvla" else None,
        "center_crop_scale": 0.9 if args.policy == "openvla" else None,
        "attn_impl": "sdpa" if args.policy == "openvla" else None,
        "camera_height": 256,
        "camera_width": 256,
        "num_steps_wait": 10,
        "max_steps": {suite: SUITE_MAX_STEPS.get(suite, 300) for suite in sorted(set(suites))},
        "control_mode": "relative",
        "write_frames": True,
        "write_video": True,
        "run_id_override": args.run_id,
        "implementation": implementation_fingerprint(args.policy),
        "runtime": runtime_provenance(
            {
                "daft": "daft",
                "huggingface_hub": "huggingface-hub",
                "lerobot": "lerobot",
                "libero": "libero",
                "modal": "modal",
                "numpy": "numpy",
                "torch": "torch",
                "transformers": "transformers",
            }
        ),
    }
    evaluation_id = evaluation_fingerprint(evaluation_config)
    out_dir = args.out / args.policy / evaluation_id
    media_root = args.out.parent
    cfg = RolloutConfig(
        policy_type=args.policy,
        suites=suites,
        n_episodes_per_task=args.episodes,
        task_ids=task_ids,
        seed=args.seed,
        control_mode=args.control_mode,
        model_id=recorded_model_id,
        unnorm_key=unnorm_key,
        device=args.device,
        out_dir=out_dir,
        frames_dir=media_root / "frames" / args.policy / evaluation_id,
        videos_dir=media_root / "videos" / args.policy / evaluation_id,
        run_id=args.run_id or f"rollout-{args.policy}-{evaluation_id}",
    )
    if args.dry_run:
        print(f"[dry-run] rollout plan: {cfg}")
        return 0

    write_evaluation_manifest(cfg.out_dir, evaluation_id, evaluation_config)

    from huggingface_hub import snapshot_download

    from harness.bench.libero import run_sweep

    model_path = snapshot_download(repo_id=load_model_id, revision=model_revision)

    if args.policy == "openvla":
        from harness.policy.openvla import OpenVLAPolicy
        policy = OpenVLAPolicy(
            model_id=model_path,
            unnorm_key=cfg.unnorm_key, device=cfg.device, attn_impl="sdpa",
        )
    else:
        from harness.policy.vla_jepa import VLAJEPAPolicy
        policy = VLAJEPAPolicy(policy_path=model_path, device=cfg.device)

    results = run_sweep(cfg, policy)
    n_success = sum(r.success for r in results)
    print(f"{n_success}/{len(results)} episodes succeeded -> {cfg.out_dir}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    cfg = IngestConfig(source=args.source, input_path=args.input,
                       out_dir=args.out, limit_episodes=args.limit)
    run_id = args.run_id or _auto_run_id(f"ingest-{args.source}")
    if args.dry_run:
        print(f"[dry-run] ingest plan: {cfg} run_id={run_id}")
        return 0

    from harness.core.writer import write_episode
    from harness.ingest.hdf5 import Hdf5Ingestor

    ingestor = Hdf5Ingestor(camera_role_map=cfg.camera_role_map)
    n = 0
    for episode in ingestor.load(cfg.input_path, limit=cfg.limit_episodes):
        write_episode(episode, cfg.out_dir, run_id=run_id)
        n += 1
    print(f"ingested {n} episodes -> {cfg.out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
