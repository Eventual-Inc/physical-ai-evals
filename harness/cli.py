"""Command-line entrypoints: ``harness rollout`` and ``harness ingest``.

argparse-based (zero extra deps). Wired to the real module paths; the heavy work is in
stubs (rollout sweep / ingest loop), so the commands construct config + dispatch and
will ``NotImplementedError`` until those land. ``--dry-run`` prints the resolved plan
without importing any heavyweight policy/env stack — useful to sanity-check wiring.

Examples
--------
    harness rollout --policy openvla --suite libero_goal --episodes 10
    harness rollout --policy vla_jepa --suite libero_spatial,libero_object --device mps
    harness ingest --source hdf5 --input demos/libero_goal.hdf5 --out data/rollouts
    harness ingest --source aloha --input demos/aloha_task --out data/rollouts
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

from harness.config import CORE_SUITES, IngestConfig, RolloutConfig


def _auto_run_id(prefix: str) -> str:
    return f"{prefix}-{_dt.datetime.now():%Y%m%d-%H%M%S}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description="VLA rollout -> parquet harness")
    sub = p.add_subparsers(dest="command", required=True)

    # --- rollout ---
    r = sub.add_parser("rollout", help="run a policy through LIBERO -> rollout parquet")
    r.add_argument("--policy", required=True, choices=["openvla", "vla_jepa"],
                   help="which VLA backend to roll out")
    r.add_argument("--suite", default=",".join(CORE_SUITES),
                   help="comma-separated LIBERO suite keys (default: 4 core suites)")
    r.add_argument("--episodes", type=int, default=50,
                   help="trials per task (canonical protocol = 50)")
    r.add_argument("--task-ids", default=None,
                   help="comma-separated task ids; default = all tasks in each suite")
    r.add_argument("--seed", type=int, default=7)  # canonical eval seed (OpenVLA-origin protocol)
    r.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    r.add_argument("--model-id", default=None, help="override the policy checkpoint id")
    r.add_argument("--unnorm-key", default=None,
                   help="OpenVLA action unnormalization key (the SUITE name, e.g. libero_goal)")
    r.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    r.add_argument("--out", type=Path, default=Path("data/rollouts"))
    r.add_argument("--run-id", default=None)
    r.add_argument("--dry-run", action="store_true",
                   help="print the resolved plan; do not import policy/env stacks")
    r.set_defaults(func=_cmd_rollout)

    # --- ingest ---
    i = sub.add_parser("ingest", help="normalize a dataset -> rollout parquet")
    i.add_argument(
        "--source",
        required=True,
        choices=["lerobot", "droid", "hdf5", "aloha", "egodex", "abc"],
    )
    i.add_argument("--input", required=True,
                   help="local path | HF repo_id (lerobot) | gs:// data_dir (droid)")
    i.add_argument("--out", type=Path, default=Path("data/rollouts"))
    i.add_argument("--limit", type=int, default=None, help="cap episodes (smoke tests)")
    i.add_argument("--run-id", default=None)
    i.add_argument("--dry-run", action="store_true")
    i.set_defaults(func=_cmd_ingest)

    # --- ABC metadata query ---
    q = sub.add_parser("abc-query", help="query ABC-130K metadata without downloading episodes")
    q.add_argument("--split", default="train", choices=["train", "val"])
    q.add_argument("--contains", default=None, help="filter task names by substring")
    q.add_argument("--task", default=None, help="list episodes for this task")
    q.add_argument("--episode", default=None, help="with --task, list files for this episode")
    q.add_argument("--limit", type=int, default=20)
    q.add_argument("--token-env", default="HF_TOKEN",
                   help="environment variable containing a Hugging Face token")
    q.set_defaults(func=_cmd_abc_query)

    return p


def _cmd_rollout(args: argparse.Namespace) -> int:
    suites = tuple(s.strip() for s in args.suite.split(",") if s.strip())
    task_ids = (
        tuple(int(t) for t in args.task_ids.split(",")) if args.task_ids else None
    )
    cfg = RolloutConfig(
        policy_type=args.policy,
        suites=suites,
        n_episodes_per_task=args.episodes,
        task_ids=task_ids,
        seed=args.seed,
        control_mode=args.control_mode,
        model_id=args.model_id,
        unnorm_key=args.unnorm_key,
        device=args.device,
        out_dir=args.out,
        run_id=args.run_id or _auto_run_id(f"rollout-{args.policy}"),
    )
    if args.dry_run:
        print(f"[dry-run] rollout plan: {cfg}")
        return 0

    # Lazy imports: only pull the heavy policy/env stacks when actually running.
    from harness.rollout.libero_runner import run_sweep

    if args.policy == "openvla":
        from harness.policies.openvla import OpenVLAPolicy
        policy = OpenVLAPolicy(
            model_id=cfg.model_id or "openvla/openvla-7b-finetuned-libero-spatial",
            unnorm_key=cfg.unnorm_key, device=cfg.device,
        )
    else:
        from harness.policies.vla_jepa import VLAJEPAPolicy
        policy = VLAJEPAPolicy(policy_path=cfg.model_id or None, device=cfg.device)

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

    from harness.writer import write_episode

    if args.source == "lerobot":
        from harness.ingest.lerobot import LeRobotIngestor as Ing
    elif args.source == "droid":
        from harness.ingest.droid import DroidIngestor as Ing
    elif args.source == "hdf5":
        from harness.ingest.hdf5 import Hdf5Ingestor as Ing
    elif args.source == "aloha":
        from harness.ingest.aloha import AlohaIngestor as Ing
    elif args.source == "egodex":
        from harness.ingest.egodex import EgoDexIngestor as Ing
    else:
        from harness.ingest.abc import ABCIngestor as Ing

    ingestor = Ing(camera_role_map=cfg.camera_role_map)
    n = 0
    for episode in ingestor.load(cfg.input_path, limit=cfg.limit_episodes):
        write_episode(episode, cfg.out_dir, run_id=run_id)
        n += 1
    print(f"ingested {n} episodes -> {cfg.out_dir}")
    return 0


def _cmd_abc_query(args: argparse.Namespace) -> int:
    from harness.ingest.abc_query import (
        dataset_info,
        format_bytes,
        list_episode_files,
        list_episodes,
        list_tasks,
        token_from_env,
    )

    token = token_from_env(args.token_env)
    if args.episode and not args.task:
        raise SystemExit("--episode requires --task")

    info = dataset_info(token=token)
    storage = format_bytes(int(info.get("usedStorage") or 0))
    print(
        f"{info.get('id', 'XDOF/ABC-130k')} "
        f"gated={info.get('gated')} downloads={info.get('downloads')} storage={storage}"
    )

    if args.task and args.episode:
        files = list_episode_files(args.task, args.episode, split=args.split, token=token)
        print(f"files for {args.split}/{args.task}/{args.episode}:")
        for entry in files[: args.limit]:
            size = f" {format_bytes(entry.size)}" if entry.size else ""
            print(f"  {entry.type:9} {entry.path}{size}")
        return 0

    if args.task:
        episodes = list_episodes(args.task, split=args.split, token=token)
        print(f"episodes for {args.split}/{args.task}: {len(episodes)} visible")
        for episode in episodes[: args.limit]:
            print(f"  {episode}")
        return 0

    tasks = list_tasks(split=args.split, contains=args.contains, token=token)
    suffix = f" matching {args.contains!r}" if args.contains else ""
    print(f"tasks in {args.split}{suffix}: {len(tasks)} visible")
    for task in tasks[: args.limit]:
        print(f"  {task}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
