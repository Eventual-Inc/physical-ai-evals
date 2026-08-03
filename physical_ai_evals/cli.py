"""Command-line entrypoint for local evaluation and result inspection."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physical-ai-evals",
        description="Evaluate OpenVLA or VLA-JEPA on the LIBERO benchmark family.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("evaluate", help="run or resume an evaluation")
    run.add_argument(
        "--policy",
        choices=("openvla", "vla_jepa", "vla_jepa_cutile"),
        required=True,
    )
    run.add_argument(
        "--benchmark",
        choices=("libero", "libero_para", "libero_pro"),
        default="libero",
    )
    run.add_argument("--suite", default="libero_spatial")
    run.add_argument(
        "--tasks",
        default="",
        help="comma-separated task IDs (LIBERO/Para) or task keys (Para/Pro)",
    )
    run.add_argument(
        "--perturbations",
        default="",
        help="comma-separated Para paraphrase types or Pro perturbations",
    )
    run.add_argument("--episodes", type=int, default=50)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--model-id", default="")
    run.add_argument("--revision", default="")
    run.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    run.add_argument("--out", type=Path, default=Path("data/evaluations"))
    run.add_argument("--env-batch-size", type=int, default=1)
    run.add_argument("--no-video", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(function=_evaluate)

    read = commands.add_parser("read", help="summarize an evaluation directory")
    read.add_argument("path", type=Path)
    read.add_argument(
        "--group-by",
        default="",
        help="comma-separated episode columns, for example task_id or perturbation",
    )
    read.set_defaults(function=_read)
    return parser


def _benchmark(args: argparse.Namespace):
    from physical_ai_evals import libero, libero_para, libero_pro

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()] or None
    perturbations = [item.strip() for item in args.perturbations.split(",") if item.strip()] or None
    if args.benchmark == "libero":
        return libero(
            args.suite,
            task_ids=[int(task) for task in tasks] if tasks else None,
            episodes=args.episodes,
            seed=args.seed,
        )
    if args.benchmark == "libero_para":
        task_ids = (
            [int(task) for task in tasks]
            if tasks and all(task.isdigit() for task in tasks)
            else None
        )
        return libero_para(
            task_ids=task_ids,
            task_keys=tasks if tasks and task_ids is None else None,
            paraphrase_types=perturbations,
            episodes=args.episodes,
            seed=args.seed,
        )
    return libero_pro(
        args.suite,
        task_keys=tasks,
        perturbations=perturbations,
        episodes=args.episodes,
        seed=args.seed,
    )


def _policy(args: argparse.Namespace):
    from physical_ai_evals import openvla, vla_jepa, vla_jepa_cutile

    if args.policy == "openvla":
        suite = "libero_goal" if args.benchmark == "libero_para" else args.suite
        return openvla(
            suite,
            model_id=args.model_id or None,
            revision=args.revision or None,
            device=args.device,
        )
    kwargs = {"device": args.device}
    if args.model_id:
        kwargs["model_id"] = args.model_id
    if args.revision:
        kwargs["revision"] = args.revision
    if args.policy == "vla_jepa":
        return vla_jepa(**kwargs)
    if args.device != "cuda":
        raise ValueError("vla_jepa_cutile requires --device cuda")
    kwargs.pop("device")
    return vla_jepa_cutile(**kwargs)


def _evaluate(args: argparse.Namespace) -> int:
    benchmark = _benchmark(args)
    policy = _policy(args)
    if args.dry_run:
        print(
            {
                "policy": f"{policy.policy_id}@{policy.revision}",
                "benchmark": benchmark.name,
                "benchmark_revision": benchmark.revision,
                "suite": args.suite,
                "episodes_per_task": args.episodes,
                "seed": args.seed,
                "env_batch_size": args.env_batch_size,
                "out": str(args.out),
            }
        )
        return 0

    from physical_ai_evals import evaluate

    evaluation = evaluate(
        policy,
        benchmark,
        out=args.out,
        write_video=not args.no_video,
        env_batch_size=args.env_batch_size,
    )
    metrics = evaluation.metrics().to_pydict()
    print(
        f"{metrics['successes'][0] or 0}/{metrics['episodes'][0]} succeeded "
        f"({evaluation.success_rate():.3f}) -> {evaluation.path}"
    )
    return 0


def _read(args: argparse.Namespace) -> int:
    from physical_ai_evals import read_evaluation

    evaluation = read_evaluation(args.path)
    groups = tuple(item.strip() for item in args.group_by.split(",") if item.strip())
    evaluation.metrics(*groups).show()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
