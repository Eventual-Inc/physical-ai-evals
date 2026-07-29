"""Regression tests for Modal command construction in the Makefile."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _dry_run(target: str, **variables: str) -> list[str]:
    command = ["make", "-n", target]
    command.extend(f"{key}={value}" for key, value in variables.items())
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return shlex.split(result.stdout)


def test_empty_smoke_filters_do_not_consume_boolean_flag():
    command = _dry_run(
        "smoke-openvla",
        BENCHMARK="libero",
        SUITE="libero_spatial",
        PERTURBATIONS="",
    )

    assert "--perturbations" not in command
    assert command[-1] == "--smoke-test"


def test_rollout_includes_only_nonempty_optional_filters():
    command = _dry_run(
        "rollout-vla-jepa",
        BENCHMARK="libero_pro",
        SUITE="libero_spatial",
        TASKS="task-a,task-b",
        PERTURBATIONS="lan",
        EPISODES="1",
    )

    assert command[command.index("--tasks") + 1] == "task-a,task-b"
    assert command[command.index("--perturbations") + 1] == "lan"
    assert command[command.index("--episodes") + 1] == "1"
