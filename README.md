# physical-ai-evals

[![CI](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Run OpenVLA and VLA-JEPA on LIBERO, LIBERO-Para, and LIBERO-Pro from one
Python 3.12 API — pinned model and simulator environments, a provenance
manifest for every run, and step-level traces in LeRobot's episode/step/video
layout that show why an episode failed, not just how often.

```python
import physical_ai_evals as pae

evaluation = pae.evaluate(
    policy=pae.openvla("libero_spatial"),
    benchmark=pae.libero(
        "libero_spatial",
        task_ids=[0, 1],
        episodes=5,
        seed=7,
    ),
    out="data/evaluations",
)

print(evaluation.success_rate())
evaluation.metrics("task_id").show()
evaluation.steps.where(evaluation.steps["next.done"]).show()
```

## Reproduction status

Runs target the canonical LIBERO protocol: 50 episodes per task from the
published fixed initial states at seed 7, success rate aggregated per suite.
Every evaluation records checkpoint revisions, benchmark data revisions,
implementation source hash, package versions, and GPU details in
`manifest.json`, so a number is never separated from the configuration that
produced it.

| Suite | OpenVLA (published) | OpenVLA (this harness) | VLA-JEPA (this harness) |
|---|---|---|---|
| `libero_spatial` | 84.7 | pending | pending |
| `libero_object` | 88.4 | pending | pending |
| `libero_goal` | 79.2 | pending | pending |
| `libero_10` | 53.7 | pending | pending |

Published numbers are the fine-tuned OpenVLA results from the
[OpenVLA paper](https://arxiv.org/abs/2406.09246). Harness numbers are
reported only with the trace and manifest that produced them attached; none
exist yet. The earlier `rollout-v1` pilot trace is deliberately not presented
as a reproduction — its simulator seed was unverified and it used 10 of the 50
initial states; see [Evaluation protocol](docs/evaluation.md).

## See why an episode failed

A success rate is one scalar; the trace is the evidence. Every episode writes
typed step Parquet plus both camera videos, so failure analysis is a query,
not a re-run:

```python
run = pae.read_evaluation("data/evaluations/<evaluation_id>")
failures = run.episodes.where(run.episodes["success"] == False)
failure_steps = failures.join(run.steps, on="episode_key")
failure_steps.select("task_key", "frame_index", "action").show()
```

```text
<out>/<evaluation_id>/
  manifest.json
  timings.jsonl
  episodes/episode_key=.../*.parquet
  steps/episode_key=.../*.parquet
  videos/<episode_key>/primary.mp4
  videos/<episode_key>/wrist.mp4
```

The layout follows LeRobot's episode/step/video concepts without claiming to
be a complete LeRobot training dataset. Constant evaluation provenance lives
in the manifest; episode metadata and outcomes are not repeated on every
transition. Video paths are relative to the evaluation directory.

## Why evaluations resume and stay trustworthy

An evaluation has three useful objects:

- a `Benchmark`, whose planned rollouts are a lazy Daft DataFrame;
- a structural `Policy`, loaded once for every `evaluate()` call; and
- an `Evaluation` with lazy `episodes` and `steps` DataFrames.

The imperative boundary is deliberately small: an action depends on the previous
simulator observation, so the rollout loop is stateful. It does not perform nested
Daft writes. After every episode, Daft writes the step partition first and the
episode completion row last. Resume anti-joins only against outcomes whose steps
are contiguous and complete — a crash never silently drops or duplicates data.

## Install

The tested runtime is Python 3.12.

```bash
make setup
make check
```

The two policy environments cannot be combined — this is the usual VLA
reproduction wall, stated up front:

- OpenVLA is pinned to Torch 2.2, NumPy 1.x, and Transformers 4.40.1.
- VLA-JEPA is pinned to a LeRobot commit whose stack uses Torch 2.7+,
  NumPy 2.x, and Transformers 5.x.

Modal is the maintained installation path for both. For a Linux GPU machine,
install the package and simulator extra in each isolated policy environment.
Install VLA-JEPA's pinned LeRobot dependency as documented in
[`pyproject.toml`](pyproject.toml).

## All three benchmarks

```python
# Standard LIBERO.
standard = pae.libero("libero_goal", episodes=50, seed=7)

# Language perturbations; environments and fixed initial states come from
# the corresponding standard LIBERO-Goal task.
para = pae.libero_para(
    task_ids=[0, 1],
    paraphrase_types=["act", "obj"],
    episodes=50,
    seed=7,
)

# Published Pro BDDL and initial states, executed by the pinned Pro simulator.
pro = pae.libero_pro(
    "libero_spatial",
    perturbations=["lan"],
    episodes=50,
    seed=7,
)
```

OpenVLA has one checkpoint per base suite. Use the goal checkpoint for
LIBERO-Para; `physical_ai_evals.modal` resolves that mapping automatically.
VLA-JEPA uses its pinned LIBERO checkpoint for all suites.

## Bring your own policy

The point of a harness is evaluating *your* checkpoint. No inheritance or
source edit is required:

```python
from functools import partial
import numpy as np
import physical_ai_evals as pae

class MyPolicy:
    action_dim = 7
    control_mode = "relative"

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def reset(self, instruction):
        self.instruction = instruction

    def act(self, observation):
        return np.zeros(7, dtype=np.float32)

    def close(self):
        pass

policy = pae.PolicySpec(
    factory=partial(MyPolicy, "checkpoints/my-policy"),
    policy_id="my-team/my-policy",
    revision="git-commit-or-checkpoint-digest",
)
evaluation = pae.evaluate(
    policy,
    pae.libero("libero_spatial", task_ids=[0], episodes=2),
    out="data/evaluations",
)
```

The same wrapper pattern applies to a LeRobot policy: construct it inside
`factory` and map its action selection to `act`.

This works directly in a local or own-GPU Python process. A custom class used on
Modal must also be included in the container image; the bundled Modal entrypoint
only constructs the two built-in policies.

## Modal

One app exposes both policies while retaining two pinned images:

```bash
# One-time setup:
modal token new
modal secret create HF_TOKEN HF_TOKEN=...

# Real CPU simulator smoke (constructs, resets, and steps an environment)
# before spending GPU time:
make smoke-openvla BENCHMARK=libero SUITE=libero_spatial
make smoke-vla-jepa BENCHMARK=libero_para SUITE=libero_goal
make smoke-openvla BENCHMARK=libero_pro SUITE=libero_spatial PERTURBATIONS=lan

# GPU evaluations:
make rollout-openvla BENCHMARK=libero_pro SUITE=libero_spatial \
  PERTURBATIONS=lan TASKS=libero_spatial_lan:pick_up_the_bowl \
  EPISODES=5
make rollout-vla-jepa BENCHMARK=libero_para SUITE=libero_goal \
  PERTURBATIONS=act EPISODES=5
```

Modal commits its output volume after each completed episode, so a container
failure preserves all prior completions. Re-running the exact configuration
resumes it.

## Read LeRobot datasets with Daft

The flat `datasets` module is a thin, revision-checked layer over
`daft.datasets.lerobot`:

```python
from physical_ai_evals.datasets import ALOHA, lerobot_episodes

lerobot_episodes(ALOHA).select("episode_index", "tasks", "length").show()
```

See [Dataset readers](docs/datasets.md), [Evaluation protocol](docs/evaluation.md),
and [Troubleshooting](docs/troubleshooting.md).

## Citing

Citation metadata lives in [`CITATION.cff`](CITATION.cff). Please also cite
the upstream benchmark, policy, and checkpoint papers relevant to your
evaluation.

## License

Repository code is Apache-2.0. Upstream datasets, models, simulators, and
software retain their own terms; see [Third-party notices](THIRD_PARTY_NOTICES.md).
