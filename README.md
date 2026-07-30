# physical-ai-evals

[![CI](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Run OpenVLA and VLA-JEPA on LIBERO, LIBERO-Para, and LIBERO-Pro from one Python
3.12 API. The stateful model/simulator loop stays in one process; Daft handles
episode specification, resume planning, typed Parquet writes, lazy reads, and
metrics.

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

## Why this shape

An evaluation has three useful objects:

- a `Benchmark`, whose episode specifications are a lazy Daft DataFrame;
- a structural `Policy`, loaded once for every `evaluate()` call; and
- an `Evaluation` with lazy `episodes` and `steps` DataFrames.

The imperative boundary is deliberately small: an action depends on the previous
simulator observation, so the rollout loop is stateful. It does not perform nested
Daft writes. After every episode, Daft writes the step partition first and the
episode completion row last. Resume anti-joins only against outcomes whose steps
are contiguous and complete.

## Install

The tested runtime is Python 3.12.

```bash
make setup
make check
```

The two policy environments cannot be combined:

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

No inheritance or source edit is required:

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

This works directly in a local or own-GPU Python process. A custom class used on
Modal must also be included in the container image; the bundled Modal entrypoint
only constructs the two built-in policies.

## Output and queries

```text
<out>/<evaluation_id>/
  manifest.json
  episodes/episode_key=.../*.parquet
  steps/episode_key=.../*.parquet
  videos/<episode_key>/primary.mp4
  videos/<episode_key>/wrist.mp4
```

The normalized layout follows LeRobot's episode/step/video concepts without
claiming to be a complete LeRobot training dataset. Constant evaluation
provenance lives in the manifest; episode metadata and outcomes are not repeated
on every transition. Video paths are relative to the evaluation directory.

```python
run = pae.read_evaluation("data/evaluations/<evaluation_id>")
failures = run.episodes.where(run.episodes["success"] == False)
failure_steps = failures.join(run.steps, on="episode_key")
failure_steps.select("task_key", "frame_index", "action").show()
```

The historical published pilot trace remains `rollout-v1`. It is intentionally
version-noted rather than presented as an `eval-v1` reproduction fixture; see
[Evaluation protocol](docs/evaluation.md).

## Modal

One app exposes both policies while retaining two images:

```bash
# One-time setup:
modal token new
modal secret create hf-token HF_TOKEN=...

# Real CPU simulator smoke (constructs, resets, and steps an environment):
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

## License

Repository code is Apache-2.0. Upstream datasets, models, simulators, and
software retain their own terms; see [Third-party notices](THIRD_PARTY_NOTICES.md).
