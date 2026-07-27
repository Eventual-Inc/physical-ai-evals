# physical-ai-evals

[![CI](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

An evaluation harness for running VLA policies on LIBERO and preserving the result as
queryable, one-row-per-step Parquet. The repository currently ships OpenVLA and VLA-JEPA
adapters, local and Modal execution paths, a normalized episode schema, and an exploratory
paired pilot artifact.

The project is intended as research infrastructure. Its strongest claim is that an evaluation
can be inspected and extended from recorded trajectories; it does **not** claim that the pilot
below is a reproduction of a published leaderboard result.

## Exploratory paired pilot

The artifact labeled `2026-07-02` contains the two policies evaluated on the same 100
LIBERO-Spatial episode specifications: 10 tasks × fixed initial-state indices 0–9. The label is
an artifact identifier; the execution timestamp was not recorded. This is a paired exploratory
case study, not the 50-trials-per-task LIBERO reference protocol.

| Policy | Successes | Rate | Descriptive 95% Wilson interval |
|---|---:|---:|---:|
| VLA-JEPA | 99/100 | 99% | 94.6%–99.8% |
| OpenVLA | 84/100 | 84% | 75.6%–89.9% |

The Wilson intervals treat episodes as independent and identically distributed. That
assumption is not justified by ten fixed initial states nested within ten tasks, so the
intervals are descriptive only; they are not uncertainty estimates for a task distribution or
a new benchmark sample.

The paired outcomes expose more information than the two marginal rates:

| VLA-JEPA outcome | OpenVLA outcome | Episode specifications |
|---|---|---:|
| success | success | 84 |
| success | failure | 15 |
| failure | success | 0 |
| failure | failure | 1 |

Per-task successes, with ten fixed initial states for each policy:

| LIBERO-Spatial task id | VLA-JEPA | OpenVLA |
|---:|---:|---:|
| 0 | 10/10 | 10/10 |
| 1 | 10/10 | 8/10 |
| 2 | 10/10 | 9/10 |
| 3 | 10/10 | 9/10 |
| 4 | 10/10 | 7/10 |
| 5 | 9/10 | 6/10 |
| 6 | 10/10 | 9/10 |
| 7 | 10/10 | 10/10 |
| 8 | 10/10 | 7/10 |
| 9 | 10/10 | 9/10 |

The self-contained evidence bundle is in
[`results/libero-spatial-pilot-2026-07-02/`](results/libero-spatial-pilot-2026-07-02/).
Start with its [`README.md`](results/libero-spatial-pilot-2026-07-02/README.md), then inspect
the machine-readable [`manifest.json`](results/libero-spatial-pilot-2026-07-02/manifest.json),
[`summary.json`](results/libero-spatial-pilot-2026-07-02/summary.json), and
[`episodes.csv`](results/libero-spatial-pilot-2026-07-02/episodes.csv). `SHA256SUMS` binds the
files in the bundle.

### Pilot limitations

- The stored episode seed and filenames say `7`, because `7` was requested. The referenced
  historical Modal path constructed LIBERO with `env_seed=0`, but the exact executed code
  revision is unavailable; treat the simulator seed as 0/unverified. Policy RNG state was
  neither explicitly controlled nor recorded. The artifact therefore does not establish a
  seed-7 simulator run or bit-for-bit replayability.
- Initial-state indices 0–9 were fixed and paired across policies, but there were only ten
  trials per task. The LIBERO reference evaluation uses 50 trials per task.
- The OpenVLA Parquet rows store an empty `model` field. The expected checkpoint was
  `openvla/openvla-7b-finetuned-libero-spatial`, but the raw artifact does not independently
  establish its resolved checkpoint revision. VLA-JEPA records
  `lerobot/VLA-JEPA-LIBERO`, also without a revision.
- All 17 failed episodes have raw `terminal_failure="unlabeled"`. The notebook's 15 OpenVLA
  and 1 VLA-JEPA "repeated-close" assignments are automated, post-hoc candidates based on
  gripper commands and finger separation. They have not been manually validated as object
  drops and reacquisitions.
- A `rollout-v1` row mixes `obs_t` image/state and `action_t` with post-action reward,
  terminal flag, end-effector position, and gripper state from `obs_{t+1}`; the terminal next
  frame is absent. See [row timing](docs/EVAL_PATTERNS.md#rollout-v1-row-timing).
- The traces can support diagnosis hypotheses, but they do not by themselves identify whether
  a behavior was caused by a policy, preprocessing, simulator state, or another harness choice.

These limitations are preserved in the bundle rather than repaired retrospectively. A future
protocol-conformant rerun should pin code and checkpoint revisions, record resolved
configuration, control policy and simulator RNG, and use 50 initial states per task.

## Quickstart

Python 3.12 is the recommended development interpreter. The two policy stacks use separate
environments because their `transformers` requirements conflict.

```bash
make setup
make check

# Resolve plans without importing a policy or simulator stack.
.venv/bin/harness rollout --policy vla_jepa --suite libero_spatial \
  --task-ids 0 --episodes 2 --seed 7 --dry-run
.venv/bin/harness ingest --source hdf5 --input demos/libero_goal.hdf5 \
  --out data/rollouts --dry-run
```

`make setup` installs the CPU policy-adapter test dependencies; `make check` runs the full
lint, type, and test gate. `uv sync && uv run pytest` is a lighter core-only path, but optional
policy-adapter tests may skip. `make docs-build` builds this documentation in strict mode.

Install optional local stacks only when needed:

```bash
pip install -e ".[openvla]"
pip install "lerobot[vla_jepa] @ git+https://github.com/huggingface/lerobot@052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34"
pip install -e ".[ingest_hdf5]"
```

## Modal rollouts

Each policy has its own Modal image. The episode worker invokes the selected policy and LIBERO
simulator together; this deployment choice does not imply that both policy stacks share one
environment or that the architecture is preferable for every evaluator.

One-time setup:

```bash
# `make setup` already includes Modal. For a Modal-only environment instead:
uv sync --frozen --extra modal
.venv/bin/modal token new
.venv/bin/modal secret create hf-token HF_TOKEN=<your-hf-token>
```

Small explicit sweeps:

```bash
# OpenVLA
.venv/bin/modal run harness/cloud/openvla_app.py \
  --suites libero_spatial --task-ids 0 --episodes 2 --seed 7 \
  --model-id openvla/openvla-7b-finetuned-libero-spatial \
  --model-revision 962318cec55ac10993ff0f5f43eda9a270b4c873

# VLA-JEPA
.venv/bin/modal run harness/cloud/vla_jepa_app.py \
  --suites libero_spatial --task-ids 0 --episodes 2 --seed 7 \
  --model-id lerobot/VLA-JEPA-LIBERO \
  --model-revision 735d9f692981e286ade093b5046627eda876e5d0
```

Before interpreting a new result, verify the emitted manifest/configuration and confirm the
resolved suite-to-checkpoint mapping. See [Evaluation patterns](docs/EVAL_PATTERNS.md) for the
upstream protocol and local deviations.

## Querying outcomes and candidate signatures

`success` is denormalized onto every step. Aggregate by both policy and episode; the same
episode id intentionally appears under both policies.

```python
import daft

steps = daft.read_parquet("data/rollouts/*/*.parquet")
episodes = steps.groupby("policy_type", "episode_id").agg(
    daft.col("success").any_value().alias("success"),
    daft.col("step_idx").count().alias("steps"),
)
failures = episodes.where(episodes["success"] == False)
failures.groupby("policy_type").agg(
    daft.col("episode_id").count().alias("failed_episodes")
).show()
```

Do not group the pilot by `terminal_failure` expecting behavioral classes: all raw failures are
`unlabeled`. [`notebooks/failure_modes.py`](notebooks/failure_modes.py) demonstrates an
explicit post-hoc repeated-close **candidate** detector. Its thresholds and labels require
validation against video or manual annotation before being treated as failure modes. Its
action-to-gripper features are transition-aligned; other row fields are not a simultaneous
snapshot.

## Repository map

```text
harness/
  core/       episode model, schema, configuration, writer, geometry
  policy/     Policy interface plus OpenVLA and VLA-JEPA adapters
  bench/      LIBERO environment and rollout loop
  cloud/      Modal images, sweep enumeration, and rollout UDF
  ingest/     normalized dataset adapters
  analysis/   exploratory behavioral-signature helpers
docs/         protocol crosswalk, provenance notes, and observed friction points
notebooks/    paired-pilot analysis and a separate synthetic detector demo
results/      immutable, checksummed research artifacts
tests/        CPU-oriented unit and integration tests
```

The stable extension seams are:

1. Implement [`Policy`](harness/policy/base.py) with `reset(instruction)` and `act(obs)`.
2. Adapt a benchmark to the gym-shaped loop in [`harness/bench/libero.py`](harness/bench/libero.py).
3. Produce [`Episode`](harness/core/episode.py) and `Step` values and serialize them with the
   shared writer; [`harness/ingest/hdf5.py`](harness/ingest/hdf5.py) is one example.

## Research lineage and citation

The implementation builds on the primary projects below. The links point to their papers or
source repositories; they do not imply endorsement of this pilot.

- [LIBERO paper](https://arxiv.org/abs/2306.03310) and
  [official implementation](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [OpenVLA paper](https://arxiv.org/abs/2406.09246) and
  [official implementation](https://github.com/openvla/openvla)
- [VLA-JEPA implementation](https://github.com/ginwind/VLA-JEPA) and the
  [LeRobot checkpoint revision pinned for future runs](https://huggingface.co/lerobot/VLA-JEPA-LIBERO/tree/735d9f692981e286ade093b5046627eda876e5d0)
- [OpenVLA-Spatial checkpoint revision pinned for future runs](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial/tree/962318cec55ac10993ff0f5f43eda9a270b4c873)
- [LeRobot source revision used by the Modal image](https://github.com/huggingface/lerobot/tree/052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34)
- [Daft](https://github.com/Eventual-Inc/Daft), used for the rollout data plane

Please cite the software metadata in [`CITATION.cff`](CITATION.cff), and cite the upstream
papers/models relevant to the policy and benchmark you use.

## License

Apache-2.0 applies to this repository's code. Upstream software, model weights, datasets, and
assets remain subject to their own terms; see [Third-party notices](THIRD_PARTY_NOTICES.md).
Built by [Eventual](https://eventual.ai), the team behind [Daft](https://daft.ai).
