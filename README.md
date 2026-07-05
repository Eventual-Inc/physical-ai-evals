# vla_jepa_harness

This repository provides a modern, unified Python environment for running robotics model
benchmarks across VLA, JEPA, LIBERO, MuJoCo, and robosuite.

Many academic robotics repositories ship with strict dependency pins and isolated environment
assumptions. Those constraints often make it difficult to reproduce results, compare models,
or integrate research code into production-grade systems. After testing the dependency
requirements across several fragmented implementations, we found that many of these
constraints were not fundamental. They could be resolved with standard engineering practices,
modern Python tooling, and careful compatibility fixes.

The purpose of this project is to reduce that friction. It gives physical AI researchers and
industry practitioners a coherent starting point for benchmarking new models without needing
to reconstruct a fragile environment for every paper or baseline. The repository includes the
compatibility work, bug fixes, and workflow templates needed to run these systems in a single
environment that can also scale on [Modal](https://modal.com).

The aim is not only reproducibility, but **operational reproducibility**: the ability to run,
modify, compare, scale, and extend research systems in a way that matches how real engineering
teams work.

**And one number is not enough.** You can get a success rate; you can't easily answer _why_
your VLA fails. This repo makes *"run VLA-JEPA and OpenVLA on LIBERO, on your own GPU, end to
end — and see why it fails"* a solved, repeatable thing:

1. **Reproduce** — both policies run **in-process** on Modal GPUs against the canonical LIBERO
   protocol (50 trials/task, seed 7 — the OpenVLA-origin constants that openpi, starVLA, and
   allenai/vla-eval all inherit; see [docs/EVAL_PATTERNS.md](docs/EVAL_PATTERNS.md)). Every
   dependency landmine we hit is logged in [NOTES.md](NOTES.md).
2. **Understand** — every rollout streams to a canonical **one-row-per-step parquet schema**
   ([`harness/schema.py`](harness/schema.py)), so "success rate dropped" decomposes into
   *policy* failures (re-grasp loops, drops) vs *harness* failures (bad unnorm ⇒ saturated
   actions; bad init state; preprocessing drift) with a Daft query instead of scrubbing video.
3. **Generalize** — the same `Episode`/`Step` representation ingests
   **DROID · LeRobot · HDF5 · ALOHA · EgoDex · ABC**, so your demonstration data and your
   eval rollouts land in one queryable frame.

**The stack is deliberately opinionated:** Python end to end — [Daft](https://daft.ai) as the
data plane (datasets → DataFrames in, parquet out), Modal for GPU placement, PyTorch policies
**in-process** (no policy server, no WebSocket — the container boundary and `@daft.cls` worker
replace them), MuJoCo/robosuite for sim. Where the tech-agnostic route
(allenai/vla-evaluation-harness) buys generality with a mandatory network boundary and
per-benchmark Docker, this buys reproducibility with one process you can read top to bottom.

## Quickstart

```bash
pip install -e ".[dev]"
# daft.datasets.lerobot + Hdf5File are on Daft nightly until the next release:
pip install daft --pre --extra-index-url https://nightly.daft.ai
pytest                          # 39 tests, CPU-only, no weights/sim needed

harness rollout --policy vla_jepa --suite libero_spatial --task-ids 0 --episodes 2 --dry-run
harness ingest  --source hdf5 --input demos/libero_goal.hdf5 --out data/rollouts --dry-run
```

`--dry-run` prints the resolved plan without importing any heavy policy/sim stack.

**Python: one interpreter — 3.12 — everywhere** (core supports 3.10–3.13; the `vla_jepa`
extra needs ≥3.12 because lerobot does). The two policy stacks still need **separate
environments/images**, but the split is the *transformers pin* (OpenVLA ==4.40.1 vs
VLA-JEPA's 5.4–5.6), never the Python version — the old "LIBERO needs Python 3.8" story is
a myth we falsified on Modal (see [NOTES.md](NOTES.md)):

```bash
pip install -e ".[openvla]"        # transformers==4.40.1 stack (py3.12-verified)
pip install -e ".[vla_jepa]"       # lerobot@pinned-SHA (Qwen3-VL / transformers 5.x, py>=3.12)
pip install -e ".[ingest_hdf5]"    # h5py (also: ingest_aloha / ingest_egodex / ingest_abc)
pip install -e ".[embed]"          # sentence-transformers for the clustering pass
```

## Running rollouts on Modal

Rollouts run as a `@daft.cls` UDF on Modal GPUs — one episode spec per row, per-step
trajectories written to a parquet glob. Two apps (both Python 3.12) because the transformers
pins conflict (OpenVLA ==4.40.1; VLA-JEPA's lerobot stack 5.4–5.6):

```bash
pip install -e ".[modal]"

# OpenVLA (image verified: builds + LIBERO imports green)
modal run harness/rollout/modal_app.py --policy-type openvla --suites libero_spatial --task-ids 0 --episodes 2

# VLA-JEPA — in-process via the lerobot port + lerobot/VLA-JEPA-LIBERO checkpoint.
# No policy server: lerobot[vla_jepa,libero] puts the policy AND the sim in one process
# (hf-libero ships LIBERO's bddl/assets in the wheel — no git clone, no config patching).
modal run harness/rollout/modal_vla_jepa_app.py --smoke-test
modal run harness/rollout/modal_vla_jepa_app.py --download-only
modal run harness/rollout/modal_vla_jepa_app.py --suites libero_spatial --task-ids 0 --episodes 2
```

## Reading the output in Daft

One row per step, one part file per episode, one glob:

```python
import daft
df = daft.read_parquet("data/rollouts/*.parquet")
failures = df.where(df["success"] == False)        # the wedge: only the failures
failures.groupby("terminal_failure").count().show()
```

[`notebooks/regrasp_demo.py`](notebooks/regrasp_demo.py) runs the whole failure-forensics
loop on synthetic rollouts (no GPU): Daft glob → re-grasp detector over the per-step
gripper/object signal → the annotated grasp→lift→drop→re-grasp trajectory plot.

## Ingesting datasets

Six sources normalize onto the same `Episode`/`Step` waist and emit the identical schema —
`daft.datasets.{lerobot,droid}` do the reading where Daft is native, our adapters do the rest:

```bash
harness ingest --source lerobot --input org/dataset-name --out data/rollouts
harness ingest --source droid   --input /path/to/droid_raw --out data/rollouts
harness ingest --source aloha   --input demos/aloha_task   --out data/rollouts
```

### ABC (abc.bot)

ABC publishes tooling at `amazon-far/abc`; query the gated HF tree before downloading:

```bash
harness abc-query --split train --contains bottles --limit 10
```

Use ABC's downloader/converter for a small subset, then ingest the exported episodes
(`episode_<uuid>/{states_actions.bin, combined_camera-images-rgb.mp4, episode_metadata.json}`):

```bash
pip install -e ".[dev,ingest_abc]"
harness ingest --source abc --input /path/to/abc/cache --out data/rollouts
```

## Repo map

```
harness/
  schema.py            the rollout parquet schema (one row per step) — the contract
  config.py            RolloutConfig / IngestConfig / EmbedConfig (canonical protocol defaults)
  writer.py            write_rows/write_episode + RolloutWriter (streaming capture)
  cli.py               harness rollout / ingest / abc-query
  _modal.py            modal-free deploy infra (paths, HF cache, weight resolution)
  ingest/
    base.py            Episode / Step / Ingestor + to_step_rows()
    lerobot.py         LeRobot v3 via daft.datasets.lerobot.read()
    droid.py           raw DROID via daft.datasets.droid.raw() + our trajectory.h5 parser
    hdf5.py            robomimic/LIBERO HDF5
    aloha.py           ALOHA / Mobile ALOHA HDF5 (robot-native joint actions)
    egodex.py          EgoDex HDF5 annotations + MP4 paths (human egocentric)
    abc.py             ABC exported episodes  ·  abc_query.py: HF metadata queries
  rollout/
    policy.py          Policy ABC (reset / act) — the seam every backend implements
    libero_runner.py   make_env / run_episode / run_sweep (protocol-faithful closed loop)
    rollout_udf.py     @daft.cls LIBERO rollout UDF (one episode-spec row -> one episode)
    modal_app.py       OpenVLA Modal app (image verified)
    modal_vla_jepa_app.py  VLA-JEPA Modal app — in-process lerobot, no server (GPU-unverified)
  policies/
    openvla.py         OpenVLA in-process via HF predict_action (suite-name unnorm_key)
    vla_jepa.py        VLA-JEPA in-process via the lerobot port + hub checkpoint
docs/EVAL_PATTERNS.md  the VLA evaluation grammar: 9 components, model x benchmark matrix, lexicon
notebooks/             regrasp_demo.py (failure forensics on synthetic data) + notebook outline
NOTES.md               the reproducibility-gotchas log — every landmine, with fixes
BACKLOG.md             explicitly-not-shipping list + idea parking lot
tests/                 39 CPU-only tests: schema · ingest (all six) · policies · rollout loop
```

## The three deliverables

1. **Harness** (this repo) — reproducible rollouts → parquet.
2. **Notebook** — comparative failure-mode analysis (VLA-JEPA vs OpenVLA), re-grasp detection
   as the hero. Outline: [`notebooks/README.md`](notebooks/README.md).
3. **Blog post + social** — including the reproducibility story from [NOTES.md](NOTES.md).
