# vla_jepa_harness

**You can get a VLA's success rate; you can't easily answer _why_ it fails.** This harness
turns thousands of LIBERO rollouts into a queryable [Daft](https://www.daft.ai)
DataFrame so you can _cluster failure modes_ — the re-grasp loops, the dropped objects,
the wrong-object grabs — instead of scrubbing videos and hand-rolling scripts.

Ingest **DROID / LeRobot / HDF5** or run **your own policy** in LIBERO → one parquet
schema → mine the failures in Daft.

> Deliverable **1 of 3** for the Eventual VLA content project (harness · notebook · blog).
> The headline comparison is **VLA-JEPA vs OpenVLA on LIBERO**, and the hero result is
> **automatic re-grasp detection** (see [`notebooks/`](notebooks/README.md)).

---

## Status

The full harness is implemented and tested on CPU (30 tests). The Modal sweep is built but
**not yet deploy-verified** (needs a GPU + real LIBERO).

| Real (implemented + tested) | Not yet runnable here |
|---|---|
| `harness/schema.py` — rollout parquet schema | `harness/rollout/{rollout_udf,modal_app}.py` — Modal sweep (GPU+LIBERO; deploy-unverified) |
| `harness/ingest/*` — HDF5, raw-DROID & LeRobot (v3 via vendored Daft #7090/#7089) ✅ | `harness/ingest/lerobot.py::_load_v21` — LeRobot v2.1 fallback (stub) |
| `harness/policies/{openvla,vla_jepa}.py` — VLA adapters ✅ (fake-backend tested) | |
| `harness/rollout/{libero_runner,policy}.py` + `writer.py` incl. `RolloutWriter` ✅ (fake env/policy tested) | |
| `harness/config.py` · `cli.py` | |
| `tests/` — schema · hdf5/droid · daft-ingest · policies · rollout (**30 ✅**) | |

Ingest reads HDF5 ourselves and delegates LeRobot/DROID to Daft's own readers (PRs #7090/#7089,
temporarily vendored in `harness/_vendor` until they land — see NOTES.md). The policy adapters
and the LIBERO rollout loop + `RolloutWriter` are implemented; the closed loop is tested against
a fake env/policy producing real `ROLLOUT_SCHEMA` parquet. The Modal sweep (`rollout_udf` wraps
the loop as a `@daft.cls` UDF; `modal_app` is the deployment shell) follows the daft-examples
conventions but is **not yet deploy-verified** — the open question is LIBERO + a VLA policy
coexisting in one container image (NOTES.md). Real-weight inference needs that GPU env.

## Quickstart

```bash
pip install -e ".[dev]"        # core deps only (pyarrow, numpy, daft, sklearn, imageio)
pytest                          # the schema round-trips through parquet

harness rollout --policy openvla --suite libero_goal --episodes 10 --dry-run
harness ingest  --source hdf5 --input demos/libero_goal.hdf5 --out data/rollouts --dry-run
```

`--dry-run` prints the resolved plan without importing any heavy policy/sim stack. Drop it
to actually run (once the stubs are implemented and the relevant extra is installed).

Heavy stacks are **optional extras** and several are **mutually incompatible** — install
each in its own environment (see [NOTES.md](NOTES.md)):

```bash
pip install -e ".[openvla]"        # transformers==4.40.1 — its own venv
pip install -e ".[vla_jepa]"       # modern lerobot/Qwen3-VL — its own venv
pip install -e ".[libero]"         # robosuite==1.4.0 — needs Python 3.8, its own conda env
pip install -e ".[ingest_hdf5]"    # h5py    (or ingest_lerobot / ingest_droid)
pip install -e ".[embed]"          # sentence-transformers for the clustering pass
```

## Running rollouts on Modal

Rollouts run on Modal GPUs as a `@daft.cls` UDF — one episode spec per row, the per-step
trajectory written to a parquet glob the notebook reads. Follows the daft-examples
`models/<name>/modal_app.py` convention (shared `daft-model-cache` / `daft-model-outputs`
Volumes, `hf-token` Secret). **Not yet deploy-verified** — see the image-coexistence note in
NOTES.md before the first run.

```bash
pip install -e ".[modal]"
modal run harness/rollout/modal_app.py --policy-type openvla  --download-only
modal run harness/rollout/modal_app.py --policy-type openvla  --suites libero_goal --episodes 5
modal run harness/rollout/modal_app.py --policy-type vla_jepa --suites libero_goal --episodes 5
```

## Reading the output in Daft

The harness writes a directory of parquet part files (one per episode), **one row per
step**, that Daft reads as a glob:

```python
import daft
df = daft.read_parquet("data/rollouts/*.parquet")
failures = df.where(df["success"] == False)       # the wedge: only the failures
failures.groupby("terminal_failure").count().show()
```

## Repo map

```
harness/
  schema.py            CONCRETE  rollout parquet schema (one row per step)
  config.py            CONCRETE  RolloutConfig / IngestConfig / EmbedConfig
  writer.py            real      write_rows/write_episode + RolloutWriter (streaming capture)
  cli.py               real      `harness rollout` / `harness ingest`
  _modal.py            real      modal-free deploy infra (paths, HF cache, weight resolution)
  ingest/
    base.py            CONCRETE  Episode / Step / Ingestor + to_step_rows()
    hdf5.py            real      robomimic/LIBERO HDF5 -> Episode
    droid.py           real      raw-DROID via daft.datasets.droid.raw() + trajectory.h5
    lerobot.py         real      LeRobot v3 via daft.datasets.lerobot.read() (v2.1 = stub)
  rollout/
    policy.py          CONCRETE  Policy ABC (reset / act) + Observation
    libero_runner.py   real      make_env / run_episode / run_sweep (OpenPI-faithful)
    rollout_udf.py     real*     @daft.cls LIBERO rollout UDF (*deploy-unverified: GPU+sim)
    modal_app.py       real*     Modal deployment shell (image/volumes/entrypoints)
  policies/
    openvla.py         real      OpenVLA baseline (lazy load; GPU for real inference)
    vla_jepa.py        real      VLA-JEPA headliner via LeRobot PreTrainedPolicy
  _vendor/             TEMP      vendored Daft #7090/#7089 readers — delete when merged
tests/*.py             real      schema · hdf5/droid · daft-ingest · policies · rollout (30 tests)
notebooks/README.md    outline   comparative failure-mode notebook (deliverable 2)
NOTES.md               the reproducibility-gotchas log (feeds the blog, deliverable 3)
BACKLOG.md             explicitly-not-shipping list + idea parking lot
```

## The three deliverables

1. **Harness** (this repo) — reproducible rollouts → parquet.
2. **Notebook** — comparative failure-mode analysis (VLA-JEPA vs OpenVLA), re-grasp
   detection as the screenshot. Outline: [`notebooks/README.md`](notebooks/README.md).
3. **Blog post + social** — including the reproducibility story from [NOTES.md](NOTES.md).
