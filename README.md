# vla_jepa_harness

**You can get a VLA's success rate; you can't easily answer _why_ it fails.** This harness
turns thousands of LIBERO rollouts into a queryable [Daft](https://www.daft.ai)
DataFrame so you can _cluster failure modes_ — the re-grasp loops, the dropped objects,
the wrong-object grabs — instead of scrubbing videos and hand-rolling scripts.

Ingest **DROID / LeRobot / HDF5 / ALOHA / EgoDex / ABC** or run **your own policy** in LIBERO → one parquet
schema → mine the failures in Daft.

> Deliverable **1 of 3** for the Eventual VLA content project (harness · notebook · blog).
> The headline comparison is **VLA-JEPA vs OpenVLA on LIBERO**, and the hero result is
> **automatic re-grasp detection** (see [`notebooks/`](notebooks/README.md)).

---

## Status

The full harness is implemented and tested on CPU. OpenVLA has a Modal app with a
CPU image smoke path; full GPU rollouts still need deploy verification. VLA-JEPA uses
its own Modal app because the official StarVLA policy-server image is much heavier.

| Real (implemented + tested) | Not yet runnable here |
|---|---|
| `harness/schema.py` — rollout parquet schema | `harness/rollout/{rollout_udf,modal_app}.py` — OpenVLA Modal sweep (GPU rollout deploy-unverified) |
| `harness/ingest/*` — HDF5, raw-DROID, LeRobot, ALOHA, EgoDex & ABC ✅ | `harness/ingest/lerobot.py::_load_v21` — LeRobot v2.1 fallback (stub) |
| `harness/policies/{openvla,vla_jepa}.py` — VLA adapters ✅ (fake-backend/client tested) | `harness/rollout/modal_vla_jepa_app.py` — VLA-JEPA Modal sweep (heavy image, deploy-unverified) |
| `harness/rollout/{libero_runner,policy}.py` + `writer.py` incl. `RolloutWriter` ✅ (fake env/policy tested) | |
| `harness/config.py` · `cli.py` | |
| `tests/` — schema · hdf5/droid · daft-ingest · extra datasets · policies · rollout ✅ | |

Ingest reads HDF5/ALOHA/EgoDex/ABC ourselves and delegates LeRobot/DROID to Daft's own readers (PRs #7090/#7089,
temporarily vendored in `harness/_vendor` until our pinned Daft release ships them — see NOTES.md). The policy adapters
and the LIBERO rollout loop + `RolloutWriter` are implemented; the closed loop is tested against
a fake env/policy producing real `ROLLOUT_SCHEMA` parquet. The Modal sweep (`rollout_udf` wraps
the loop as a `@daft.cls` UDF; the Modal app owns the image/volumes/entrypoint) follows the
daft-examples conventions. Real-weight inference needs the GPU env.

## Quickstart

```bash
pip install -e ".[dev]"        # core deps only (pyarrow, numpy, daft, sklearn, imageio)
pytest                          # the schema round-trips through parquet

harness rollout --policy openvla --suite libero_goal --episodes 10 --dry-run
harness ingest  --source hdf5 --input demos/libero_goal.hdf5 --out data/rollouts --dry-run
```

`--dry-run` prints the resolved plan without importing any heavy policy/sim stack. Drop it
to actually run once the relevant extra/server environment is installed.

Heavy stacks are **optional extras** and several are **mutually incompatible** — install
each in its own environment (see [NOTES.md](NOTES.md)):

```bash
pip install -e ".[openvla]"        # transformers==4.40.1 — its own venv
pip install -e ".[vla_jepa]"       # lightweight VLA-JEPA WebSocket client deps
pip install -e ".[libero]"         # robosuite==1.4.0 — needs Python 3.8, its own conda env
pip install -e ".[ingest_hdf5]"    # h5py    (or ingest_lerobot / ingest_droid)
pip install -e ".[ingest_aloha]"   # h5py    (or ingest_egodex / ingest_abc)
pip install -e ".[embed]"          # sentence-transformers for the clustering pass
```

## Using Real ABC Data

ABC publishes its data tooling separately at <https://abc.bot/> / `amazon-far/abc`,
and the Hugging Face tree can be queried before you download anything:

```bash
harness abc-query --split train --contains bottles --limit 10
harness abc-query --split train --task organize_the_condiment_bottles --limit 5
harness abc-query --split train \
  --task organize_the_condiment_bottles \
  --episode episode_000fced0-d49c-49c7-8615-debb589a97ec
```

Use ABC's downloader/converter only after you have picked a small subset. The ABC
training-format episode directory is exactly what `harness ingest --source abc` expects:

```
episode_<uuid>/
  states_actions.bin
  combined_camera-images-rgb.mp4
  episode_metadata.json
```

Minimal preview path:

```bash
git clone https://github.com/amazon-far/abc.git
cd abc
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv python pin 3.12
uv sync
uv run prepare.py                                # preview data, about 130MB
```

That creates an ABC cache, by default `abc/cache/`, with folders like
`train_real/episode_<uuid>/...` and `val_real/...`. From this repo:

```bash
pip install -e ".[dev,ingest_abc]"
harness ingest --source abc --input /path/to/abc/cache --out data/rollouts
```

For the gated full ABC-130K MCAP data, accept access on Hugging Face and use ABC's
converter, then ingest the converted output:

```bash
export HF_TOKEN=...
cd /path/to/abc
uv run export_hf_task.py --task organize_the_condiment_bottles --split train --max-episodes 1

cd /path/to/vla_jepa_harness
harness ingest --source abc --input /path/to/abc/cache/train_real --out data/rollouts
```

## Running rollouts on Modal

Rollouts run on Modal GPUs as a `@daft.cls` UDF — one episode spec per row, the per-step
trajectory written to a parquet glob the notebook reads. Follows the daft-examples
`models/<name>/modal_app.py` convention (shared `daft-model-cache` / `daft-model-outputs`
Volumes, `hf-token` Secret). OpenVLA and VLA-JEPA use separate Modal apps so the StarVLA
image does not block OpenVLA deploys.

```bash
pip install -e ".[modal]"
modal run harness/rollout/modal_app.py --policy-type openvla  --download-only
modal run harness/rollout/modal_app.py --policy-type openvla  --suites libero_goal --episodes 5
modal run harness/rollout/modal_vla_jepa_app.py --download-only
modal run harness/rollout/modal_vla_jepa_app.py --suites libero_goal --episodes 5
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
    aloha.py           real      ALOHA / Mobile ALOHA HDF5 -> Episode
    egodex.py          real      EgoDex HDF5 annotations + MP4 path -> Episode
    abc.py             real      ABC exported states_actions.bin episodes (abc.bot) -> Episode
  rollout/
    policy.py          CONCRETE  Policy ABC (reset / act) + Observation
    libero_runner.py   real      make_env / run_episode / run_sweep (OpenPI-faithful)
    rollout_udf.py     real*     @daft.cls LIBERO rollout UDF (*deploy-unverified: GPU+sim)
    modal_app.py       real*     OpenVLA Modal deployment shell
    modal_vla_jepa_app.py real*  VLA-JEPA Modal shell + policy-server subprocess
  policies/
    openvla.py         real      OpenVLA baseline (lazy load; GPU for real inference)
    vla_jepa.py        real      VLA-JEPA headliner via official WebSocket policy server
  _vendor/             TEMP      vendored Daft #7090/#7089 readers — delete after Daft pin bump
tests/*.py             real      schema · hdf5/droid · daft-ingest · extra datasets · policies · rollout
notebooks/README.md    outline   comparative failure-mode notebook (deliverable 2)
NOTES.md               the reproducibility-gotchas log (feeds the blog, deliverable 3)
BACKLOG.md             explicitly-not-shipping list + idea parking lot
```

## The three deliverables

1. **Harness** (this repo) — reproducible rollouts → parquet.
2. **Notebook** — comparative failure-mode analysis (VLA-JEPA vs OpenVLA), re-grasp
   detection as the screenshot. Outline: [`notebooks/README.md`](notebooks/README.md).
3. **Blog post + social** — including the reproducibility story from [NOTES.md](NOTES.md).
