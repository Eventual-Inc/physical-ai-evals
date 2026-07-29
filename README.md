# physical-ai-evals

[![CI](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Python tools for querying robot datasets, normalizing demonstrations, and evaluating VLA
policies on LIBERO. Rollouts use a shared one-row-per-step Parquet schema so they can be
queried with [Daft](https://github.com/Eventual-Inc/Daft).

The current `rollout-v2` schema stores actions, proprioceptive state, and end-effector
positions as fixed-shape Daft tensors, and reserves a typed 1,024-dimensional embedding
column. Read complete rollout parts with `daft.read_parquet()` so those logical types and
nullable tensor values are reconstructed as NumPy arrays.

## Install

Python 3.12 is the tested runtime.

```bash
make setup
make check
```

For HDF5 ingest without the development dependencies:

```bash
pip install -e ".[ingest_hdf5]"
```

## Query datasets

ALOHA, EgoDex, and ABC-130K use Daft's LeRobot v3 reader. Episode queries do not decode
video; `raw()` returns the lazy frame table and decodes cameras only when requested.

```python
from physical_ai_evals.datasets import abc, aloha, egodex

aloha.episodes().select("episode_index", "tasks", "length").show()

egodex.catalog(split="test").limit(10).show()
egodex.episodes("add_remove_lid", split="test").limit(5).show()

abc.episodes(
    repo_id=abc.SMOKE_REPO_ID,
    revision=abc.SMOKE_REVISION,
).limit(5).show()
```

The default sources are recorded at exact Hugging Face commits. Readers reject a changed
repository head instead of silently querying different data.

LIBERO-Para and LIBERO-PRO expose manifest-only task catalogs with revision-pinned BDDL paths:

```python
import daft
from physical_ai_evals.bench.libero import libero_para, libero_pro

para = libero_para.raw()
para = para.where(daft.col("environment_task_id") == 3).limit(5)
libero_para.instructions(para).show()

pro = libero_pro.raw()
pro = pro.where(
    (daft.col("suite") == "libero_spatial")
    & (daft.col("perturbation") == "lan")
).limit(5)
libero_pro.instructions(pro).show()
```

Daft also provides `daft.datasets.droid.raw()` directly. See
[Dataset catalogs](docs/datasets.md) for query behavior and source revisions.

## Published rollout trace

The historical LIBERO-Spatial trace is hosted on
[Hugging Face](https://huggingface.co/datasets/Eventual-Inc/physical-ai-evals-libero-spatial-pilot)
rather than committed to this repository. Query the immutable published revision directly:

```python
import daft

steps = daft.read_parquet(
    "hf://datasets/Eventual-Inc/physical-ai-evals-libero-spatial-pilot"
    "@ddb8a88fcc579ebf077a9ca2d1e026a7e1cf4429/steps.parquet"
)
```

The dataset card records the trace's protocol and provenance limitations.

## Ingest HDF5 demonstrations

The HDF5 adapter uses `daft.file.Hdf5File` and reads only the selected episodes and required
state/action arrays.

```bash
physical-ai-evals ingest \
  --source hdf5 \
  --input demos/libero_goal.hdf5 \
  --out data/rollouts
```

```python
from physical_ai_evals.ingest import Hdf5Ingestor

episodes = Hdf5Ingestor().load("demos/libero_goal.hdf5", limit=10)
```

## Evaluate policies

Resolve an evaluation plan without loading a model or simulator:

```bash
physical-ai-evals rollout \
  --policy vla_jepa \
  --suite libero_spatial \
  --task-ids 0 \
  --episodes 2 \
  --seed 7 \
  --dry-run
```

OpenVLA and VLA-JEPA use separate optional environments because their dependency constraints
conflict. The repository includes Modal entry points for both:

```bash
make smoke-openvla
make smoke-vla-jepa
```

Protocol differences that affect comparisons are listed in
[Evaluation protocol](docs/evaluation.md).

## Layout

```text
physical_ai_evals/  package source
tests/              unit and integration tests
docs/               dataset, evaluation, and troubleshooting reference
examples/notebooks/ small analysis examples
```

The main extension points are:

- [`Policy`](physical_ai_evals/policy/base.py) for model adapters.
- [`bench/libero.py`](physical_ai_evals/bench/libero.py) for the environment loop.
- [`Episode` and `Step`](physical_ai_evals/core/episode.py) for normalized records.
- [`Ingestor`](physical_ai_evals/ingest/base.py) for external datasets.

## License and citation

The repository code is Apache-2.0. Upstream datasets, models, simulators, and software retain
their own terms; see [Third-party notices](THIRD_PARTY_NOTICES.md). Citation metadata is in
[`CITATION.cff`](CITATION.cff).
