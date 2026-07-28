# Dataset catalogs

## ALOHA

```python
from physical_ai_evals.datasets import aloha

episodes = aloha.episodes()
episodes.select("episode_index", "tasks", "length").show()

frames = aloha.raw(load_video_frames=False)
frames.where(frames["episode_index"] == 0).limit(100).show()
```

`episodes()` returns one row per episode. `raw()` returns one row per frame and leaves video
encoded unless `load_video_frames` names a camera. The default public sample is
`lerobot/aloha_mobile_shrimp` at revision
`6e828202059d2cc204b61ff968c232d202127a34`.

## EgoDex

The EgoDex conversion stores each activity and split as an independent LeRobot dataset.
Inspect that structure before selecting an activity:

```python
from physical_ai_evals.datasets import egodex

egodex.catalog(split="test").select("task_name", "dataset_uri").show()
episodes = egodex.episodes("add_remove_lid", split="test")
episodes.limit(5).show()
```

`catalog()` queries only the Hugging Face manifest. `raw(activity, split=...)` returns the
selected frame table and accepts Daft's `load_video_frames` option. The default source is
`griffinlabs/EgoDex-LeRobot-v3.0` at revision
`41d60b449629b2181ff5b735d31c2a2cf8b3cad8`.

## ABC-130K

```python
from physical_ai_evals.datasets import abc

episodes = abc.episodes(
    repo_id=abc.SMOKE_REPO_ID,
    revision=abc.SMOKE_REVISION,
)
episodes.select("episode_index", "tasks", "length").show()
```

`abc.raw()` defaults to the public `lerobot/abc_130k_v3_train` conversion at revision
`68651e4929d9fb00f798937b2d62617cab5c771d`. The smaller
`lerobot/abc_130k_v3_smoke` source at revision
`b342a0ff262195d49bae3eece6e3f40c6e1dbe15` is useful for testing queries.

The original `XDOF/ABC-130k` repository is gated and stores raw MCAP data. These helpers
query the public LeRobot conversion; they do not bypass the original repository's access
terms.

## LIBERO-Para

```python
import daft
from physical_ai_evals.datasets import libero_para

tasks = libero_para.raw()
tasks = tasks.where(
    (daft.col("environment_task_id") == 3)
    & (daft.col("paraphrase_type") == "obj")
).limit(20)
tasks = libero_para.instructions(tasks)
tasks.show()
```

`raw()` returns one row per published BDDL variant with:

| Column | Meaning |
|---|---|
| `environment_suite` | Base LIBERO suite used by the tasks |
| `environment_task_id` | Base environment task |
| `paraphrase_type` | `act`, `obj`, or `comp` |
| `paraphrase_key` | Perturbed language component |
| `variant_id` | Published variant number |
| `bddl_path` | Revision-pinned `hf://` path |

The default source is `HAI-Lab/LIBERO-Para` at revision
`d306f66f8b441cad1155b21a3f69e440079c81c9`.

## LIBERO-PRO

```python
import daft
from physical_ai_evals.datasets import libero_pro

tasks = libero_pro.raw()
tasks = tasks.where(
    (daft.col("suite") == "libero_spatial")
    & (daft.col("perturbation") == "lan")
).limit(20)
tasks = libero_pro.instructions(tasks)
tasks.show()
```

`raw()` returns one row per published BDDL task with its base `suite`, `suite_variant`,
`perturbation`, `bddl_path`, and an `init_path` when the repository contains a matching
initial-state file.

The default source is `zhouxueyang/LIBERO-Pro` at revision
`c86fc3b8293185a6f373677018ff3e37f8391602`.

## Query behavior

The ALOHA, EgoDex, and ABC readers verify the current Hub commit before constructing Daft's
LeRobot read plan. Daft 0.7.21 reads the remote Parquet and video shards lazily; frame video
is decoded only when `load_video_frames` is set.

LIBERO catalog construction calls the Hugging Face manifest API but does not download task files.
`instructions()` adds a lazy Daft expression based on `bddl_path.download()`. Filter and
limit first so execution reads only the selected BDDL files.

Dataset readers accept an `io_config` for authenticated or customized object-store access.
Pass alternate `repo_id` and `revision` values explicitly when using a fork.

Daft also exposes `daft.datasets.droid.raw()` for the published DROID dataset. Use that
reader directly rather than wrapping it here.

Dataset licenses and simulator dependencies are not installed or redistributed by these
catalogs.

## Published rollout trace

The historical LIBERO-Spatial trace is stored in the
[`Eventual-Inc/physical-ai-evals-libero-spatial-pilot`](https://huggingface.co/datasets/Eventual-Inc/physical-ai-evals-libero-spatial-pilot)
Hugging Face dataset. Revision
`ddb8a88fcc579ebf077a9ca2d1e026a7e1cf4429` contains 23,283 `rollout-v1` transition rows.
Its dataset card documents the missing execution and model provenance.
