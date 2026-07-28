# Dataset catalogs

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

Catalog construction calls the Hugging Face manifest API but does not download task files.
`instructions()` adds a lazy Daft expression based on `bddl_path.download()`. Filter and
limit first so execution reads only the selected BDDL files.

Both functions accept an `io_config` for authenticated or customized object-store access.
Pass alternate `repo_id` and `revision` values to `raw()` explicitly when using a fork.

Daft also exposes `daft.datasets.droid.raw()` for the published DROID dataset. Use that
reader directly rather than wrapping it here.

Dataset licenses and simulator dependencies are not installed or redistributed by these
catalogs.
