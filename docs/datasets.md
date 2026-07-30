# Dataset readers

## Generic LeRobot v3

The reader functions return Daft DataFrames directly:

```python
from physical_ai_evals.datasets import (
    ALOHA,
    lerobot,
    lerobot_episodes,
    lerobot_tasks,
)

episodes = lerobot_episodes(ALOHA)
tasks = lerobot_tasks(ALOHA)
frames = lerobot(ALOHA, load_video_frames=False)
```

`lerobot()` decodes no video unless `load_video_frames` names a camera (or is
`True`). `lerobot_episodes()` reads episode metadata only.

`LeRobotSource` binds a Hugging Face dataset to an exact revision:

```python
from physical_ai_evals.datasets import LeRobotSource, lerobot_episodes

source = LeRobotSource("organization/dataset", "<40-character-commit>")
lerobot_episodes(source).show()
```

Daft 0.7.21 recursive Hugging Face globs lose the `@revision` component in
listed paths. The wrapper therefore verifies that the repository head still
equals the recorded commit before constructing the lazy read. It fails closed
if the repository moved.

## Recorded sources

| Constant | Dataset | Revision |
|---|---|---|
| `ALOHA` | `lerobot/aloha_mobile_shrimp` | `6e828202059d2cc204b61ff968c232d202127a34` |
| `ABC_130K` | `lerobot/abc_130k_v3_train` | `68651e4929d9fb00f798937b2d62617cab5c771d` |
| `ABC_130K_SMOKE` | `lerobot/abc_130k_v3_smoke` | `b342a0ff262195d49bae3eece6e3f40c6e1dbe15` |

## LIBERO task plans

Each LIBERO constructor exposes its lazy executable episode plan directly:

```python
import daft
from physical_ai_evals import libero_para, libero_pro

para = libero_para(task_ids=[3], episodes=1).specs
para.groupby("perturbation").agg(
    daft.col("bddl_path").count().alias("variants")
).show()

pro = libero_pro(
    "libero_spatial",
    perturbations=["lan"],
    episodes=1,
).specs
pro.select("task_key", "bddl_path", "init_path").show()
```

The constructors use Daft glob, regex, join, download, and expression nodes to
turn the benchmark files into rows that the LIBERO runtime can execute.
