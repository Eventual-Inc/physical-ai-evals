# physical-ai-evals

[![CI](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Run OpenVLA and VLA-JEPA on LIBERO simulator tasks.

- **Scripts** in [`examples/`](examples/): one file per policy, with pinned
  dependencies, that runs one episode and writes a video.
- **The `physical_ai_evals` package**: runs many episodes, resumes after a
  crash, and writes every step to Parquet.

Both follow OpenVLA's and LeRobot's own LIBERO evaluation code. On the same
episode, the package and the OpenVLA script produce identical actions.

## What has been run

LIBERO-Spatial task 0, on 2026-10-04.

| What | Hardware | Result |
|---|---|---|
| `examples/openvla_libero.py` | Modal A10G | success, 85 steps |
| `examples/vla_jepa_libero_pro.py` (LIBERO-Pro `libero_spatial_lan`) | Modal A10G | success, 95 steps |
| `examples/vla_jepa_libero_pro_daft.py`, same task, initial states 0–3 in one batch | Modal A10G | 4 of 4 succeeded |
| package, OpenVLA, initial states 0 and 1 | Modal A10G | 1 of 2 succeeded |
| package, VLA-JEPA, initial states 0 and 1 | Modal A10G | 2 of 2 succeeded |
| package vs. `openvla_libero.py`, initial state 0 | Modal A10G | identical actions on all 85 steps |

These runs show that the code works end to end. They are not benchmark
results: no full-suite numbers have been produced yet.

Not yet run end to end: the package on LIBERO-Para and LIBERO-Pro, and the
LeRobot dataset readers.

## Run a script

Requires [uv](https://docs.astral.sh/uv/) and a Linux CUDA GPU or an Apple
Silicon Mac.

```bash
uv run examples/openvla_libero.py
uv run examples/vla_jepa_libero_pro.py
```

On Modal:

```bash
modal token new
modal secret create HF_TOKEN HF_TOKEN=...
uv run examples/openvla_libero_modal.py
uv run examples/vla_jepa_libero_pro_modal.py
```

## Run the package

OpenVLA and VLA-JEPA need incompatible dependencies, so the package runs each
in its own Modal image:

```bash
make setup
make rollout-openvla SUITE=libero_spatial TASKS=0 EPISODES=2
make rollout-vla-jepa SUITE=libero_spatial TASKS=0 EPISODES=2
```

The same run from Python, inside a policy's environment:

```python
import physical_ai_evals as pae

evaluation = pae.evaluate(
    pae.openvla("libero_spatial"),
    pae.libero("libero_spatial", task_ids=[0], episodes=2),
    out="data/evaluations",
)
evaluation.episodes.select("init_state_id", "success", "length").show()
```

Seeds, settle steps, step budgets, and observation handling are listed in
[Evaluation protocol](docs/evaluation.md).

## License

Apache-2.0. Models, datasets, and simulators keep their own terms; see
[Third-party notices](THIRD_PARTY_NOTICES.md). Citation metadata is in
[`CITATION.cff`](CITATION.cff).
