# physical-ai-evals

[![CI](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Eventual-Inc/physical-ai-evals/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Single-file scripts that run vision-language-action (VLA) policies on LIBERO
simulator tasks. Each script pins its model revisions, simulator, and Python
dependencies in a [PEP 723](https://peps.python.org/pep-0723/) header, so
`uv run` builds the environment. Each one runs a pinned episode, prints the
result, and writes an MP4 of the agent-view camera.

## Scripts

| Script | Policy | Task | Hardware |
|---|---|---|---|
| [`openvla_libero.py`](examples/openvla_libero.py) | OpenVLA 7B, LIBERO-Spatial fine-tune | LIBERO-Spatial task 0, initial state 0 | CUDA or Apple Silicon (MPS) |
| [`openvla_libero_modal.py`](examples/openvla_libero_modal.py) | same script on Modal | same | Modal A10G |
| [`vla_jepa_libero_pro.py`](examples/vla_jepa_libero_pro.py) | VLA-JEPA through LeRobot | LIBERO-Pro `libero_spatial_lan` task `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, initial state 0 | CUDA or MPS |
| [`vla_jepa_libero_pro_modal.py`](examples/vla_jepa_libero_pro_modal.py) | same script on Modal | same | Modal A10G |
| [`vla_jepa_libero_pro_daft.py`](examples/vla_jepa_libero_pro_daft.py) | VLA-JEPA, with episodes as Daft rows and batched inference | same by default; `run_vla_jepa_rollouts()` takes a list of episodes | CUDA or MPS |

The Daft script steps each episode independently and only sends episodes whose
action queue is empty to the next inference batch, so finished episodes do not
take up inference.

## Run

Requires [uv](https://docs.astral.sh/uv/) and either a Linux CUDA GPU or an
Apple Silicon Mac. uv installs Python 3.12 if needed.

```bash
uv run examples/vla_jepa_libero_pro.py
```

The first run downloads about 16 GB of VLA-JEPA, Qwen3-VL, and V-JEPA2
weights. The script prints a dict with `success` and `control_steps`, and
writes `vla_jepa_libero_pro.mp4`. Set `VLA_JEPA_LIBERO_PRO_VIDEO` (or
`OPENVLA_LIBERO_VIDEO` for OpenVLA) to change the video path.

On Modal:

```bash
modal token new
modal secret create HF_TOKEN HF_TOKEN=...
uv run examples/vla_jepa_libero_pro_modal.py
```

The Modal launchers write the video to the `daft-model-outputs` volume and
cache weights in `daft-model-cache`.

OpenVLA and VLA-JEPA cannot share an environment. OpenVLA needs Torch 2.2,
NumPy 1.26, and Transformers 4.40.1. VLA-JEPA needs a pinned LeRobot commit
with Torch 2.11, NumPy 2.2, and Transformers 5.5. Each script carries its own
pins, so this only matters if you install them by hand.

## What has been checked

| Script | Where | Result |
|---|---|---|
| `vla_jepa_libero_pro.py` | Apple M4 Max (MPS), 2026-10-04 | success in 94 control steps |
| `vla_jepa_libero_pro_modal.py` | Modal CUDA, before PR #16 | success in 94 control steps |
| `vla_jepa_libero_pro_daft.py` | Apple M4 Max (MPS), 2026-10-04 | success in 93 control steps, one episode |
| `openvla_libero.py` | MPS and Modal A10G, before PR #15 | ran end to end; outcome not recorded |

Each row is a single episode. They show that the scripts run end to end; they
are not success-rate measurements. The Daft script's multi-episode batching is
covered only by a unit test with fake environments
([`tests/test_vla_jepa_libero_pro_daft.py`](tests/test_vla_jepa_libero_pro_daft.py)),
not by a real multi-episode run.

## The `physical_ai_evals` package

The `physical_ai_evals/` package (`pae.evaluate`, resumable Parquet traces,
the Modal apps behind `make rollout-*`, and the LeRobot dataset readers)
predates these scripts and has not been brought in line with them. CI runs its
unit tests, but it has not produced published benchmark numbers. Its
documentation is in [`docs/`](docs/index.md), and
[`examples/notebooks/`](examples/notebooks/README.md) uses it.

## Citing

Citation metadata lives in [`CITATION.cff`](CITATION.cff). Please also cite
the upstream benchmark, policy, and checkpoint papers relevant to your
evaluation.

## License

Repository code is Apache-2.0. Upstream datasets, models, simulators, and
software retain their own terms; see [Third-party notices](THIRD_PARTY_NOTICES.md).
