# Troubleshooting

## Images and installation

| Symptom | Check |
|---|---|
| OpenVLA remote code breaks in model load | Use Transformers `4.40.1`; newer releases are not compatible with the pinned checkpoint code. |
| Torch reports `_ARRAY_API not found` | OpenVLA's Torch 2.2 build requires NumPy 1.x. |
| VLA-JEPA import fails | Use the pinned LeRobot commit in `physical_ai_evals/modal.py`; it requires Python 3.12 and a Transformers 5.x stack. |
| OpenVLA and VLA-JEPA dependencies conflict | They cannot share an environment. Use the two Modal images or two isolated GPU environments. |
| daft-cuTile Modal import rejects the checkout | Set `DAFT_CUTILE_SOURCE_ROOT` to a clean checkout at the exact revision named in `physical_ai_evals/cutile_vla_jepa.py`. |
| daft-cuTile cohort exceeds four environments | Use `CUTILE_ENV_BATCH_SIZE=1` through `4`; the production engine has static B1 and B4 graphs. |
| EGL context creation fails | On Linux set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`; use `cgl` on macOS. |
| `evdev` cannot find `linux/input.h` | Install `linux-libc-dev`. |
| Extension build cannot find a compiler | Install `build-essential`, `clang`, and `cmake`. |
| LIBERO-Pro BDDL cannot resolve an object | Confirm the pinned Pro fork, rather than standard LIBERO, is the imported `libero` package. |

## Rollouts

| Symptom | Check |
|---|---|
| OpenVLA gripper never opens | Verify the checkpoint-specific RLDS-to-LIBERO gripper conversion and suite `unnorm_key`. |
| Success changes after preprocessing edits | Compare camera rotation, OpenVLA center crop, resize, and input range. |
| `torch.from_numpy` rejects negative strides | Preserve the adapter's contiguous copy after camera rotation. |
| Later episodes hit the horizon unexpectedly | `env.reset()` must precede every `set_init_state()`. |
| A resume skips damaged data | It should not: remove no files manually; the completion validator rejects corrupt/gapped partitions and overwrites them. |
| A custom policy works locally but not on Modal | Include its defining module in the image and call `evaluate()` from a custom Modal function. |

Run a real CPU simulator smoke before spending GPU time:

```bash
make smoke-openvla BENCHMARK=libero SUITE=libero_spatial
make smoke-vla-jepa BENCHMARK=libero_para SUITE=libero_goal
make smoke-openvla BENCHMARK=libero_pro SUITE=libero_spatial PERTURBATIONS=lan
```
