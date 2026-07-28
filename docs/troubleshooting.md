# Troubleshooting

These fixes apply to the pinned OpenVLA, VLA-JEPA, and LIBERO stacks used by this repository.

## Environment

| Symptom | Check |
|---|---|
| EGL context creation fails | Set `MUJOCO_GL=egl` before importing MuJoCo or robosuite. Use `cgl` on macOS. |
| `evdev` cannot find `linux/input.h` | Install `linux-libc-dev`. |
| Extension build cannot find a compiler | Install `build-essential` and `clang` in the image. |
| First `import libero` raises `EOFError` | Create LIBERO's path configuration non-interactively during the image build. |
| LIBERO imports but environment creation fails | Test construction of an environment; a `--no-deps` install may omit runtime-only imports. |
| Torch reports `_ARRAY_API not found` | The Torch 2.2 stack requires NumPy 1.x; use `numpy==1.26.4`. |

## Evaluation

| Symptom | Check |
|---|---|
| Episodes always hit the step cap | Verify the checkpoint-specific gripper transform and action unnormalization key. |
| Success changes after preprocessing edits | Compare image rotation, resize, center crop, and input value range with the reference evaluator. |
| `torch.from_numpy` rejects negative strides | Call `np.ascontiguousarray` after rotating an image with slicing. |
| Later episodes fail or raise after many steps | Call `env.reset()` before every `set_init_state`; the robosuite step counter otherwise carries across episodes. |
| Episode length appears doubled | Group by both `policy_type` and `episode_id`. |
| MuJoCo or SciPy fails after rebuilding | Compare resolved transitive versions with the pinned image; the tested stack uses `mujoco==3.9.0` and `scipy==1.15.3`. |

Run the Modal image smoke tests before a sweep:

```bash
make smoke-openvla
make smoke-vla-jepa
```
