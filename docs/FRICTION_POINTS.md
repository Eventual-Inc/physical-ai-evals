# Friction points observed in the tested stacks

These are engineering observations from this repository's OpenVLA and VLA-JEPA/LIBERO
environments, not general benchmark findings. A “fix” below means the pinned test stack built
or ran after the change; it does not establish that the same intervention is correct for every
model, simulator, operating system, or dependency revision. Pin and report the resulting
environment rather than treating this page as a timeless compatibility guarantee.

## Build & environment (bites at image build / first import — loudly)

| # | Symptom | Fix |
|---|---------|-----|
| 1 | Resolver reports a LIBERO conflict | In the [audited LIBERO revision](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/setup.py), `setup.py` does not declare the full training requirements. Install and test the runtime subset explicitly. OpenVLA and VLA-JEPA still use separate images here because their policy stacks conflict. |
| 2 | EGL / "failed to create GL context" | Set `MUJOCO_GL=egl` (macOS: `cgl`) **before** any robosuite/mujoco import |
| 3 | `evdev` build fails: `linux/input.h` missing | `apt install linux-libc-dev` (robosuite → pynput → evdev) |
| 4 | `error: command 'clang' failed` while building an extension | The tested CUDA runtime image needed `build-essential` and `clang`; record the base-image digest because compiler availability varies. |
| 5 | `egl_probe`: "CMake must be installed" | `apt install cmake` (`hf-libero` dep) |
| 6 | `EOFError` on `import libero` in a non-interactive container | In the tested revision, first import may prompt for path configuration. Create the config during the image build (the VLA-JEPA image uses `printf 'n\n' \| python -c 'import libero.libero'`) and verify all resolved paths in a smoke test. |
| 7 | LIBERO environment construction fails on a missing `matplotlib` or `einops` import | A `--no-deps` install omits packages reached only during environment construction. Add an environment-construction smoke test, not only an `import libero` test. |
| 20 | `uv lock`/`uv run` "No solution found" on a project that pip-installs fine | uv resolves **every** extra for **every** python/platform — one exotic (git-dep) extra bricks the lock for all users. Scope with `[tool.uv] environments`, park the exotic stack behind a pointer extra, pin `.python-version` |

## Silent success-rate killers (nothing raises; the number is just wrong)

| # | Symptom | Fix |
|---|---------|-----|
| 8 | **0% SR, every episode runs to the step cap** | Check the gripper convention. OpenVLA's [reference evaluator](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L219-L225) normalizes, binarizes, and inverts its output before LIBERO. Match the evaluated checkpoint rather than applying this transformation universally. |
| 9 | Success rate changes after an evaluation-code update | Verify whether the checkpoint was trained with crop augmentation. OpenVLA's reference evaluator defaults to center crop and asserts it for checkpoint names containing `image_aug`; preserve the resolved choice in the manifest. |
| 10 | `unnorm_key` assertion (fine-tunes) / plausible flailing (base models) | For the LIBERO fine-tunes the key is the **suite name** (`libero_spatial`), not `<suite>_no_noops` |
| 11 | `Failed to initialize NumPy: _ARRAY_API not found` | numpy 2 snuck in via a dependency bump (daft→pyarrow); torch 2.2 is numpy-1-compiled → pin `numpy==1.26.4` |
| 12 | pip warns `opencv-python requires numpy>=2` | opencv 4.13 declares numpy≥2; pin `opencv-python==4.9.0.80` on a numpy-1 stack |
| 13 | Success rate is low although stored frames look plausible | The pinned OpenVLA [LIBERO utility](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/libero_utils.py#L50-L58) rotates agent-view images by 180 degrees to match training preprocessing. Confirm the required orientation for the specific checkpoint. |
| 14 | SR wrecked, tensors well-formed | Model-side processor runs `do_rescale=False` → images must be float **[0, 1]**; 0–255 floats fail silently |

## Sweep & analysis killers (only appear at scale)

| # | Symptom | Fix |
|---|---------|-----|
| 15 | `ValueError: executing action in terminated episode` mid-sweep | `env.reset()` **every** episode: `set_init_state` alone leaves robosuite's step counter running *across* episodes; ~cumulative step 1000 the env poisons itself. Invisible in short runs — and it degrades outcomes *before* it crashes |
| 16 | `torch.from_numpy`: "negative strides are not supported" | The 180° de-rotation is a reversed **view** — `np.ascontiguousarray` first |
| 17 | A 500-step episode in a 250-cap suite | `episode_id` names the episode *spec* (identical across policies by design) — group by `(policy_type, episode_id)` or you chimera trajectories |
| 18 | `mj_fullM()` TypeError / scipy-vs-numpy conflict after a rebuild | Unpinned transitives drift with the build date (mujoco>=3.10 breaks robosuite 1.4.x bindings; scipy>=1.18 wants numpy>=2) → pin the sweep-verified set: `mujoco==3.9.0`, `scipy==1.15.3` |
| 19 | Detached Modal sweep dies when the laptop does | `modal run -d` survives network drops, **not** client teardown → `modal deploy` + `Function.spawn()`, and make sweeps resumable (part filename = deterministic episode id) |

---

Several of these issues produced valid-shaped tensors and completed episodes in the tested
stack. Treat a changed success rate as an observation to audit, not immediate evidence about a
model or a particular layer of the harness.
