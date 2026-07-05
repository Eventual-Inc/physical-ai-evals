# Friction points — the condensed field guide

**19 landmines between you and a reproducible VLA eval.** Symptom → fix, grouped by when it
bites. This is the quick-reference layer; the chronological story is
[`FRICTION_LOG.md`](FRICTION_LOG.md) and the deep per-topic detail is
[`../NOTES.md`](../NOTES.md).

## Build & environment (bites at image build / first import — loudly)

| # | Symptom | Fix |
|---|---------|-----|
| 1 | "LIBERO conflicts with modern stacks" | Myth — its `setup.py` installs nothing; the scary `requirements.txt` is training-only. One env works |
| 2 | EGL / "failed to create GL context" | Set `MUJOCO_GL=egl` (macOS: `cgl`) **before** any robosuite/mujoco import |
| 3 | `evdev` build fails: `linux/input.h` missing | `apt install linux-libc-dev` (robosuite → pynput → evdev) |
| 4 | `error: command 'clang' failed` on any C ext | CUDA *runtime* bases ship no compiler; Modal's `add_python` is clang-built → `apt install build-essential clang` |
| 5 | `egl_probe`: "CMake must be installed" | `apt install cmake` (`hf-libero` dep) |
| 6 | `EOFError` on `import libero` in a container | LIBERO prompts `input()` on first import — bake its config at image build: `printf 'n\n' \| python -c 'import libero.libero'`. (Survived into the `hf-libero` wheel, too) |
| 7 | LIBERO env construction dies on `matplotlib`/`einops` | `pip --no-deps` drops runtime deps only `libero.libero.envs` needs — import-only smoke tests under-test |

## Silent success-rate killers (nothing raises; the number is just wrong)

| # | Symptom | Fix |
|---|---------|-----|
| 8 | **0% SR, every episode runs to the step cap** | RLDS gripper convention (0..1, ~1=open) fed raw into LIBERO (−1=open/+1=close): the hand can never open. Normalize [0,1]→[−1,1], binarize, **invert** (OpenVLA's own eval utils) |
| 9 | SR quietly below published | Apply center-crop at eval (0.9 area, resize back) when the checkpoint trained with crop augmentation |
| 10 | `unnorm_key` assertion (fine-tunes) / plausible flailing (base models) | For the LIBERO fine-tunes the key is the **suite name** (`libero_spatial`), not `<suite>_no_noops` |
| 11 | `Failed to initialize NumPy: _ARRAY_API not found` | numpy 2 snuck in via a dependency bump (daft→pyarrow); torch 2.2 is numpy-1-compiled → pin `numpy==1.26.4` |
| 12 | pip warns `opencv-python requires numpy>=2` | opencv 4.13 declares numpy≥2; pin `opencv-python==4.9.0.80` on a numpy-1 stack |
| 13 | SR ~0, frames "look valid" | LIBERO renders agentview **180° rotated** vs released checkpoints — de-rotate `img[::-1, ::-1]` before the policy |
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

*None of the silent ones raise. All of them read as "the model is bad."*
