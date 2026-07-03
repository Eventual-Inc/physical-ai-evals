# Friction log — everything we hit, in order

The chronological record: each entry is one thing that actually broke (or silently lied),
when, what it looked like, and what fixed it. The condensed symptom→fix reference is
[`FRICTION_POINTS.md`](FRICTION_POINTS.md); deep per-topic detail lives in
[`../NOTES.md`](../NOTES.md). Commits are receipts.

Convention: **[where it bit]** symptom → root cause → fix.

## 2026-06-11 — scaffold day (local)

1. **[local env]** Fresh workspace defaulted to Python 3.14 → no pyarrow wheels exist for it;
   nothing imported. → Pin the dev env to 3.13 (`uv venv --python 3.13`). *(f3a2386)*
2. **[tests]** Schema round-trip "mismatch" that wasn't: parquet renames list fields
   `item`→`element` on write; float32 round-trips are approximate. → Compare with
   `Schema.equals(check_metadata=False)`; approximate float asserts. *(f3a2386)*
3. **[ingest]** Daft's glob returns `file://` URIs for local paths; h5py can't open URIs. →
   Strip the scheme before opening (`_local_path`). *(f3a2386)*

## 2026-06-12 — first Modal images (build + first GPU run)

4. **[image build]** One broken image blocked everything: Modal builds *every* image an app
   references, so the unverified VLA-JEPA image failed the OpenVLA smoke too. → Split apps /
   don't reference unbuilt images. *(356b599)*
5. **[image build]** `evdev` C extension: `linux/input.h` missing (pulled via
   robosuite→pynput). → `apt linux-libc-dev`. *(356b599)*
6. **[image build]** `error: command 'clang' failed: No such file or directory` — CUDA
   *runtime* base has no compiler, and Modal's `add_python` interpreter is clang-built. →
   `apt build-essential clang`. *(356b599)*
7. **[runtime]** Editable-installed LIBERO (`pip install -e`) reported success but
   `import libero` failed inside the Modal function. → `PYTHONPATH=/opt/LIBERO`. (Obsoleted
   later by the `hf-libero` wheel.) *(356b599)*
8. **[runtime]** `EOFError: EOF when reading a line` on `import libero` — LIBERO prompts
   `input()` for a dataset path on first import; containers have no stdin. → Pre-write its
   config at image build. *(356b599)*
9. **[runtime]** `Failed to initialize NumPy: _ARRAY_API not found` — daft pulled pyarrow 24
   which pulled numpy 2.2.6; torch 2.2 is compiled against numpy 1.x. → Pin
   `numpy==1.26.4`. *(356b599)*
10. **[build hygiene]** opencv 4.13 declares `numpy>=2`, warning against the pinned 1.26.4. →
    Pin `opencv-python==4.9.0.80` (last numpy-1-clean line). *(356b599)*
11. **[first GPU run]** `ModuleNotFoundError: matplotlib` at env construction —
    `pip --no-deps` on LIBERO dropped runtime deps that only `libero.libero.envs` imports;
    the import-only CPU smoke never touched them. → Add `matplotlib`, `einops` to the sim
    pins. Lesson: import smokes under-test; only real env construction catches these.
    *(356b599)*
12. **[first GPU run]** `AssertionError: The unnorm_key you chose is not in the set of
    available dataset statistics ... dict_keys(['libero_spatial'])` — the LIBERO fine-tunes
    key their norm stats by **suite name**, not the `<suite>_no_noops` dataset name the docs
    suggest. → Fix the key table + fall back to the sole available key. *(98c4e67)*

## 2026-07-02 — the in-process flip (build + first real rollouts)

13. **[image build]** `egl_probe`: "CMake must be installed" — `hf-libero` depends on a
    CMake-built C extension; CUDA base has no cmake. → `apt cmake`. *(269ec16)*
14. **[runtime]** The `input()` prompt again — it survived into the packaged `hf-libero`
    wheel. → Bake the config at build: `printf 'n\n' | python -c 'import libero.libero'`.
    *(269ec16)*
15. **[first in-process rollout]** `ValueError: At least one stride in the given numpy array
    is negative` — the runner's 180° de-rotation (`img[::-1, ::-1]`) is a reversed *view*;
    PIL-based policies copy implicitly, `torch.from_numpy` refuses. →
    `np.ascontiguousarray` in the adapter. *(9ea1c92)*

## 2026-07-02 — the sweeps (scale reveals everything)

16. **[sweep, ~episode 4]** `ValueError: executing action in terminated episode`, then every
    subsequent episode dead — `set_init_state` restores sim state but does NOT clear
    robosuite's internal step counter, which accumulates *across* episodes toward the horizon
    (1000) when the env is cached. Invisible in short runs (our 2-episode verification =
    ~330 cumulative steps). → `env.reset()` before `set_init_state`, every episode. openpi's
    loop does this; now we know why. *(7410577)*
17. **[sweep results]** **OpenVLA 0/7, every episode at the 250-step cap** — no error, just a
    plausible-looking robot that never succeeds. One parquet query
    (`min/max gripper_action = [0.000, 0.996]`) → the RLDS gripper convention fed raw into
    LIBERO: the hand could never open, polarity inverted on top. → normalize → binarize →
    invert (OpenVLA's own eval utils), plus the center-crop its eval applies for
    aug-trained checkpoints. **After: 10/10 on task 0; final sweep 84/100 vs the published
    84.7%.** *(df6c8ee)*
18. **[ops]** Both detached sweeps died mid-run with "Received a cancellation signal" —
    `modal run --detach` survives network drops but a *client teardown* propagates a cancel.
    → `modal deploy` + `Function.spawn()` (nothing local to kill) + resumable sweeps that
    skip episodes whose part file already exists (filenames are deterministic episode ids).
    *(b457249)*
19. **[analysis]** A 500-step episode in a 250-cap suite — `episode_id` names the episode
    *spec*, so both policies produce `libero_spatial/5/1/7`, and a `groupby("episode_id")`
    chimera'd two policies' trajectories into one phantom (113 phantom grasp cycles). Caught
    only because 500 > 250. → Group by `(policy_type, episode_id)`; schema docs updated;
    policy-qualified id parked as a breaking change. *(b137672)*
20. **[observation]** After the `env.reset()` fix, episode `libero_spatial/0/1/7` (VLA-JEPA)
    flipped from fail-at-cap to success — cached-env contamination degrades outcomes well
    before it hard-crashes. (n=1; GPU nondeterminism not fully excluded.) *(noted in NOTES)*

## Standing lessons

- **Loud failures are the cheap ones.** Everything in the build section cost minutes.
  Everything in the "silent" sections cost a sweep — or would have cost a wrong conclusion.
- **Import-only smoke tests under-test.** Env construction, GPU inference, and multi-episode
  sweeps each surfaced a class of bug the previous layer could not see.
- **Record enough to interrogate.** Every silent bug in this log was diagnosed from the
  per-step parquet in one or two queries. That is the whole thesis of the repo.
