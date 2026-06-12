# NOTES — reproducibility gotchas

The running log of every silent-failure trap, version pin, and env conflict we hit. This
is not housekeeping: **the blog's reproducibility story comes straight from here.** Most
of these fail *silently* (success rate quietly drops to ~0; no exception), which is
exactly the pain the harness exists to surface — so every entry is a future paragraph.

Convention: one bullet per gotcha, lead with the symptom, then the fix.

## Environments — you need FOUR, and several are mutually exclusive

- **Core / notebook env (Python ≥3.10):** `pyarrow`, `numpy<2`, `daft==0.7.15`, sklearn,
  imageio. This is what the schema/writer/ingest-base and the notebook run in.
- **LIBERO sim env (Python 3.8.13):** `robosuite==1.4.0`, `bddl==1.0.1`, `numpy==1.22.4`,
  `gym==0.25.2`, `easydict==1.9`, `robomimic==0.2.0`, `torch==1.11.0+cu113`. **Does not
  co-install with the core env** — `numpy==1.22.4` is required by robosuite 1.4 but the
  core stack wants `numpy>=1.24`. Keep sim in its own conda env; the harness talks to it
  via the persisted parquet schema, not in-process.
- **OpenVLA env:** `torch==2.2.0`, `transformers==4.40.1`, `tokenizers==0.19.1`,
  `timm==0.9.10`. **Newer transformers silently breaks** the remote `predict_action`
  code path (load error or wrong output). Own venv.
- **VLA-JEPA env:** modern LeRobot + Qwen3-VL-2B + V-JEPA2 stack. Conflicts with OpenVLA's
  old transformers. The research repo uses conda env `VLA_JEPA`, Python 3.10. Own venv.
- Corollary: **do not assume both policies import in one process.** The harness isolates
  them (separate runs, or subprocess); the parquet schema is the only thing they share.

## LIBERO / robosuite / MuJoCo

- **Set `MUJOCO_GL` BEFORE importing robosuite/mujoco.** `egl` on Linux (GPU), `cgl` on
  macOS (no GPU), `osmesa` = CPU/slow, `glfw` fails headless. Import-first-set-after is too
  late → "EGL error" / "Failed to create GL context" (LIBERO issue #115). The runner sets
  `os.environ['MUJOCO_GL']` at the top of `_set_mujoco_gl()` before any sim import.
- **robosuite 1.4 uses the modern DeepMind `mujoco` binding, not `mujoco-py`.** Do not
  install mujoco-py. Do not bump robosuite to 1.5 — it changes obs key conventions and the
  controller config format and breaks BDDL loading.
- **`numpy>=1.24` breaks robosuite 1.4** (removed `np.float`/`np.bool`). Pin `1.22.4`.
- **Agentview images render 180° rotated** vs what released VLA checkpoints expect. Feed
  the raw frame and success silently tanks to ~0 — no error. The runner de-rotates with
  `img[::-1, ::-1]` before handing the frame to the policy. Eyeball a dumped frame as the
  first verification step.
- **`control_mode` (relative vs absolute) must match the checkpoint's training
  parameterization.** Mismatch = near-zero success, no exception.
- **`env.step` is the OLD gym 4-tuple** `(obs, reward, done, info)` (gym==0.25.2), NOT
  gymnasium's 5-tuple. A harness expecting `(obs, reward, terminated, truncated, info)`
  misparses.
- **`env.close()` between episodes/tasks** — leaked MuJoCo GL contexts crash or leak memory
  across a long sweep. `hard_reset=True` is the default.
- **Determinism = the (suite, task_id, init_state_id, seed) quadruple.** Seed alone is not
  enough — `set_init_state(init_state)` selects the object layout. This quadruple is the
  `episode_id` and the qualified-rollout unit.
- **Settle before policy control:** ~10 steps of zero action after `set_init_state` to let
  objects fall into place; cap episodes at the suite step budget (~220 short / ~520 long),
  not LIBERO's default horizon of 1000.

## Policies

- **OpenVLA `unnorm_key` is a silent trap.** Wrong key → flailing-but-plausible actions,
  not a crash. Per LIBERO suite it is `<suite>_no_noops` (e.g. `libero_goal_no_noops`);
  BridgeV2 is `bridge_orig`. Confirm the key list in the checkpoint's `norm_stats`.
- **Gripper sign/convention can be inverted** between a policy's output and the env's
  expected command (VLA-JEPA binarizes at `gripper_threshold=0.5`). Mismatch → never grasps
  or never releases. Verify on a known-good demo.
- **`flash_attention_2` is unavailable on Apple Silicon/CPU** (the likely dev machine →
  darwin/MPS). Use `attn_implementation='sdpa'`/`'eager'` + float32; expect single-episode
  smoke throughput, not full-suite runs.
- **VLA-JEPA chunking:** `chunk_size=7`, `n_action_steps=7`; `select_action` buffers a
  chunk and re-plans only on boundaries. **`reset()` between episodes** clears the buffer —
  skip it and a new episode replays the previous chunk.
- **VLA-JEPA load path is honest-uncertain:** the exact LeRobot policy *class symbol* was
  not verified verbatim. Load via `from_pretrained(policy.path)` / the policy factory, do
  NOT hard-import a guessed class name. And do not confuse **VLA-JEPA** (arXiv 2602.10098,
  ginwind) with the unrelated **JEPA-VLA** (arXiv 2602.11832).

## Ingest

- **LeRobot v2.1 vs v3.0 split.** Current `lerobot` (0.5.1) writes v3.0
  (many-episodes-per-file); VLA-JEPA consumes v2.1 (one-file-per-episode). Branch on
  `dataset.meta.info['codebase_version']`. Also `lerobot 0.5.1` needs Python ≥3.12 vs
  VLA-JEPA's 3.10 — pin the lerobot the VLA-JEPA path actually uses.
- **LeRobot images come back float CHW (normalized) by default.** Pass `return_uint8=True`
  (or convert) — the common schema fixes uint8 HWC.
- **HDF5 demo keys sort wrong:** `demo_10` sorts before `demo_2` lexicographically. Sort by
  the integer suffix or the episode order is scrambled.
- **Reward is sparse, not dense:** robomimic/HDF5 reward is 1 only on success; `dones` is 1
  at the terminal step. `success = (rewards.max()==1) or dones[-1]`.
- **Instruction lives in different places:** LeRobot per-frame `task`; DROID per-step
  `language_instruction`; HDF5 only in `data.attrs['env_args']` (constant per demo). Each
  adapter hoists it to the `Episode`.
- **Camera key names are not shared:** `observation.images.<cam>` vs `exterior_image_1_left`
  vs `agentview_image`. Never hardcode — map to canonical roles (`primary`/`wrist`) via
  `DEFAULT_CAMERA_ROLE_MAPS`.
- **DROID is heavy:** TF + tensorflow_datasets + GCS, f64 tensors. Use the `droid_100`
  subset (~2GB, identical schema) for smoke tests; `.numpy()` + cast f64→f32.

## Daft-native ingest — DROID & LeRobot are landing IN Daft (reshapes our adapter strategy)

As of 2026-06-11 there are two open **draft** PRs adding first-party dataset readers to Daft
itself — i.e. the "ingest DROID/LeRobot" half of our wedge, done by the library. This turns
our ingest layer from "hand-roll three readers" into "delegate two to Daft, own the third
(HDF5) + the entire rollout side." On-message for the GTM story: *Daft already ingests your
robot data.*

- **LeRobot — Daft PR #7090 `daft.datasets.lerobot`** (draft, @srilman). `read()` returns a
  **one-row-per-FRAME** DataFrame with episode metadata broadcast across frames — the SAME
  shape as our `ROLLOUT_SCHEMA`, just native LeRobot column names (`episode_index`,
  `frame_index`, `timestamp`, `observation.state`, `action`, `task`, `videos/{key}/...`).
  Also `read_episodes()` (one row/episode), `load_episode_frames()`, `read_tasks()`. Frames
  decode lazily from MP4 by absolute timestamp (needs PyAV + Pillow). Accepts HF repo id /
  `s3://` / `hf://`. **GAP: v3.0 ONLY** (raises otherwise) — VLA-JEPA consumes v2.1, so the
  v2.1 path still needs the lerobot lib. Implication: `ingest/lerobot.py` should DELEGATE to
  this, not re-implement — it becomes a column projection onto `ROLLOUT_SCHEMA`.
- **DROID — Daft PR #7089 `daft.datasets.droid`** (draft, @srilman). `raw()` reads the RAW
  DROID release (`gs://gresearch/robotics/droid_raw`), **one row per EPISODE**: unnested
  metadata (`uuid`, `success`, `current_task`, camera serials, `trajectory_length`, ...) +
  lazy file refs (`trajectory` → the per-episode `trajectory.h5`, `wrist_video` /
  `ext1_video` / `ext2_video` MP4s). **TWO GAPS: (1) it does NOT parse `trajectory.h5`** —
  actions/proprio/state are an explicit upstream `# TODO`, so per-step action signal is NOT
  available via Daft yet; **(2) it can't read the RLDS/TFDS curated version.** So frame-level
  DROID *with actions* still needs either our RLDS/tfds path OR us parsing the raw
  `trajectory.h5` (which is HDF5 — the format we own; see BACKLOG for the contribute-upstream
  angle).
- **HDF5 — no Daft PR.** Uncontested; ours to build. And the raw-DROID `trajectory.h5` is
  itself HDF5, so the h5-reading machinery does double duty.
- Both PRs are **DRAFT** — don't hard-pin against them (our pinned `daft==0.7.15` does not yet
  ship `daft.datasets.{lerobot,droid}`). Track #7089 / #7090; intended surface is
  `daft.datasets.{lerobot,droid}`.

**Our parsers (implemented 2026-06-11, h5py-based, TF-free):** robomimic/LIBERO HDF5 and
raw-DROID `trajectory.h5`, both projecting onto the canonical `ROLLOUT_SCHEMA` via
`Episode.to_step_rows` (14 synthetic-fixture tests pass; verified end-to-end through
`harness ingest`). CAVEATS for the implementation pass: the DROID `trajectory.h5` parser is
written against the EXPECTED raw layout (`action_dict` vs top-level `action` is probed; obs
leaves assumed plural, e.g. `joint_positions`) but was NOT confirmed against a real
`droid_raw` file — h5py-dump one real episode to verify group/leaf names before a full run.
DROID `control_mode='absolute'` (pose targets, not deltas) while LIBERO is `'relative'`, so
the notebook MUST read `control_mode` before any cross-source action comparison. HDF5 gripper
sign (`gripper_state = qpos[:,0]-qpos[:,1]`) is suite-dependent — verify on a known-good demo
so slip/grasp detection isn't inverted.

**Vendored Daft readers (temporary):** LeRobot & DROID ingest now call Daft's own readers
(PRs #7090/#7089), vendored verbatim in `harness/_vendor/daft_datasets` because they're
unmerged — our pinned `daft==0.7.15` lacks `daft.datasets.{lerobot,droid}` but has every
internal API they use, so they run as-is. `harness._vendor.daft_datasets.install()`
monkey-patches them onto `daft.datasets` and no-ops once a real Daft ships them. DELETE the
vendor package + the two `install()` call sites when the PRs land. Gotcha: Daft's glob returns
`file://` URIs for local paths — h5py needs a plain path, so DROID strips the scheme
(`_local_path`). LeRobot `read()` returns frame-level native columns; we normalize through
`Episode` (guaranteed schema parity) instead of projecting in-Daft — the lazy/at-scale
`write_parquet`-straight-from-Daft path is parked in BACKLOG.

**Policies (implemented, GPU-deferred):** OpenVLA + VLA-JEPA adapters carry the real load +
predict logic behind a lazy heavy-import (torch/transformers/lerobot/PIL never import at module
load). They're unit-tested with INJECTED fake backends (`_vla`/`_processor`, `_policy`); real-
weight inference needs the GPU env (Modal), not a CPU box. VLA-JEPA chunking lives INSIDE the
LeRobot policy (`select_action` buffers a chunk; `reset()` clears it) — the adapter just
forwards, so `reset()` between episodes is mandatory or a new episode replays the old chunk.

## Rollout / Modal

- **#1 ROLLOUT RISK — LIBERO + a VLA policy must coexist in ONE container image.** The closed
  loop calls `policy.act(obs)` and `env.step(action)` in the same process, so the rollout image
  needs robosuite 1.4 / numpy 1.22.4 (LIBERO, historically Python 3.8) AND the policy stack
  (OpenVLA: torch 2.2 / transformers 4.40; VLA-JEPA: modern lerobot) together. Modern LIBERO
  forks run on 3.10/3.11 (OpenPI does), so `modal_app.py` builds on 3.11 — but the exact
  numpy/torch pin compatibility is UNVERIFIED until the first deploy. If they can't coexist,
  fall back to the **OpenPI policy-server / env-client split** (a websocket between two images:
  policy in its env, LIBERO in another). The two policy stacks also conflict with each other →
  SEPARATE images (`openvla_image`, `vla_jepa_image`).
- **Rollout = a `@daft.cls` UDF.** One row per episode spec `(suite, task_id, init_state_id,
  seed)`; the UDF runs one episode via `run_episode`, streams per-step rows to a `RolloutWriter`
  (one parquet part + frames + mp4 per episode on the OUTPUT Volume), returns an episode summary
  struct. The queryable ROLLOUT_SCHEMA frame is the parquet-part glob, not the summary. Mirrors
  the daft-examples `models/<name>` layout: `model.py`(=`rollout_udf.py`, never imports modal)
  + `modal_app.py` shell. Verified with a fake env/policy in `tests/test_rollout.py`; the GPU
  UDF + Modal deploy are not run on CPU CI.
- **Modal conventions (from the user's daft-examples setup):** shared `daft-model-cache` /
  `daft-model-outputs` Volumes at `/models` / `/outputs`, `hf-token` Secret, HF Xet cache env,
  `add_local_python_source("harness")`, `MUJOCO_GL=egl` in the image env. GPU memory snapshots
  are opt-in and only pay off when the model loads inside the snapshot window — our UDF loads
  lazily on the Daft worker (after the snapshot), so they're left off for now (parked).
- **Determinism carries through:** `run_episode` stamps the `(suite, task_id, init_state_id,
  seed)` quadruple as `episode_id`; `env.close()` between tasks (the UDF caches one live env per
  worker) avoids leaking MuJoCo GL contexts across a long sweep.

## Daft (notebook side)

- **Lazy eval:** nothing runs until `.collect()` / `.show()` / `.to_pydict()` /
  `write_parquet`. A cell that just builds the DataFrame does no work and raises no error —
  failures surface only at materialization. Tests must force a trigger.
- **`write_parquet` returns a DataFrame of written paths, not `None`.** Don't treat the
  result as the original data.
- **No built-in clustering.** `df.kmeans()` does not exist. Embed in Daft →
  `df.select('embedding').to_pandas()` → `np.stack` → sklearn KMeans/HDBSCAN → attach
  labels back. `cosine_distance` exists but is pairwise, not clustering.
- **Prefer `@daft.cls` over the legacy `@daft.udf`.** Official example pages still use the
  deprecated `@daft.udf(concurrency=, num_gpus=, batch_size=)` form; it works on 0.7.x but
  warns and may break on 0.8. Pick one style.
- **Pin `daft==0.7.15`.** `0.7.11` was YANKED on PyPI.
- **Use `on_error='null'`** on `download`/`decode_image` so a few bad frames don't kill a
  job over thousands of rollouts.
- **`cosine_distance` needs `embedding(float32, DIM)` or `fixed_size_list(float32, DIM)`.**
  We store the portable `list<float32>` on disk → `.cast()` in the notebook before vector
  ops.
