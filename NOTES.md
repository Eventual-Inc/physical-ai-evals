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
- **VLA-JEPA env:** StarVLA + Qwen3-VL-2B + V-JEPA2 stack. Conflicts with OpenVLA's
  old transformers. The public repo uses conda env `VLA_JEPA`, Python 3.10, and serves the
  policy through `deployment/model_server/server_policy.py`. Own venv/image.
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
  objects fall into place; cap episodes at the suite step budget (`spatial=250`, `object=280`,
  `goal=300`, `long=520`), not LIBERO's default horizon of 1000.

## Policies

- **OpenVLA `unnorm_key` for the LIBERO fine-tunes is the SUITE name, not `<suite>_no_noops`.**
  VERIFIED on Modal: `openvla-7b-finetuned-libero-spatial`'s norm_stats has the single key
  `'libero_spatial'` — passing `'libero_spatial_no_noops'` (the training-dataset name) raises
  `AssertionError: ... please choose from dict_keys(['libero_spatial'])`. So it's a HARD crash
  here (the base OXE model's wrong-key case is the silent-flailing trap; BridgeV2 base is
  `bridge_orig`). Our policy now falls back to the sole norm_stats key when the guess is absent.
- **Gripper sign/convention can be inverted** between a policy's output and the env's
  expected command (VLA-JEPA binarizes at `gripper_threshold=0.5`). Mismatch → never grasps
  or never releases. Verify on a known-good demo.
- **`flash_attention_2` is unavailable on Apple Silicon/CPU** (the likely dev machine →
  darwin/MPS). Use `attn_implementation='sdpa'`/`'eager'` + float32; expect single-episode
  smoke throughput, not full-suite runs.
- **VLA-JEPA chunking:** the HF LIBERO config has `future_action_window_size=6`, so the
  server returns 7 normalized actions per request. The harness caches that chunk locally and
  re-plans only on chunk boundaries. **`reset()` between episodes** clears the local chunk —
  skip it and a new episode replays the previous chunk.
- **VLA-JEPA load path is StarVLA WebSocket, not LeRobot.** The official LIBERO eval starts
  `deployment/model_server/server_policy.py --ckpt_path .../LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`
  and the sim talks to it with `WebsocketClientPolicy`. The stats key is `franka`; the HF
  config must be patched to point `framework.qwenvl.base_vlm` and `framework.vj2_model.base_encoder`
  at local Qwen3-VL/V-JEPA2 snapshots, and `flash_attention_2` can be changed to `sdpa` for
  a tractable Modal build. Do not confuse **VLA-JEPA** (arXiv 2602.10098, ginwind) with the
  unrelated **JEPA-VLA** (arXiv 2602.11832).

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

## Daft-native ingest — DROID & LeRobot are in Daft (reshapes our adapter strategy)

As of 2026-07-01, Daft PRs #7090 and #7089 are **merged**. Our pinned `daft==0.7.15`
still lacks `daft.datasets.{lerobot,droid}`, so the harness keeps temporary vendored copies
that no-op once the Daft pin is bumped. This turns our ingest layer from "hand-roll three
readers" into "delegate two to Daft, own HDF5 + raw-DROID trajectory parsing + rollout."
On-message for the GTM story: *Daft already ingests your robot data.*

- **LeRobot — Daft PR #7090 `daft.datasets.lerobot`** (merged, @srilman). `read()` returns a
  **one-row-per-FRAME** DataFrame with episode metadata broadcast across frames — the SAME
  shape as our `ROLLOUT_SCHEMA`, just native LeRobot column names (`episode_index`,
  `frame_index`, `timestamp`, `observation.state`, `action`, `task`, `videos/{key}/...`).
  Also `read_episodes()` (one row/episode), `load_episode_frames()`, `read_tasks()`. Frames
  decode lazily from MP4 by absolute timestamp (needs PyAV + Pillow). Accepts HF repo id /
  `s3://` / `hf://`. **GAP: v3.0 ONLY** (raises otherwise) — VLA-JEPA consumes v2.1, so the
  v2.1 path still needs the lerobot lib. Implication: `ingest/lerobot.py` should DELEGATE to
  this, not re-implement — it becomes a column projection onto `ROLLOUT_SCHEMA`.
- **DROID — Daft PR #7089 `daft.datasets.droid`** (merged, @srilman). `raw()` reads the RAW
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
- Our pinned `daft==0.7.15` does not yet ship these modules. Track the first Daft release
  containing #7089/#7090, bump the pin, then delete `harness/_vendor/daft_datasets`.

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

**Daft-native readers (vendor shim DELETED 2026-07-02):** LeRobot & DROID ingest call Daft's
own `daft.datasets.{lerobot,droid}` directly. History: while the readers were unmerged we
vendored them (`harness/_vendor/daft_datasets`, see git history); Daft 0.7.16 released
`datasets.droid`/`VideoFile.frames()`/`read_mcap`, and **nightly** carries
`datasets.lerobot` + `Hdf5File` until the next release — dev installs:
`pip install daft --pre --extra-index-url https://nightly.daft.ai` (nightly also bumps
pyarrow to 24.x; the schema round-trips fine — suite verified). Daft PR #7160 (DROID + HDF5
API improvements) flows in via nightly as it merges. Gotcha that outlived the shim: Daft's
glob returns `file://` URIs for local paths — h5py needs a plain path, so DROID strips the
scheme (`_local_path`). LeRobot `read()` returns frame-level native columns; we normalize
through `Episode` (guaranteed schema parity) instead of projecting in-Daft — the lazy/at-scale
`write_parquet`-straight-from-Daft path is parked in BACKLOG.

**Policies (implemented, GPU-deferred):** OpenVLA + VLA-JEPA adapters carry the real load +
predict logic behind lazy heavy imports / clients (torch/transformers/PIL/VLA-JEPA checkout never
import at module load). They're unit-tested with INJECTED fake backends (`_vla`/`_processor`,
WebSocket `_client`); real-weight inference needs the GPU env (Modal), not a CPU box. VLA-JEPA
chunking is handled by the adapter: one server call returns a 7-action chunk, `act()` yields one
unnormalized LIBERO action at a time, and `reset()` clears the cached chunk between episodes.

## Rollout / Modal

- **VLA-JEPA is now IN-PROCESS via the lerobot port (2026-07-02) — the StarVLA WebSocket
  server route is deleted (git history keeps it).** lerobot merged a first-party `vla_jepa`
  policy (2026-06-04, main-only until the next release → pinned at git SHA `052d3294`, the way
  openpi pins lerobot) and the official `lerobot/VLA-JEPA-LIBERO` safetensors checkpoint is on
  the Hub. What this deletes: the ginwind repo clone, the server subprocess + port-wait, manual
  `dataset_statistics.json` unnorm, the gripper-convention flip (the lerobot postprocessor
  unnormalizes AND binarizes the gripper to LIBERO-ready {-1,+1}), and the client-side chunk
  cache (the policy's internal `n_action_steps=7` queue dequeues one action per
  `select_action`). Gotchas that remain load-bearing:
    - **`lerobot[libero]` → the `hf-libero` wheel ships LIBERO's code + bddl/assets/init_files**
      with the same `libero.libero` import paths — no git clone, no `LIBERO_CONFIG_PATH`
      pre-write, no PYTHONPATH tricks. It replaces the whole clone+config-patch machinery.
    - **Images must be (1,3,H,W) float32 in [0,1]** — lerobot's Qwen interface calls its
      processor with `do_rescale=False`, so 0-255 floats fail SILENTLY (the classic
      preprocessing-mismatch failure mode; see docs/EVAL_PATTERNS.md invariant).
    - Processor pipelines are NOT auto-wired by `from_pretrained` — load them with
      `make_pre_post_processors(..., preprocessor_overrides={"device_processor": {"device": ...}})`
      (the saved JSON pins device=cpu; this mirrors lerobot_eval).
    - **The 180° de-rotation view breaks `torch.from_numpy`.** Our runner de-rotates with
      `img[::-1, ::-1]` — a negative-stride numpy VIEW. PIL-based policies (OpenVLA) copy
      implicitly, but torch-based ones crash: "At least one stride in the given numpy array is
      negative". Fix: `np.ascontiguousarray` in the adapter's image conversion. Caught on the
      first real GPU rollout — exactly the kind of policy-boundary landmine the smoke can't see.
    - **`hf-libero` kept LIBERO's interactive first-import prompt.** The wheel still runs
      `input("...custom path for the dataset folder?...")` when no config exists → EOFError in
      a container (the same trap as the raw clone, one packaging layer later). Fix: at image
      build, `printf 'n\n' | python -c 'import libero.libero'` — LIBERO then writes its own
      default config pointing at the wheel's bddl/assets/init paths, baked into the image.
    - **`hf-libero` → `egl_probe` needs CMake.** hf-libero depends on `hf-egl-probe`/`egl_probe`
      (EGL-device probing), a C extension whose setup.py hard-requires CMake — the CUDA base
      image has none, so the whole `lerobot[libero]` install fails with "CMake must be
      installed". Fix: `apt_install("cmake")`. (Same genus as the evdev/kernel-headers and
      no-compiler gotchas on the OpenVLA image: CUDA runtime bases are build-tool-free.)
    - The VLA-JEPA image is py3.12 / transformers 5.4–5.6 / numpy 2.x — SEPARATE image from
      OpenVLA's py3.10/4.40.1 (two-app split stands). numpy-2 × daft/pyarrow inside that image
      is a smoke-test watch item; loading also pulls Qwen3-VL-2B + V-JEPA2 (3 HF repos; the
      world-model encoder loads even though inference never uses it — faithful default).
- **Single-image LIBERO + OpenVLA: VERIFIED on Modal (2026-06-12).** Image builds + LIBERO
  imports + benchmark/bddl resolution green (CPU smoke). Resolved/pinned set: **python 3.10,
  torch 2.2.0, transformers 4.40.1, robosuite 1.4.1, mujoco 3.9.0, numpy==1.26.4 (pinned),
  opencv-python==4.9.0.80 (pinned), gym 0.26.2, bddl 3.6.0, daft 0.7.15**. (Still to validate on
  GPU: EGL env rendering + OpenVLA inference — the smoke only exercised import + task metadata,
  and `gym 0.26`/`bddl 3.6` are newer than LIBERO's tested 0.25.2/1.0.1, so watch env
  construction.) The "conflict" was largely a myth. LIBERO's `setup.py` is `install_requires=[]` /
  `python_requires=">=3"`, so `pip install -e LIBERO` pulls NOTHING — its `requirements.txt`
  (numpy==1.22.4, transformers==4.21.1, robomimic) is LIBERO's TRAINING deps, never used for
  rollouts. The real combined env (OpenVLA `experiments/robot/libero/libero_requirements.txt`,
  conda **python=3.10**) is just: the policy's HF inference stack + **robosuite==1.4.1** + bddl +
  easydict + cloudpickle + gym + imageio[ffmpeg], with **numpy LEFT UNPINNED** so pip resolves
  one version for torch 2.2 + robosuite. No OpenPI split. Friction points found (each = a blog
  paragraph):
    - **LIBERO must be on DISK (git clone), not pip-from-git.** `bddl_files`/`init_states` are
      package data `get_libero_path` resolves from the repo dir, so we `git clone /opt/LIBERO` +
      `pip install --no-deps -e /opt/LIBERO`.
    - **robosuite 1.4.1, not 1.4.0** — what OpenVLA actually uses (1.5 changes obs keys/controller).
    - **robosuite → pynput → evdev needs kernel headers.** robosuite 1.4.1 depends on `pynput`
      (keyboard/SpaceMouse device input we never use), which on Linux pulls **`evdev`**, a C
      extension that fails to build with "`linux/input.h` missing". Fix: `apt_install("linux-libc-dev")`
      (provides the userspace kernel headers). Dead weight but harmless — we drive the env in code.
    - **CUDA base has NO compiler + add_python wants clang.** `nvidia/cuda:*-runtime` ships no
      gcc/clang, and Modal's `add_python` installs a python-build-standalone interpreter whose
      sysconfig CC is `clang` — so building any C ext (evdev) fails with "command 'clang' failed:
      No such file or directory". Fix: `apt_install("build-essential", "clang")`. (debian_slim
      images dodge this; the CUDA-registry path does not.)
    - **Editable LIBERO install doesn't import in the Modal function** — `pip install -e
      /opt/LIBERO` reports success but `import libero` raises ModuleNotFound at runtime. Fix:
      `.env({"PYTHONPATH": "/opt/LIBERO"})` (mirrors the sam3d reference's clone+sys.path pattern);
      keeps bddl_files on disk too.
    - **LIBERO's `__init__` interactively prompts on first import.** `libero/libero/__init__.py`
      runs `input("...custom path for the dataset folder?...")` if `~/.libero/config.yaml`
      (or `$LIBERO_CONFIG_PATH`) is missing → **EOFError** in a container. Fix: pre-write the
      config at build (set `LIBERO_CONFIG_PATH` + `echo` the path dict pointing at the clone's
      `bddl_files`/`init_files`/`assets`). `get_libero_path` only *warns* on missing `datasets`,
      so we don't need the (separately-downloaded) datasets for rollouts.
    - **numpy 2 sneaks in via daft→pyarrow and breaks torch 2.2.** OpenVLA's recipe leaves numpy
      unpinned (fine in their conda env), but our `daft` layer pulls **pyarrow 24 → numpy 2.2.6**,
      and torch 2.2.0 is compiled for numpy 1.x → `Failed to initialize NumPy: _ARRAY_API not
      found`. Fix: pin **`numpy<2`** in the daft layer. (This is why OpenVLA's "unpinned numpy"
      doesn't transfer — adding Daft changes the resolution.)
    - **`--no-deps` LIBERO drops runtime deps the ENV needs.** The CPU smoke imported only
      `libero.libero.benchmark` (clean), but constructing the env imports `libero.libero.envs`,
      whose `env_wrapper.py` does `import matplotlib.cm` → ModuleNotFound (matplotlib was in
      LIBERO's dropped requirements). Fix: add `matplotlib` + `einops` to the sim pins. Lesson:
      the import-only smoke under-tests; only a real env construction (GPU) catches these.
    - **Watch the other loose pins pip resolved** (still unpinned): gym→**0.26.2** (not 0.25.2),
      bddl→**3.6.0** (not 1.0.1). robosuite's `step` is its own 4-tuple so the gym-0.26 5-tuple
      change shouldn't reach us, but gym.spaces / bddl 3.x API drift could bite at task
      construction — the smoke test is what tells us. Pin once green.
    - **Don't pip-install the openvla package** — inference is pure HF
      `AutoModelForVision2Seq(trust_remote_code=True).predict_action`, so only transformers
      4.40.1 / tokenizers 0.19.1 / timm 0.9.10 / torch 2.2.0 (+torchvision/torchaudio) / json-numpy
      / pillow are needed. The full openvla repo is training-only.
    - **flash-attn omitted** (notoriously painful build); the policy uses
      `attn_implementation='sdpa'` on A100 — slower than flash but reliable. flash-attn parked.
    - **numpy left unpinned** to match OpenVLA's working recipe — PIN the resolved version here
      once the build succeeds (reproducibility), and watch for the np.float/np.bool removal if
      robosuite 1.4.1 still uses them under numpy≥1.24.
- **VLA-JEPA Modal is split into its own app:** `harness/rollout/modal_vla_jepa_app.py` builds
  the official `ginwind/VLA-JEPA` StarVLA server stack separately from OpenVLA. This avoids the
  Modal gotcha where one app's broken/heavy image can block an unrelated OpenVLA run. The function
  downloads the HF `ginwind/VLA-JEPA` snapshot plus `Qwen/Qwen3-VL-2B-Instruct` and
  `facebook/vjepa2-vitl-fpc64-256`, patches `LIBERO/config.{json,yaml}` to local cache paths and
  `attn_implementation: sdpa`, starts `deployment/model_server/server_policy.py` as a subprocess,
  then runs the standard Daft rollout UDF against `127.0.0.1:10093`. Still deploy-unverified:
  the first real run must validate the StarVLA requirements build, SDPA compatibility, and server
  throughput on A100.
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
