# The VLA evaluation grammar

How the research community actually evaluates VLA policies on benchmarks — the shared
pattern, the one-to-one translations per (model × benchmark), and the canonical terminology.
Extracted 2026-07-01 from the real eval entry points of **openpi**, **starVLA** (VLA-JEPA's
base), **openvla**, **lerobot**, and **allenai/vla-evaluation-harness**, verified against
files, not memory. This is the spec our harness aligns to.

## 1. The nine components (every framework has all nine; only the names differ)

| Component | openpi | starVLA | OpenVLA | lerobot | vla-eval (allenai) |
|---|---|---|---|---|---|
| Checkpoint | train-config name + ckpt dir | `--ckpt_path` (HF id/dir) | `pretrained_checkpoint` | `--policy.path` | `checkpoint`/`config_name` (YAML) |
| Normalization stats | "norm stats" `assets/<asset_id>/norm_stats.json` | `dataset_statistics.json` + server-side `PolicyNormProcessor`; **`unnorm_key`** | **`unnorm_key`** into `model.norm_stats` (baked in ckpt) | dataset `stats` + `Normalizer/UnnormalizerProcessorStep` | `unnorm_key`/`unnorm_type` passthrough |
| Policy wrapper | `Policy` + `LiberoInputs/Outputs` | `PolicyServerWrapper` + `model2{bench}_client.py` | `get_model`/`get_vla_action` | `PreTrainedPolicy` | `ModelServer.predict()` |
| Serving split | **policy server** (websocket+msgpack, handshake metadata) | same pattern, port 10093 | **in-process** | in-process (`lerobot-eval`) | mandatory `vla-eval serve` + Docker benchmark |
| Env construction | LIBERO `benchmark_dict[suite]()`; `gymnasium.make` | identical LIBERO code | identical (the **origin**) | `EnvConfig` registry → `make_env` | `Benchmark.get_tasks()` |
| Chunk handling | **`action_horizon`**, `ActionChunkBroker`, `replan_steps` / `open_loop_horizon` | **`action_chunk_size`**, `AdaptiveEnsembler` | none (chunk of 1) | **`chunk_size`** vs **`n_action_steps`** | `chunk_size` + `action_ensemble` (newest/average/ema) |
| Success criterion | env `done` | same | same | `is_success` terminal info | `EpisodeResult = {"success": bool}` |
| Trials × seeds | 50 trials/task, **seed=7**, `init_states[episode_idx]`, `num_steps_wait=10` | identical | identical (origin) | `eval.n_episodes`, seed=1000 | `episodes_per_task=50`, `params.seed=7` |
| Recording | mp4/episode + console SR | mp4 + logs | wandb + mp4 | videos + JSON | SQLite keyed on `eval_id`, shard merge |

**The invariant that dominates everything** (stated loudest by starVLA's client-side
`[TRAIN/TEST CONSISTENCY CHECK]` banner): *eval-time observation construction must byte-match
training-time preprocessing* — image size/rotation/crop, camera count **and order**, state
on/off and dim order, unnorm stats, chunk size. Mismatches never raise; they surface as a
mysteriously low success rate. This is the pain the per-step rollout schema exists to make
debuggable.

Canonical LIBERO protocol constants (OpenVLA-origin, inherited verbatim by openpi/starVLA/vla-eval):
**50 trials/task · seed=7 · `num_steps_wait=10` dummy steps · 256px render, 180° rotation,
pad-resize 224 · max_steps: spatial 220 / object 280 / goal 300 / libero_10 520 / libero_90 400**
(we use 250 for spatial as the non-truncating cross-policy cap — VLA-JEPA's published cap; documented deviation).

## 2. One-to-one translations: {π0, OpenVLA, VLA-JEPA} × {LIBERO, ALOHA-sim, DROID}

**Structural fact first: DROID has no simulator.** DROID eval = the physical Franka rig
(`droid.robot_env.RobotEnv`, success judged by a human) or community alternatives (RoboArena
distributed real-robot eval; nascent real-to-sim efforts). SimplerEnv covers Google-Robot +
WidowX/Bridge only — not DROID. **DROID's role in a modern stack is data (ingest, fine-tune,
analyze), not benchmark.** ALOHA's sim story is `gym-aloha` (AlohaTransferCube/Insertion).

| | LIBERO (robosuite/MuJoCo) | ALOHA-sim (gym-aloha) | DROID |
|---|---|---|---|
| **π0/π0.5 (openpi)** | ✅ `pi05_libero` (SOTA) | ✅ `pi0_aloha_sim` | ⚠️ real-robot only (`pi05_droid` et al.) |
| **OpenVLA** | ✅ `openvla-7b-finetuned-libero-*` | ❌ can't express 14-dim bimanual (OFT territory) | ⚠️ no ckpt; base evals via SimplerEnv proxies |
| **VLA-JEPA (starVLA)** | ✅ three routes (below) | ❌ no ckpt, no starVLA ALOHA bench | ❌ no sim, no ckpt |

### π0 × LIBERO
- Serve: `uv run scripts/serve_policy.py --env=LIBERO policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero`
- Client obs dict: `observation/image` (agentview 256 → rot180 → pad-resize 224), `observation/wrist_image`, `observation/state` = concat(eef_pos, quat2axisangle(eef_quat), gripper_qpos) → **8-dim**, `prompt`
- Action: 7-dim delta-EEF+gripper; chunk `action_horizon=10`, execute **`replan_steps=5`**
- Env: LIBERO OffScreenRenderEnv, protocol constants above

### π0 × ALOHA-sim
- Serve: `serve_policy.py --env=ALOHA_SIM` (ckpt `pi0_aloha_sim`)
- Env: `gymnasium.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")`
- Obs: `{"state": 14-dim joints, "images": {"cam_high": 224 CHW}}`; action 14-dim joints, 50 Hz, `ActionChunkBroker(action_horizon=10)`

### π0 × DROID (real rig; documented for completeness)
- `RobotEnv(action_space="joint_velocity", gripper_action_space="position")`, 15 Hz
- Obs: exterior ZED + wrist cam (pad-resize 224) + `joint_position`(7) + `gripper_position`(1)
- Action: 8-dim (7 joint-velocity + gripper pos); `open_loop_horizon=8`; norm-stats asset_id `droid`

### OpenVLA × LIBERO  *(implemented in this repo — `harness/policies/openvla.py`)*
- In-process: `AutoModelForVision2Seq` + `predict_action(unnorm_key=<suite>[_no_noops], do_sample=False)`
- Single 224 image (rot180, center-crop area 0.9 if trained w/ augs), **no proprio, no chunking**
- Gripper: normalize [0,1]→[-1,1] then **invert** (RLDS 0=close vs env -1=open)

### VLA-JEPA × LIBERO  *(implemented — `harness/policies/vla_jepa.py` = route 1)*
1. **starVLA policy server** (the official repo path): `deployment/model_server/server_policy.py` + websocket client; server-side unnorm via `dataset_statistics.json`; `{"image": [agentview, wrist], "lang": str}` @224; chunk from ckpt config; DDIM 10 steps
2. **allenai vla-eval**: `configs/model_servers/starvla/*.yaml` + LIBERO benchmark configs
3. **lerobot** now ships `policies/vla_jepa/` (port of ginwind/VLA-JEPA; `chunk_size=7`) + a `libero` env type → `lerobot-eval --env.type=libero` with **no starVLA server infra**

## 3. Terminology lexicon (use these words)

- **episode**: the universal eval unit. **rollout** = the act of running one. **trajectory** = a *dataset* record (e.g. DROID `trajectory.h5`). Don't interchange.
- **benchmark → task suite → task → episode/trial**: the hierarchy (vla-eval formalizes it; LIBERO's own code confusingly calls suites "benchmarks").
- **success rate (SR)**: per-task, then aggregate. CALVIN uses avg rollout length instead.
- **trials**: `num_trials_per_task` (=`episodes_per_task`); canonical 50.
- **action chunk**: the predicted action sequence. Predicted length: `action_horizon` (openpi) / `action_chunk_size` (starVLA) / `chunk_size` (lerobot). Executed-before-replan: `replan_steps` or `open_loop_horizon` (openpi) / `n_action_steps` (lerobot). Overloaded even within openpi — always disambiguate predicted vs executed.
- **proprio(ceptive) state**: `observation/state` (openpi) / `agent_pos` (gym) / `observation.state` (lerobot). OpenVLA takes none.
- **embodiment**: robot platform + action-space identity (openpi encodes as norm-stats `asset_id`: `franka`, `trossen`, `droid`, ...).
- **unnorm_key / dataset statistics**: OpenVLA coined `unnorm_key`; starVLA + vla-eval adopted verbatim; openpi says "norm stats".
- **policy server / remote inference**: openpi's coinage; websocket+msgpack+handshake-metadata is the de-facto standard (starVLA port 10093; vla-eval mandates the split).
- **init states**: LIBERO's fixed per-task initial states, indexed by episode for reproducibility.
- **language instruction**: `prompt` (openpi) / `lang` (starVLA) / `task_description` (vla-eval, SimplerEnv) / `task` (lerobot).
- **closed-loop eval** vs vla-eval's **"realtime"/Sim2Live** (env keeps moving during inference latency).
- **SimplerEnv**: "visual matching" and "variant aggregation" — the two named real-to-sim setups.

## 4. Where this repo maps on

| Grammar component | Here |
|---|---|
| Policy wrapper | `harness/rollout/policy.py::Policy` (reset/act) |
| In-process route | `policies/openvla.py` |
| Policy-server route | `policies/vla_jepa.py` (websocket client, server-side style unnorm) |
| Env construction + protocol | `rollout/libero_runner.py` + `config.py` (canonical constants) |
| Chunk handling | inside each policy adapter (one action out per `act`) |
| Recording | `writer.RolloutWriter` → per-step `ROLLOUT_SCHEMA` parquet (+frames/mp4) |
| Trials × seeds | `RolloutConfig` (50 trials, seed 7, `init_state_id` in the episode key) |

What the per-step parquet adds that none of the five frameworks record: the **failure
forensics layer** — every step's action/gripper/eef/object state queryable after the fact, so
"SR dropped" decomposes into *policy* failures (re-grasp, drop) vs *harness* failures (bad
unnorm ⇒ saturated actions; bad init state ⇒ object teleport; preprocessing drift) instead of
a silent scalar.

## 5. Our stance: in-process by default, no policy server

The ecosystem's websocket "policy server" solves three problems we don't have:
1. **Real-robot network boundary** (openpi's origin) — out of scope; our evals are sim.
2. **Dependency isolation** (starVLA's reason: model env vs benchmark env) — disproven for our
   cases: our A100 run executed robosuite + MuJoCo EGL + torch 2.2 + a 7B VLA in ONE process
   (see NOTES.md), and lerobot ships `vla_jepa` + a `libero` env in one package.
3. **Batched serving at scale** (vla-eval's shard×server design) — `@daft.cls(gpus=1.0)` on a
   Modal GPU *is* the model server: Daft does placement/concurrency/batching over episode-spec
   rows; Modal does GPU provisioning. The container boundary replaces the socket.

For **Genesis** a socket is actively harmful: its native mode is GPU-parallel batched envs
sharing the CUDA context with the policy — observations never leave the device. In-process is
the only pattern that preserves that (and enables a future batched runner: one UDF call = B
episodes stepping in lockstep in one Genesis scene).

**The `Policy` ABC is the seam.** In-process adapters are the default route for every model we
target: OpenVLA (HF `predict_action`), VLA-JEPA (the lerobot `policies/vla_jepa` port), π0
(openpi's `policy_config.create_trained_policy(...).infer(obs)` — the server is just a wrap
around this). The websocket **client** adapter is retained as an escape hatch: for models whose
env genuinely cannot co-install, and to evaluate anything already served by an openpi/starVLA
server without porting it. Envs sit behind the runner's gym-shaped seam
(`set_init_state/step/check_success/close`) — MuJoCo/robosuite today, Genesis as a second
backend later.

Positioning vs allenai/vla-eval: they buy *generality* with a mandatory network boundary and
per-benchmark Docker; we buy *reproducibility* with one opinionated process — Python, Daft,
Modal, PyTorch, MuJoCo/Genesis — where the whole evaluation is a DataFrame operation. Their
protocol stays reachable through the adapter if we ever want to sit behind their harness.
Trade named honestly: a sim segfault kills the worker, not an env client — absorbed by
one-parquet-part-per-episode (crash loses nothing) + Daft/Modal row-level retries.
