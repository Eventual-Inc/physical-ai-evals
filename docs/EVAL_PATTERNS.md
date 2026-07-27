# Evaluation protocol and provenance

This page separates three things that are easy to conflate:

1. defaults in upstream evaluation implementations;
2. choices made by this repository's current harness; and
3. the configuration actually evidenced by a stored result.

The source review below was refreshed on 2026-07-10. Upstream code links are pinned to exact
Git revisions so later changes do not silently alter the evidence.

## There is no single seed convention

The commonly repeated shorthand “50 trials per task, seed 7” hides materially different RNG
behavior.

| Implementation | Trials per task | General RNG | LIBERO environment RNG | Fixed initial states |
|---|---:|---|---|---|
| OpenVLA LIBERO evaluator | 50 | `set_seed_everywhere(7)` by default | `env.seed(0)` is hardcoded | `initial_states[episode_idx]` |
| VLA-JEPA LIBERO evaluator | 50 | `numpy.random.seed(7)` by default | the requested seed is passed to `env.seed` | `initial_states[episode_idx]` |
| AllenAI VLA evaluation harness config | 50 | benchmark parameter `seed: 7` | delegated to its benchmark implementation | benchmark-managed |

Sources: OpenVLA's
[`GenerateConfig` and evaluation loop](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L57-L98),
its
[`get_libero_env`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/libero_utils.py#L18-L25),
VLA-JEPA's
[`Args`, RNG setup, and environment construction](https://github.com/ginwind/VLA-JEPA/blob/ec8c70f6e155e2377bbd4d787004c14179c00c7c/examples/LIBERO/eval_libero.py),
and AllenAI's pinned
[`libero/spatial.yaml`](https://github.com/allenai/vla-evaluation-harness/blob/ac8b6943fd0575e2993496742c8ddc1aeac4bb0b/configs/benchmarks/libero/spatial.yaml).

Consequently, a credible manifest should record at least:

- the requested CLI seed;
- the seed passed to the simulator;
- framework RNG seeds and deterministic settings;
- policy sampling/generator state;
- task id and fixed initial-state index; and
- whether the environment was reset before `set_init_state`.

A single `seed` column cannot represent all of these.

## Reference implementation crosswalk

The OpenVLA and VLA-JEPA evaluators agree on several structural choices, but their source code
should be treated as reference implementations rather than a universal LIBERO standard.

| Component | OpenVLA reference | VLA-JEPA reference | This harness |
|---|---|---|---|
| Task suite | LIBERO benchmark registry | LIBERO benchmark registry | same registry |
| Episode selection | fixed state at `episode_idx` | fixed state at `episode_idx` | task id + fixed `init_state_id` |
| Trials per task | 50 | 50 | configurable; historical pilot used 10 |
| Stabilization | 10 dummy actions | 10 dummy actions | configurable, default 10 |
| Camera render | 256 × 256 | 256 × 256 | configurable |
| Spatial step cap | 220 | 250 | 250 unless overridden |
| Success | LIBERO `done` | LIBERO `done` | recorded Boolean terminal outcome |
| Recording | logs, optional W&B, MP4 | logs and MP4 | one Parquet part per episode, optional media |

The OpenVLA constants and loop are visible in its pinned
[`run_libero_eval.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L66-L85)
and
[episode loop](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L153-L229).
The VLA-JEPA constants and loop are in its pinned
[`eval_libero.py`](https://github.com/ginwind/VLA-JEPA/blob/ec8c70f6e155e2377bbd4d787004c14179c00c7c/examples/LIBERO/eval_libero.py#L42-L64).

### Observation and action preprocessing

Preprocessing is part of the evaluated system, not an incidental implementation detail.
OpenVLA's reference path:

- rotates the agent-view image by 180 degrees before resize;
- uses the task-suite name as the initial action unnormalization key, with a `_no_noops`
  fallback when present in checkpoint statistics;
- conditionally center-crops for checkpoints trained with image augmentation; and
- maps and inverts the gripper output before calling the LIBERO environment.

Primary sources: pinned
[`libero_utils.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/libero_utils.py#L28-L58)
and
[`run_libero_eval.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py#L94-L112).

VLA-JEPA's upstream evaluator also rotates the observation and obtains action normalization and
chunk configuration from the checkpoint. See pinned
[`eval_libero.py`](https://github.com/ginwind/VLA-JEPA/blob/ec8c70f6e155e2377bbd4d787004c14179c00c7c/examples/LIBERO/eval_libero.py#L150-L227)
and
[`model2libero_interface.py`](https://github.com/ginwind/VLA-JEPA/blob/ec8c70f6e155e2377bbd4d787004c14179c00c7c/examples/LIBERO/model2libero_interface.py#L94-L164).

Any change to image orientation, resize/crop, camera order, proprioceptive state, action
unnormalization, gripper convention, chunk execution, stabilization steps, or step cap creates a
different evaluated system. These fields belong in a result manifest.

## Historical pilot configuration

The bundled result labeled `2026-07-02` is intentionally described as an exploratory paired
pilot; the label is not a recorded execution timestamp.

| Field | Artifact evidence | Limitation |
|---|---|---|
| Suite | `libero_spatial` | one suite only |
| Tasks | ids 0–9 | all ten spatial tasks |
| Initial states | ids 0–9 | 10 rather than 50 per task |
| Pairing | same task/init ids for both policies | useful for descriptive paired counts |
| Stored/requested seed | 7 | not the effective simulator seed |
| Effective simulator seed | 0 in the referenced historical Modal path | exact executed code unavailable; treat as 0/unverified |
| Policy RNG | not recorded | exact replay is not established |
| OpenVLA model | empty raw `model` field | expected checkpoint/revision not independently proven |
| VLA-JEPA model | `lerobot/VLA-JEPA-LIBERO` | revision not recorded |
| Terminal failure labels | 17 × `unlabeled` | behavioral labels are post-hoc candidates |

The result therefore does not satisfy the repository's future provenance requirements, even
though the paired episode keys and stored trajectories permit useful auditing. The full record
and checksums are in the
[pilot bundle](https://github.com/Eventual-Inc/physical-ai-evals/tree/main/results/libero-spatial-pilot-2026-07-02).

## `rollout-v1` row timing

A pilot step row is transition-aligned, not a same-instant state snapshot. For row `t`, the
current rollout loop records:

| Field group | Time semantics |
|---|---|
| `frame_path`, `wrist_path`, `state` | pre-action observation `obs_t` |
| `action`, `gripper_action` | action `action_t` selected from `obs_t` |
| `reward`, `done` | transition result after applying `action_t` |
| `eef_pos`, `gripper_state` | post-action observation `obs_{t+1}` |

The loop extracts image/state, calls the policy, steps the environment, and only then reads the
end-effector and gripper values; see
[`harness/bench/libero.py`](https://github.com/Eventual-Inc/physical-ai-evals/blob/main/harness/bench/libero.py#L97-L106).
The terminal `obs_{t+1}` image/state is not stored, so the last media frame precedes the
terminal transition.

The notebook's commanded-action to post-action finger-gap heuristic is therefore aligned to a
transition response. It must not be interpreted as though image, proprioceptive state,
end-effector position, and finger separation were sampled simultaneously. A future schema
revision should use explicit `obs_*` and `next_obs_*` fields and retain the terminal next
observation.

## Mapping to this repository

| Responsibility | Current path |
|---|---|
| Episode and step model | `harness/core/episode.py` |
| Parquet contract | `harness/core/schema.py` |
| Rollout configuration | `harness/core/config.py` |
| Policy interface | `harness/policy/base.py` |
| OpenVLA adapter | `harness/policy/openvla.py` |
| VLA-JEPA adapter | `harness/policy/vla_jepa.py` |
| LIBERO loop | `harness/bench/libero.py` |
| Cloud rollout UDF | `harness/cloud/rollout_udf.py` |
| OpenVLA Modal app | `harness/cloud/openvla_app.py` |
| VLA-JEPA Modal app | `harness/cloud/vla_jepa_app.py` |
| Exploratory analysis | `notebooks/failure_modes.py` |

OpenVLA and VLA-JEPA use separate Modal images because their dependency requirements differ.
Each episode worker invokes its selected policy and simulator together. This describes the
current deployment; it is not evidence that co-location is generally superior to a policy
server or that every VLA stack can share an environment.

### Controls for future runs

The current Modal path adds safeguards that are **not retroactive evidence** for the historical
pilot:

- OpenVLA accepts one suite per run, resolves the suite-specific checkpoint and unnormalization
  key, and rejects incompatible or ambiguous combinations.
- Known OpenVLA checkpoints and VLA-JEPA resolve to immutable Hugging Face revisions; custom
  checkpoints require an explicit 40-character revision.
- The row seed now controls Python, NumPy, Torch, CUDA (when available), and LIBERO environment
  seeding for that episode.
- Evaluation-affecting configuration, source fingerprints, runtime versions, and GPU details
  are written to a manifest under a configuration-derived evaluation id.
- Resume validation checks suite, task, initial state, seed, policy, and immutable model identity
  before reusing an episode part.
- The OpenVLA image checks out LIBERO source revision
  [`8f1084e`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01);
  the VLA-JEPA image pins LeRobot revision
  [`052d329`](https://github.com/huggingface/lerobot/tree/052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34).

These controls make a new run more auditable. They cannot recover the execution revision,
policy RNG state, or model revision missing from the traces in the artifact labeled
`2026-07-02`.

## Failure analysis: observation before attribution

The schema preserves signals that may help form diagnostic hypotheses: actions, measured
gripper state, end-effector position, rewards, terminal state, and optional media paths. It does
not make a policy-versus-harness causal attribution.

For the pilot, all failed rows have `terminal_failure="unlabeled"`. The notebook derives
candidate signatures under explicit thresholds. A repeated close command with a finger-gap
signal is not sufficient evidence of an object drop and reacquisition. Before promoting a
candidate to a failure mode, manually annotate a prespecified sample, report agreement and
error rates, and freeze the rule before applying it to a held-out result.

## Minimum result checklist

A research-facing result should contain:

- code revision and dirty-tree status;
- resolved model repository and immutable revision for every policy;
- suite, task ids, initial-state ids, trial count, and pairing key;
- requested, simulator, framework, and policy RNG seeds;
- camera, preprocessing, state, action, gripper, and chunk configuration;
- simulator, benchmark, CUDA, driver, and key dependency versions;
- per-episode outcomes and per-task aggregates;
- raw step traces or a documented reason they cannot be shared;
- a machine-readable manifest and checksums; and
- limitations that distinguish recorded facts, inferences, and unvalidated heuristics.

## Primary source ledger

- [LIBERO paper](https://arxiv.org/abs/2306.03310) and
  [source at `8f1084e`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)
- [OpenVLA paper](https://arxiv.org/abs/2406.09246) and
  [source at `c8f03f4`](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9)
- [VLA-JEPA source at `ec8c70f`](https://github.com/ginwind/VLA-JEPA/tree/ec8c70f6e155e2377bbd4d787004c14179c00c7c)
- [LeRobot source revision used by this repository](https://github.com/huggingface/lerobot/tree/052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34),
  including the
  [VLA-JEPA policy](https://github.com/huggingface/lerobot/tree/052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34/src/lerobot/policies/vla_jepa)
- [AllenAI VLA evaluation harness at `ac8b694`](https://github.com/allenai/vla-evaluation-harness/tree/ac8b6943fd0575e2993496742c8ddc1aeac4bb0b)
- [OpenVLA-Spatial model revision `962318c`](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial/tree/962318cec55ac10993ff0f5f43eda9a270b4c873)
- [VLA-JEPA model revision `735d9f6`](https://huggingface.co/lerobot/VLA-JEPA-LIBERO/tree/735d9f692981e286ade093b5046627eda876e5d0)
- [openpi at `15a9616`](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac),
  including its
  [LIBERO example](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac/examples/libero)
