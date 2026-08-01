# Evaluation protocol

## Execution boundary

One `evaluate()` call constructs one policy and one benchmark runtime. The
policy remains loaded across every pending episode. The simulator runtime
caches the active environment and initial-state set.

For each episode:

1. seed Python, NumPy, Torch, CUDA, and the environment;
2. call `env.reset()` before `set_init_state()` so robosuite's horizon state
   cannot leak across episodes;
3. execute stabilization actions;
4. rotate both LIBERO camera views by 180 degrees;
5. reset the policy with the selected instruction;
6. run actions until `done` or the suite step cap;
7. stream both cameras to temporary MP4s and atomically rename them;
8. write the typed step partition; and
9. write the episode outcome as the completion marker.

A completion is resumable only when its step indices are unique and contiguous
from zero and the episode `length` matches the step count. Corrupt Parquet is
ignored and re-run. A crash after steps but before the completion row causes the
same episode partition to be overwritten.

## Benchmark mappings

| Benchmark | Simulator task | Instruction | Initial state |
|---|---|---|---|
| LIBERO | selected standard suite/task | standard task language | standard fixed states |
| LIBERO-Para | corresponding LIBERO-Goal task ID | selected Para BDDL language | standard LIBERO-Goal fixed states |
| LIBERO-Pro | published Pro BDDL | published Pro BDDL language | paired published Pro `.pruned_init` |

All three use the pinned LIBERO-Pro code revision
`eafdb809426b13153aa1e4c42d6601844217dfec`. The fork retains the standard
benchmark definitions and adds the Pro objects/assets needed by perturbed BDDL.
Para dataset revision
`d306f66f8b441cad1155b21a3f69e440079c81c9` and Pro dataset revision
`c86fc3b8293185a6f373677018ff3e37f8391602` are recorded in manifests.

## Policy mappings

OpenVLA checkpoint and action unnormalization keys are locked to one base
suite. LIBERO-Para uses the LIBERO-Goal checkpoint. LIBERO-Pro uses the
checkpoint for its base suite.

VLA-JEPA uses:

- checkpoint `lerobot/VLA-JEPA-LIBERO@735d9f692981e286ade093b5046627eda876e5d0`;
- LeRobot `052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34`;
- Qwen3-VL `89644892e4d85e24eaac8bacfd4f463576704203`; and
- V-JEPA2 `b3c1679b7c34d3255ef3547f27c7b226aefab26f`.

The latter two repository names are mutable inside the upstream checkpoint
config, so the adapter replaces them with exact local snapshots before model
construction.

The `vla_jepa_cutile` policy is an action-only alternative. It loads the same
policy checkpoint and pinned Qwen3-VL source into a persistent B4 daft-cuTile
session, but does not construct V-JEPA2 or LeRobot. Each inference produces a
seven-step action chunk, which the rollout consumes one simulator frame at a
time. Explicit episode-keyed noise makes action chunks stable across cohort
ordering and resume boundaries.

## Output schema

`manifest.json` contains the schema version, immutable policy identity,
benchmark revision and metadata, canonical episode-spec hash, implementation
source hash, interpreter/platform versions, package versions, and GPU details.
These values are not repeated per step.

`episodes` contains one row per episode specification, including task identity,
instruction, initial-state identity, outcome, length, and relative video paths.
`steps` contains the LeRobot-aligned transition fields:

| Field | Type |
|---|---|
| `episode_index`, `frame_index` | `Int64` |
| `timestamp` | `Float32` |
| `action` | `Tensor[Float32; 7]` |
| `observation.state` | `Tensor[Float32; 8]` |
| `observation.eef_position` | `Tensor[Float32; 3]` |
| `observation.gripper` | `Float32` |
| `reward` | `Float32` |
| `next.done` | `Bool` |

The trace mirrors LeRobot's episode/step/video concepts, but is not a complete
LeRobot training repository: it does not synthesize `meta/info.json`, task
indices, aggregate statistics, or multi-episode video shards.

## Reproduction gates

The CPU conformance benchmark runs in the repository's one Python 3.12
environment and pins:

- exact canonical episode and step content signatures;
- one policy construction across all episodes;
- zero policy construction on a complete resume;
- rollout-crash recovery;
- steps-before-completion recovery;
- corrupt and wrong-schema Parquet handling; and
- real MP4 finalization.

Modal smoke functions construct, reset, initialize, and step a real MuJoCo
environment in each policy image. They accept all three benchmark families.

GPU acceptance compares per-episode `success` and `length`, not bitwise action
tensors. CUDA kernels are not forced into deterministic algorithms because that
would make the validation path slower and less representative of production.
The cuTile lane additionally fails closed on its native transfer counters: host
camera/telemetry ingress and terminal action readback are allowed; intermediate
host-visible tensors and extra synchronization are not.

## Historical trace

The published
[`physical-ai-evals-libero-spatial-pilot`](https://huggingface.co/datasets/Eventual-Inc/physical-ai-evals-libero-spatial-pilot)
at revision `ddb8a88fcc579ebf077a9ca2d1e026a7e1cf4429` remains a
`rollout-v1` read fixture. It is not regenerated as `eval-v1` and is not a
benchmark reproduction reference: its simulator seed was unverified, it used
10 rather than 50 initial states, and it lacks current model/runtime provenance.
