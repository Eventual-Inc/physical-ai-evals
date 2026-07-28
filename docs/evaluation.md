# Evaluation protocol

VLA success rates are sensitive to evaluator details that are not captured by a suite name
and seed alone. Record the following for every result:

- code revision and dirty-tree status;
- model repository and immutable revision;
- suite, task id, initial-state id, and number of trials;
- Python, NumPy, Torch, policy, and simulator seeds;
- camera selection, orientation, resize, and crop;
- proprioceptive-state fields;
- action normalization and gripper convention;
- action-chunk length and execution policy;
- stabilization steps and episode step cap; and
- simulator, benchmark, CUDA, driver, and dependency versions.

## Reference differences

The pinned OpenVLA and VLA-JEPA LIBERO evaluators use 50 fixed initial states per task, but
their runtime behavior differs:

| Setting | OpenVLA reference | VLA-JEPA reference |
|---|---|---|
| General seed default | 7 | 7 |
| LIBERO environment seed | hard-coded 0 | requested seed |
| Agent-view image | rotated 180 degrees | rotated 180 degrees |
| Step cap, spatial suite | 220 | 250 |
| Action behavior | one action per observation | checkpoint-defined chunks |

Sources:

- [OpenVLA evaluator](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py)
- [OpenVLA LIBERO utilities](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/libero_utils.py)
- [VLA-JEPA evaluator](https://github.com/ginwind/VLA-JEPA/blob/ec8c70f6e155e2377bbd4d787004c14179c00c7c/examples/LIBERO/eval_libero.py)

Treat these as model-specific reference implementations, not one interchangeable protocol.

## Stored row timing

The current `rollout-v1` schema stores a transition per row:

| Fields | Timing |
|---|---|
| `frame_path`, `wrist_path`, `state` | observation before the action |
| `action`, `gripper_action` | selected action |
| `reward`, `done` | transition result |
| `eef_pos`, `gripper_state` | observation after the action |

The terminal next image is not stored. Analysis that combines pre-action image/state with
post-action end-effector or gripper values must account for that offset.

## Output identity

`episode_id` identifies an episode specification and can intentionally be shared by multiple
policies. Group step records by `(policy_type, episode_id)`, not `episode_id` alone.

Every published result should include its resolved configuration, per-episode outcomes,
per-task aggregates, raw step records or a documented omission, and checksums.
