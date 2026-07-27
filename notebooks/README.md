# Analysis notebooks

The real-data analysis is maintained as a paired Jupytext notebook:

- [`failure_modes.py`](failure_modes.py) is the reviewable source of record.
- [`failure_modes.ipynb`](failure_modes.ipynb) is synchronized from that source for interactive
  use.

It reads the paired LIBERO-Spatial pilot artifact labeled `2026-07-02` and computes
episode-level outcomes, per-task counts, paired outcomes, descriptive Wilson intervals, and
automated behavioral **candidates** from per-step signals. The label is an artifact identifier;
the execution timestamp was not recorded.

Important interpretation limits:

- the pilot contains 10 fixed initial states per task, not the 50-trial reference evaluation;
- stored/requested seed 7 does not describe the simulator seed; the referenced path used 0,
  while the exact executed code is unavailable, so it is treated as 0/unverified;
- policy RNG was not explicitly controlled or recorded;
- all 17 raw `terminal_failure` values are `unlabeled`; and
- `repeated_close_candidate` is a post-hoc signal rule, not a human-validated re-grasp, drop,
  reacquisition, or causal diagnosis.

`rollout-v1` rows are transition-aligned: image/state are from `obs_t`; action is `action_t`;
reward, terminal flag, end-effector position, and gripper state are post-action values associated
with `obs_{t+1}`. The terminal next frame is not stored. The notebook intentionally compares
the commanded action with the post-action gripper signal, but the row is not a simultaneous
sensor snapshot.

The checksummed source artifact is
[`results/libero-spatial-pilot-2026-07-02/`](../results/libero-spatial-pilot-2026-07-02/).
Read its manifest and limitations before reusing an aggregate.

[`regrasp_demo.py`](regrasp_demo.py) is separate synthetic demonstration code. Its scripted
scenarios are useful for exercising visualization and detector mechanics, but they are not
empirical evidence about either policy and must not be combined with the pilot result.
