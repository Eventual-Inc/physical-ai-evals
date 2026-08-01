# physical-ai-evals

`physical-ai-evals` is a Python 3.12 evaluation harness for:

- OpenVLA and VLA-JEPA;
- LIBERO, LIBERO-Para, and LIBERO-Pro;
- crash-safe episode/step/video traces; and
- revision-checked LeRobot v3 reads through Daft.

The package is flat by design. `evaluate()` owns the stateful rollout boundary;
Daft owns specifications, resume anti-joins, typed Parquet storage, lazy reads,
and metrics. Modal supplies pinned Torch policy images plus a separate H100/CUDA
13.3 lane for persistent daft-cuTile VLA-JEPA inference.

Start with the
[repository README](https://github.com/Eventual-Inc/physical-ai-evals#readme),
then use:

- [Evaluation protocol](evaluation.md) for benchmark semantics and provenance;
- [Dataset readers](datasets.md) for LeRobot queries; and
- [Troubleshooting](troubleshooting.md) for policy/simulator environments.
