# BACKLOG — not shipping this month

The June scope is **exactly three deliverables** (harness · notebook · blog), each ~1/3
of the month. Distribution's third is non-negotiable; if the harness runs over, the
notebook shrinks — distribution never does. Everything below is **explicitly out of
scope for June.** This file is the pressure-release valve: new ideas land here instead of
into the month, so scope creep can't eat the distribution window.

## Hard "not this month" (from the project contract)

These are decided. Do not scaffold them; do not let them grow in June.

- **SIMPLER benchmark** — only LIBERO this month. (There is a `lerobot/VLA-JEPA-SimplerEnv`
  checkpoint; tempting, but out.)
- **Formats beyond {DROID, LeRobot, HDF5}** — no RT-X/OXE-wide readers, no bespoke lab
  formats. The three adapters are the contract.
- **Fine-tuning anything** — we evaluate released checkpoints, we don't train. No LoRA, no
  OFT runs, no dataset regeneration beyond what an eval needs.
- **Real-robot data** — sim rollouts (LIBERO) and offline datasets only.
- **A leaderboard** — the wedge is *why* a VLA fails, not a success-rate scoreboard. A
  leaderboard reframes us as the commodity metric we're explicitly not selling.
- **Archetype integration** — no.

## Parked ideas (revisit AFTER June, only if a deliverable pulls them)

Ordered roughly by how cleanly they'd extend the existing contracts.

- **Contribute the `trajectory.h5` parser to Daft's DROID reader.** `daft.datasets.droid.raw()`
  hands you a `daft.File` to each episode's `trajectory.h5` but leaves parsing actions/proprio
  as an explicit upstream TODO — exactly our HDF5 wheelhouse. A Daft expression that reads it
  fills the gap AND is distribution gold (a merged PR to Eventual's own repo = credibility with
  the exact ICP). Distribution-flavored; revisit once our HDF5 path works.
- **LeRobot v2.1 reader.** Daft's LeRobot reader from PR #7090 is v3.0-only; v2.1 still needs
  a separate path if a deliverable needs training-format conversion.

- **Automatic failure labeling beyond re-grasp.** The schema already has slots for the full
  `TERMINAL_FAILURE_LABELS` vocabulary (no_grasp, drop_no_recover, wrong_object,
  missed_target, collision, timeout). June only needs re-grasp to land the hero moment; a
  general rule-based or learned labeler is a natural follow-up.
- **Image-embedding clustering (vs instruction-text clustering).** The `embedding` column
  and `EmbedConfig.modality='image'` are already wired; clustering on frame embeddings (a
  `@daft.cls` SigLIP/DINO encoder over `frame_path`) is a richer second pass.
- **`StreamingLeRobotDataset` ingest** — stream from the Hub without local download
  (v3.0 feature) for datasets too big to land on disk.
- **Lazy in-Daft LeRobot projection.** `ingest/lerobot.py` materializes Daft's `read()` to
  build Episodes (guaranteed schema parity, but not lazy). A scale path would project native
  columns → `ROLLOUT_SCHEMA` *in Daft* and `write_parquet` directly, staying lazy over huge
  datasets. Parked until a dataset big enough to need it appears.
- **OpenVLA-OFT / community checkpoints** (`moojink/openvla-7b-oft-*`) as extra baselines.
- **Object-pose tracking from segmentation** (`SegmentationRenderEnv`) to make
  `object_poses` richer/more reliable for drop detection, instead of proprio-derived heuristics.
- **A tiny committed parquet fixture** (`tests/fixtures/`) so the notebook + CI run with
  zero GPU and zero sim. The `.gitignore` already whitelists `tests/fixtures/**/*.parquet`.
- **Re-emitting harness rollouts back into LeRobot v2.1** (we have the writer path notes) so
  captured rollouts are re-trainable. Out of scope but the writer already documents
  `finalize()`.
- **Parallel rollout sweep** (`env.max_parallel_tasks` / batched eval) for throughput once
  the single-env loop is correct.
- **GPU memory snapshot for VLA cold starts** — the daft-examples `cls_kwargs` /
  `@modal.enter(snap=True)` pattern. Our `@daft.cls` loads lazily on the Daft worker (after the
  snapshot window) so it's off; wire the warm-load into the snapshot to enable it. Worth it once
  a full 400-episode sweep is the bottleneck.
- **Multibase entry points** (mentioned in the original README stub) — deferred; the Daft
  story is the June wedge.

## Rule

If an idea isn't one of the three deliverables, it goes here, not into the month. Adding to
this list is free; adding to June costs the distribution window.
