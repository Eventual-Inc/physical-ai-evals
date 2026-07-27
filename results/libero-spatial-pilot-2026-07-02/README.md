# LIBERO-Spatial paired pilot (artifact `2026-07-02`)

This is an **auditable analysis bundle for an exploratory pilot**, not a replayable evaluation, LIBERO reference evaluation, or confirmatory model comparison. It contains 200 policy/spec episode rows: OpenVLA and VLA-JEPA evaluated on the same 100 stored `(suite, task, init_state, seed)` specifications (10 tasks × 10 initial states). The traces contain 23,283 step rows.

## Observed result

| Policy | Successes | Episodes | Rate | Descriptive 95% Wilson interval |
|---|---:|---:|---:|---:|
| VLA-JEPA | 99 | 100 | 99% | 94.55%–99.82% |
| OpenVLA | 84 | 100 | 84% | 75.58%–89.90% |

Paired outcomes: 84 both succeeded, 15 VLA-JEPA-only succeeded, 0 OpenVLA-only succeeded, and 1 both failed. These numbers describe this fixed pilot only. They do not establish general model superiority.

## Files

- `steps.parquet` — all source step rows merged and deterministically sorted, Zstandard-compressed; source values and schema metadata are unchanged.
- `episodes.csv` — one row per policy/spec (200 rows), with stored metadata, integrity fields, and explicitly prefixed derived heuristic fields.
- `failure_signatures.csv` — raw derived behavioral features and non-causal candidate labels for the 17 failed episodes.
- `summary.json` — overall, per-task, and paired counts/rates with descriptive Wilson intervals.
- `manifest.json` — protocol, advertised commands, source-file hashes, artifact hashes, validation results, and provenance limitations.
- `SHA256SUMS` — checksums for every bundle file except the checksum file itself.
- `build_bundle.py` — deterministic builder and validation logic; requires Python and PyArrow.

Verify the committed bundle from this directory:

```bash
shasum -a 256 -c SHA256SUMS
```

If the original ignored `data/rollouts/*/*.parquet` files are present, rebuild and revalidate from any working directory:

```bash
python results/libero-spatial-pilot-2026-07-02/build_bundle.py
```

## Provenance limits

- The advertised commands requested seed 7 and every trace stores `seed=7`, but the referenced/current rollout path constructed the LIBERO environment with `env_seed=0`. The exact evaluated code revision is unavailable, so simulator seed must be treated as **0/unverified**, not confirmed seed 7.
- Policy-side random-number generation was not controlled or recorded.
- OpenVLA's stored `model` field is blank. `openvla/openvla-7b-finetuned-libero-spatial` is the intended checkpoint inferred from the suite and advertised code path, not verified trace metadata.
- Evaluation code revision, dependency lock, model revisions, container image, hardware details, and execution timestamps were not recorded and cannot be reconstructed from these files.
- This pilot uses 10 initial states per task, not the LIBERO reference protocol of 50 trials per task, and only one requested/stored seed.
- Media referenced by source `video_path`/frame columns was not present in the local evidence set and is not bundled.

## Rollout row timing

`rollout-v1` rows describe a transition, not one same-instant snapshot. `frame_path`/`wrist_path` and `state` refer to pre-action observation `obs_t`; `action` is `action_t`; and `reward`, `done`, `eef_pos`, and `gripper_state` come from post-action observation `obs_{t+1}`. The terminal `obs_{t+1}` image is not captured, so the final transition has no next-image counterpart.

## Heuristic labels

The source `terminal_failure` values are preserved (`unlabeled` for every failed episode). Derived labels in `episodes.csv` and `failure_signatures.csv` are behavioral **candidates**, not causal or video-validated failure diagnoses. `failure-signatures-v1` pairs `action_t` with post-action `gripper_state` and `eef_pos` from `obs_{t+1}`; those fields are transition-aligned but are not a same-instant snapshot. It counts commanded close transitions and uses measured finger separation (`>4 mm` while commanded closed means “held”); `repeated_close_candidate` must not be reported as a confirmed drop and re-grasp. The 4 mm hold and 2 mm closed-on-air thresholds were selected post hoc from this dataset. Review `build_bundle.py` for the complete, ordered rule set.

Wilson intervals are descriptive episode-level summaries and ignore pairing, task clustering, seed uncertainty, uncontrolled policy randomness, and analysis/model selection. No hypothesis test or population-level claim is made.
