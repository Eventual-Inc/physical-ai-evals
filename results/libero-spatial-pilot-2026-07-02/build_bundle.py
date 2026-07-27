#!/usr/bin/env python3
"""Build and validate the LIBERO-Spatial exploratory pilot evidence bundle.

This script intentionally preserves every source trace value.  It sorts rows while
merging, then places episode-level derivations in a separate CSV and JSON summary.
Run it from any directory in a checkout that still has data/rollouts/*/*.parquet.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ARTIFACT_ID = "libero-spatial-pilot-2026-07-02"
POLICIES = ("openvla", "vla_jepa")
EXPECTED_SUITE = "libero_spatial"
EXPECTED_SEED_STORED = 7
EXPECTED_TASK_IDS = tuple(range(10))
EXPECTED_INIT_STATE_IDS = tuple(range(10))
WILSON_Z_95 = 1.959963984540054
HOLD_THRESHOLD_M = 0.004
AIR_THRESHOLD_M = 0.002

BUNDLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_DIR.parents[1]
INPUT_GLOB = "data/rollouts/*/*.parquet"

EPISODE_CONSTANT_COLUMNS = (
    "schema_version",
    "episode_id",
    "run_id",
    "model",
    "policy_type",
    "source",
    "suite",
    "task_id",
    "task_name",
    "instruction",
    "bddl_file",
    "init_state_id",
    "seed",
    "control_mode",
    "success",
    "terminal_failure",
    "num_steps",
    "video_path",
)

CSV_COLUMNS = (
    "policy_type",
    "suite",
    "task_id",
    "task_name",
    "init_state_id",
    "seed",
    "episode_id",
    "run_id",
    "model",
    "source",
    "control_mode",
    "success",
    "terminal_failure",
    "num_steps",
    "observed_step_rows",
    "step_idx_min",
    "step_idx_max",
    "step_indices_contiguous",
    "reward_sum",
    "max_reward",
    "done_any",
    "instruction",
    "bddl_file",
    "video_path",
    "heuristic_version",
    "heuristic_failure_label",
    "heuristic_close_cycles",
    "heuristic_held_fraction_closed_steps",
    "heuristic_ever_held",
    "heuristic_max_lift_m",
    "heuristic_closed_on_air_fraction",
)

FAILURE_SIGNATURE_COLUMNS = (
    "policy_type",
    "suite",
    "task_id",
    "init_state_id",
    "seed",
    "episode_id",
    "source_terminal_failure",
    "method_version",
    "close_cycles",
    "held_frac",
    "ever_held",
    "max_lift_m",
    "closed_on_air_frac",
    "heuristic_label",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def one_value(values: list[Any], column: str, source_path: str) -> Any:
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{source_path}: {column} is not episode-constant")
    return first


def bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def float_text(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def failure_features(columns: dict[str, list[Any]], success: bool) -> dict[str, Any]:
    if success:
        return {
            "heuristic_version": "failure-signatures-v1",
            "heuristic_failure_label": "not_applicable_success",
            "heuristic_close_cycles": "",
            "heuristic_held_fraction_closed_steps": "",
            "heuristic_ever_held": "",
            "heuristic_max_lift_m": "",
            "heuristic_closed_on_air_fraction": "",
        }

    order = sorted(range(len(columns["step_idx"])), key=columns["step_idx"].__getitem__)
    actions = [float(columns["gripper_action"][index]) for index in order]
    states = [float(columns["gripper_state"][index]) for index in order]
    heights = [float(columns["eef_pos"][index][2]) for index in order]
    close_cycles = sum(
        actions[index] > 0 and actions[index - 1] <= 0
        for index in range(1, len(actions))
    )
    closed = [index for index, action in enumerate(actions) if action > 0]
    held = [index for index in closed if states[index] > HOLD_THRESHOLD_M]
    ever_held = bool(held)
    held_fraction = len(held) / max(len(closed), 1)
    closed_on_air_fraction = (
        sum(states[index] < AIR_THRESHOLD_M for index in closed) / max(len(closed), 1)
    )
    max_lift = max((heights[index] for index in held), default=min(heights)) - min(heights)

    # These are signal signatures, not video-validated causal diagnoses.
    if close_cycles >= 2 and ever_held:
        label = "repeated_close_candidate"
    elif not ever_held:
        label = "no_hold_signal_candidate"
    elif max_lift < 0.02:
        label = "hold_without_20mm_lift_candidate"
    elif close_cycles == 1:
        label = "held_lifted_but_unsuccessful_candidate"
    else:
        label = "timeout_or_other_candidate"

    return {
        "heuristic_version": "failure-signatures-v1",
        "heuristic_failure_label": label,
        "heuristic_close_cycles": str(close_cycles),
        "heuristic_held_fraction_closed_steps": float_text(held_fraction),
        "heuristic_ever_held": bool_text(ever_held),
        "heuristic_max_lift_m": float_text(max_lift),
        "heuristic_closed_on_air_fraction": float_text(closed_on_air_fraction),
    }


def episode_from_table(table: pa.Table, source_path: str) -> dict[str, Any]:
    columns = table.to_pydict()
    episode = {
        column: one_value(columns[column], column, source_path)
        for column in EPISODE_CONSTANT_COLUMNS
    }
    step_indices = sorted(int(value) for value in columns["step_idx"])
    expected_indices = list(range(int(episode["num_steps"])))
    if table.num_rows != episode["num_steps"]:
        raise ValueError(
            f"{source_path}: {table.num_rows} rows != stored num_steps {episode['num_steps']}"
        )
    if step_indices != expected_indices:
        raise ValueError(f"{source_path}: step_idx is not contiguous from zero")

    rewards = [float(value) for value in columns["reward"] if value is not None]
    done_values = [bool(value) for value in columns["done"] if value is not None]
    row = {
        **episode,
        "observed_step_rows": table.num_rows,
        "step_idx_min": step_indices[0],
        "step_idx_max": step_indices[-1],
        "step_indices_contiguous": True,
        "reward_sum": sum(rewards),
        "max_reward": max(rewards, default=0.0),
        "done_any": any(done_values),
        "source_path": source_path,
    }
    row.update(failure_features(columns, bool(episode["success"])))
    return row


def wilson(successes: int, episodes: int) -> dict[str, float]:
    if episodes <= 0:
        raise ValueError("Wilson interval requires at least one episode")
    proportion = successes / episodes
    z2 = WILSON_Z_95**2
    denominator = 1 + z2 / episodes
    center = (proportion + z2 / (2 * episodes)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1 - proportion) / episodes + z2 / (4 * episodes**2)
        )
        / denominator
    )
    return {
        "confidence_level": 0.95,
        "lower": round(max(0.0, center - half_width), 6),
        "upper": round(min(1.0, center + half_width), 6),
    }


def rate_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    failures = episodes - successes
    labels = Counter(
        row["heuristic_failure_label"] for row in rows if not bool(row["success"])
    )
    return {
        "episodes": episodes,
        "failures": failures,
        "heuristic_failure_label_counts": dict(sorted(labels.items())),
        "success_rate": round(successes / episodes, 6),
        "successes": successes,
        "wilson_score_interval": wilson(successes, episodes),
    }


def task_name_for(rows: list[dict[str, Any]], task_id: int) -> str:
    names = {row["task_name"] for row in rows if row["task_id"] == task_id}
    if len(names) != 1:
        raise ValueError(f"task {task_id}: expected one task name, found {sorted(names)}")
    return names.pop()


def make_summary(
    episodes: list[dict[str, Any]], validation: dict[str, Any]
) -> dict[str, Any]:
    overall = {
        policy: rate_record([row for row in episodes if row["policy_type"] == policy])
        for policy in POLICIES
    }
    for policy in POLICIES:
        task_rates = []
        for task_id in EXPECTED_TASK_IDS:
            task_rows = [
                row
                for row in episodes
                if row["policy_type"] == policy and row["task_id"] == task_id
            ]
            task_rates.append(sum(bool(row["success"]) for row in task_rows) / len(task_rows))
        overall[policy]["macro_task_success_rate"] = round(
            sum(task_rates) / len(task_rates), 6
        )

    per_task = []
    for task_id in EXPECTED_TASK_IDS:
        per_task.append(
            {
                "policies": {
                    policy: rate_record(
                        [
                            row
                            for row in episodes
                            if row["policy_type"] == policy and row["task_id"] == task_id
                        ]
                    )
                    for policy in POLICIES
                },
                "specs": 10,
                "task_id": task_id,
                "task_name": task_name_for(episodes, task_id),
            }
        )

    by_spec: dict[tuple[Any, ...], dict[str, bool]] = {}
    for row in episodes:
        spec = (row["suite"], row["task_id"], row["init_state_id"], row["seed"])
        by_spec.setdefault(spec, {})[row["policy_type"]] = bool(row["success"])
    outcome_counts = Counter()
    for result in by_spec.values():
        if result["openvla"] and result["vla_jepa"]:
            outcome_counts["both_success"] += 1
        elif result["openvla"]:
            outcome_counts["openvla_only_success"] += 1
        elif result["vla_jepa"]:
            outcome_counts["vla_jepa_only_success"] += 1
        else:
            outcome_counts["both_failure"] += 1
    outcomes = {}
    for name in (
        "both_success",
        "vla_jepa_only_success",
        "openvla_only_success",
        "both_failure",
    ):
        count = outcome_counts[name]
        outcomes[name] = {
            "count": count,
            "proportion": round(count / len(by_spec), 6),
            "wilson_score_interval": wilson(count, len(by_spec)),
        }

    return {
        "artifact_id": ARTIFACT_ID,
        "inference_notice": (
            "All 95% Wilson score intervals are descriptive episode-level intervals. "
            "They do not account for pairing, task clustering, simulator-seed uncertainty, "
            "uncontrolled policy RNG, or analysis/model selection and are not confirmatory."
        ),
        "overall": overall,
        "paired_outcomes": {
            "discordant_specs": outcome_counts["openvla_only_success"]
            + outcome_counts["vla_jepa_only_success"],
            "outcomes": outcomes,
            "pairing_key": ["suite", "task_id", "init_state_id", "stored_seed"],
            "specs": len(by_spec),
            "vla_jepa_minus_openvla_success_count": overall["vla_jepa"]["successes"]
            - overall["openvla"]["successes"],
            "vla_jepa_minus_openvla_success_rate": round(
                overall["vla_jepa"]["success_rate"]
                - overall["openvla"]["success_rate"],
                6,
            ),
        },
        "per_task": per_task,
        "status": "exploratory_pilot",
        "total_episode_rows": len(episodes),
        "validation": validation,
        "wilson_method": {
            "confidence_level": 0.95,
            "method": "Wilson score interval for one binomial proportion without continuity correction",
            "z": WILSON_Z_95,
        },
    }


def write_episodes_csv(episodes: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            writer.writerow(
                {
                    **{column: episode[column] for column in CSV_COLUMNS},
                    "success": bool_text(bool(episode["success"])),
                    "terminal_failure": episode["terminal_failure"] or "",
                    "model": episode["model"],
                    "step_indices_contiguous": bool_text(
                        bool(episode["step_indices_contiguous"])
                    ),
                    "reward_sum": float_text(float(episode["reward_sum"])),
                    "max_reward": float_text(float(episode["max_reward"])),
                    "done_any": bool_text(bool(episode["done_any"])),
                }
            )
    temporary.replace(path)


def write_failure_signatures_csv(episodes: list[dict[str, Any]], path: Path) -> None:
    failures = [episode for episode in episodes if not bool(episode["success"])]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=FAILURE_SIGNATURE_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for episode in failures:
            writer.writerow(
                {
                    "policy_type": episode["policy_type"],
                    "suite": episode["suite"],
                    "task_id": episode["task_id"],
                    "init_state_id": episode["init_state_id"],
                    "seed": episode["seed"],
                    "episode_id": episode["episode_id"],
                    "source_terminal_failure": episode["terminal_failure"] or "",
                    "method_version": episode["heuristic_version"],
                    "close_cycles": episode["heuristic_close_cycles"],
                    "held_frac": episode["heuristic_held_fraction_closed_steps"],
                    "ever_held": episode["heuristic_ever_held"],
                    "max_lift_m": episode["heuristic_max_lift_m"],
                    "closed_on_air_frac": episode[
                        "heuristic_closed_on_air_fraction"
                    ],
                    "heuristic_label": episode["heuristic_failure_label"],
                }
            )
    temporary.replace(path)


def read_and_validate_sources() -> tuple[list[pa.Table], list[dict[str, Any]], list[dict[str, Any]]]:
    source_files = sorted(REPO_ROOT.glob(INPUT_GLOB))
    if len(source_files) != 200:
        raise ValueError(f"expected exactly 200 source Parquet files, found {len(source_files)}")

    tables: list[pa.Table] = []
    episodes: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    reference_schema: pa.Schema | None = None
    for path in source_files:
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        table = pq.read_table(path)
        if reference_schema is None:
            reference_schema = table.schema
        elif table.schema != reference_schema:
            raise ValueError(f"{relative_path}: schema differs from first source file")
        tables.append(table)
        episodes.append(episode_from_table(table, relative_path))
        inputs.append(
            {
                "bytes": path.stat().st_size,
                "path": relative_path,
                "rows": table.num_rows,
                "sha256": sha256_file(path),
            }
        )
    return tables, episodes, inputs


def validate_collection(
    tables: list[pa.Table], episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    policy_specs = [
        (
            row["policy_type"],
            row["suite"],
            row["task_id"],
            row["init_state_id"],
            row["seed"],
        )
        for row in episodes
    ]
    duplicate_policy_spec_rows = len(policy_specs) - len(set(policy_specs))
    specs = [(suite, task, init, seed) for _, suite, task, init, seed in policy_specs]
    unique_specs = sorted(set(specs))
    policy_sets = {
        spec: {
            row["policy_type"]
            for row in episodes
            if (row["suite"], row["task_id"], row["init_state_id"], row["seed"])
            == spec
        }
        for spec in unique_specs
    }
    fully_paired_specs = sum(policy_set == set(POLICIES) for policy_set in policy_sets.values())
    missing_pair_members = sum(len(set(POLICIES) - policy_set) for policy_set in policy_sets.values())

    step_keys: set[tuple[Any, ...]] = set()
    duplicate_step_keys = 0
    for table in tables:
        columns = table.to_pydict()
        for index, step_idx in enumerate(columns["step_idx"]):
            key = (
                columns["policy_type"][index],
                columns["suite"][index],
                columns["task_id"][index],
                columns["init_state_id"][index],
                columns["seed"][index],
                step_idx,
            )
            if key in step_keys:
                duplicate_step_keys += 1
            step_keys.add(key)

    expected_specs = {
        (EXPECTED_SUITE, task, init, EXPECTED_SEED_STORED)
        for task in EXPECTED_TASK_IDS
        for init in EXPECTED_INIT_STATE_IDS
    }
    assertions = {
        "duplicate_policy_spec_rows": duplicate_policy_spec_rows == 0,
        "duplicate_step_keys": duplicate_step_keys == 0,
        "expected_policy_spec_rows": len(policy_specs) == 200,
        "expected_specs": set(unique_specs) == expected_specs,
        "fully_paired": fully_paired_specs == 100 and missing_pair_members == 0,
        "source_file_per_policy_spec": len(tables) == len(policy_specs),
    }
    if not all(assertions.values()):
        raise ValueError(f"collection validation failed: {assertions}")

    return {
        "all_assertions_passed": True,
        "assertions": assertions,
        "duplicate_policy_spec_rows": duplicate_policy_spec_rows,
        "duplicate_step_keys": duplicate_step_keys,
        "fully_paired_specs": fully_paired_specs,
        "missing_pair_members": missing_pair_members,
        "source_files": len(tables),
        "step_rows": sum(table.num_rows for table in tables),
        "unique_policy_spec_rows": len(set(policy_specs)),
        "unique_specs": len(unique_specs),
    }


def write_merged_steps(tables: list[pa.Table], path: Path) -> None:
    merged = pa.concat_tables(tables)
    order = pc.sort_indices(
        merged,
        sort_keys=[
            ("policy_type", "ascending"),
            ("suite", "ascending"),
            ("task_id", "ascending"),
            ("init_state_id", "ascending"),
            ("seed", "ascending"),
            ("step_idx", "ascending"),
        ],
    )
    merged = merged.take(order)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(
        merged,
        temporary,
        compression="zstd",
        compression_level=9,
        data_page_version="2.0",
        row_group_size=4096,
        use_dictionary=True,
        version="2.6",
        write_statistics=True,
    )
    temporary.replace(path)


README = """# LIBERO-Spatial paired pilot (artifact `2026-07-02`)

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
"""


def make_manifest(
    inputs: list[dict[str, Any]],
    validation: dict[str, Any],
    artifact_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    input_listing = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in inputs
    ).encode("utf-8")
    return {
        "artifact_id": ARTIFACT_ID,
        "artifact_label_date": "2026-07-02",
        "artifact_label_date_note": (
            "Version identifier only; the evaluation execution timestamp was not recorded."
        ),
        "artifacts": artifact_records,
        "builder": {
            "command": f"python results/{ARTIFACT_ID}/build_bundle.py",
            "pyarrow_version": pa.__version__,
            "python_version": platform.python_version(),
            "script": f"results/{ARTIFACT_ID}/build_bundle.py",
        },
        "commands": {
            "evidence_note": (
                "These commands were advertised alongside the rollout analysis; exact executed "
                "commands and evaluation revision were not recorded."
            ),
            "openvla_advertised": (
                "modal run harness/modal_app.py --policy-type openvla "
                "--suites libero_spatial --episodes 10 --seed 7"
            ),
            "retrieve_advertised": (
                "modal volume get daft-model-outputs rollouts/ data/rollouts/"
            ),
            "vla_jepa_advertised": (
                "modal run harness/modal_vla_jepa_app.py --suites libero_spatial "
                "--episodes 10 --seed 7"
            ),
        },
        "inputs": {
            "collection_sha256": hashlib.sha256(input_listing).hexdigest(),
            "files": inputs,
            "glob": INPUT_GLOB,
            "source_file_count": len(inputs),
            "source_total_bytes": sum(entry["bytes"] for entry in inputs),
            "source_total_rows": sum(entry["rows"] for entry in inputs),
        },
        "known_provenance_gaps": [
            {
                "field": "simulator_seed",
                "observed": (
                    "Advertised command requested seed 7 and trace rows store seed 7; the "
                    "referenced/current rollout path constructed LIBERO with env_seed=0."
                ),
                "treatment": (
                    "Source metadata is unchanged. Treat simulator seed as 0/unverified, not "
                    "confirmed seed 7."
                ),
            },
            {
                "field": "policy_rng",
                "observed": "Policy-side RNG state and seeds were not controlled or recorded.",
                "treatment": "Do not claim deterministic policy inference or exact rerun equivalence.",
            },
            {
                "field": "openvla_model",
                "observed": "Every OpenVLA trace row has an empty model field.",
                "treatment": (
                    "openvla/openvla-7b-finetuned-libero-spatial is inferred as the intended "
                    "checkpoint from suite and advertised path, not verified as executed."
                ),
            },
            {
                "field": "evaluation_code_and_environment",
                "observed": (
                    "Evaluation code revision, dependency/model revisions, container image, "
                    "hardware details, and execution timestamps were not recorded."
                ),
                "treatment": "The original execution environment is not reconstructable.",
            },
            {
                "field": "terminal_next_image",
                "observed": (
                    "rollout-v1 stores the current frame from obs_t, but does not capture the "
                    "terminal next image from obs_{t+1}."
                ),
                "treatment": (
                    "Do not infer terminal visual state from the final row's frame or treat a "
                    "row as a same-instant snapshot."
                ),
            },
        ],
        "protocol": {
            "artifact_status": "exploratory_pilot",
            "reference_protocol_deviation": (
                "10 initial states per task rather than the LIBERO reference protocol of 50 trials per task; "
                "one requested/stored seed; simulator seed provenance is discrepant."
            ),
            "episodes_per_policy": 100,
            "init_state_ids": list(EXPECTED_INIT_STATE_IDS),
            "pairing_key": ["suite", "task_id", "init_state_id", "stored_seed"],
            "policies": {
                "openvla": {
                    "stored_model": "",
                    "intended_model_inferred_not_verified": (
                        "openvla/openvla-7b-finetuned-libero-spatial"
                    ),
                },
                "vla_jepa": {"stored_model": "lerobot/VLA-JEPA-LIBERO"},
            },
            "requested_seed_advertised": 7,
            "simulator_seed_status": "env_seed_0_or_unverified",
            "stored_seed": 7,
            "suite": EXPECTED_SUITE,
            "task_ids": list(EXPECTED_TASK_IDS),
            "trials_per_task_per_policy": 10,
        },
        "schema": {
            "episodes": "episodes-v1",
            "failure_heuristics": {
                "air_threshold_m": AIR_THRESHOLD_M,
                "hold_threshold_m": HOLD_THRESHOLD_M,
                "status": "post_hoc_unvalidated_behavioral_signatures",
                "timing_note": (
                    "Uses action_t with post-action gripper_state and eef_pos from obs_{t+1}; "
                    "these fields are transition-aligned, not same-instant snapshots."
                ),
                "version": "failure-signatures-v1",
            },
            "rollout_v1_row_timing": {
                "action": "action_t",
                "done": "post-action obs_{t+1}",
                "eef_pos": "post-action obs_{t+1}",
                "frame_path": "pre-action obs_t",
                "gripper_state": "post-action obs_{t+1}",
                "reward": "post-action obs_{t+1}",
                "rows_are_same_instant_snapshots": False,
                "state": "pre-action obs_t",
                "terminal_next_image": "absent",
                "wrist_path": "pre-action obs_t",
            },
            "source_steps": "rollout-v1",
            "summary": "pilot-summary-v1",
        },
        "validation": validation,
    }


def main() -> None:
    tables, episodes, inputs = read_and_validate_sources()
    policy_order = {policy: index for index, policy in enumerate(POLICIES)}
    episodes.sort(
        key=lambda row: (
            policy_order[row["policy_type"]],
            row["suite"],
            row["task_id"],
            row["init_state_id"],
            row["seed"],
        )
    )
    validation = validate_collection(tables, episodes)

    steps_path = BUNDLE_DIR / "steps.parquet"
    episodes_path = BUNDLE_DIR / "episodes.csv"
    failure_signatures_path = BUNDLE_DIR / "failure_signatures.csv"
    summary_path = BUNDLE_DIR / "summary.json"
    manifest_path = BUNDLE_DIR / "manifest.json"
    readme_path = BUNDLE_DIR / "README.md"
    write_merged_steps(tables, steps_path)
    write_episodes_csv(episodes, episodes_path)
    write_failure_signatures_csv(episodes, failure_signatures_path)
    write_bytes(summary_path, json_bytes(make_summary(episodes, validation)))
    write_bytes(readme_path, README.encode("utf-8"))

    artifacts = {}
    for path, rows in (
        (steps_path, validation["step_rows"]),
        (episodes_path, len(episodes)),
        (failure_signatures_path, sum(not bool(row["success"]) for row in episodes)),
        (summary_path, None),
    ):
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "rows": rows,
            "sha256": sha256_file(path),
        }
    write_bytes(manifest_path, json_bytes(make_manifest(inputs, validation, artifacts)))

    checksum_paths = sorted(
        [
            Path(__file__),
            episodes_path,
            failure_signatures_path,
            manifest_path,
            readme_path,
            steps_path,
            summary_path,
        ],
        key=lambda path: path.name,
    )
    checksum_lines = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
    )
    write_bytes(BUNDLE_DIR / "SHA256SUMS", checksum_lines.encode("utf-8"))

    print(json.dumps({
        "artifact_id": ARTIFACT_ID,
        "checksums": {path.name: sha256_file(path) for path in checksum_paths},
        "validation": validation,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
