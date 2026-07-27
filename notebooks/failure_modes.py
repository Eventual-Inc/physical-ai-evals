# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Paired pilot analysis and automated behavioral candidates
#
# This notebook analyzes the LIBERO-Spatial pilot artifact labeled `2026-07-02`: 10 tasks ×
# fixed initial-state indices 0–9 × two policies. The label is not a recorded execution
# timestamp. The notebook reports recorded outcomes first, then derives explicitly heuristic
# candidate signatures from per-step signals.
#
# This is **not** the 50-trials-per-task LIBERO reference evaluation. The stored/requested seed
# is 7, while the referenced historical Modal path constructed the simulator with
# `env_seed=0`; exact executed code is unavailable, so the effective seed is 0/unverified.
# Policy RNG was not explicitly controlled or recorded. All 17 raw `terminal_failure` values
# are `unlabeled`. Nothing below turns an automated candidate into a manually verified failure
# mode or a policy-versus-harness causal attribution. In `rollout-v1`, image/state are from `obs_t`
# while `eef_pos` and `gripper_state` are from `obs_{t+1}` after `action_t`; the terminal next
# frame is absent. The action-to-post-gripper feature is transition-aligned, but a row is not a
# same-instant snapshot.

# %%
import glob as _glob
import json
import math
import os
from collections import Counter
from pathlib import Path

import daft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def _repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "docs").is_dir():
        return cwd
    if (cwd.parent / "docs").is_dir():
        return cwd.parent
    return cwd


REPO_ROOT = _repo_root()
FIGURE_DIR = REPO_ROOT / "docs" / "assets"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Set VLA_PILOT_GLOB to analyze a different artifact. In a clean clone, the checksummed bundle
# is preferred; a development checkout may instead contain the original per-episode parts.
_candidates = (
    str(REPO_ROOT / "results/libero-spatial-pilot-2026-07-02/steps.parquet"),
    str(REPO_ROOT / "data/rollouts/*/*.parquet"),
)
DATA_GLOB = os.environ.get(
    "VLA_PILOT_GLOB",
    next((pattern for pattern in _candidates if _glob.glob(pattern)), _candidates[0]),
)
if not _glob.glob(DATA_GLOB):
    raise FileNotFoundError(
        f"No pilot Parquet matched {DATA_GLOB!r}. Set VLA_PILOT_GLOB to the step-trace file(s)."
    )

df = daft.read_parquet(DATA_GLOB)
print(f"{df.count_rows():,} step rows from {DATA_GLOB}")

# %% [markdown]
# ## 1. Recorded outcomes
#
# `success` and episode metadata are denormalized onto every step row. Group by both policy and
# episode id: paired policies intentionally share the same episode id.

# %%
episodes = (
    df.groupby("policy_type", "episode_id")
    .agg(
        daft.col("success").any_value().alias("success"),
        daft.col("task_id").any_value().alias("task_id"),
        daft.col("init_state_id").any_value().alias("init_state_id"),
        daft.col("seed").any_value().alias("stored_seed"),
        daft.col("model").any_value().alias("stored_model"),
        daft.col("step_idx").count().alias("steps"),
    )
    .to_pandas()
    .sort_values(["policy_type", "task_id", "init_state_id"])
)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval; descriptive here because iid is not established."""
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return center - half_width, center + half_width


outcome_rows = []
for policy, group in episodes.groupby("policy_type", sort=True):
    successes = int(group["success"].sum())
    trials = int(len(group))
    lower, upper = wilson_interval(successes, trials)
    outcome_rows.append(
        {
            "policy": policy,
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials,
            "wilson_95_low": lower,
            "wilson_95_high": upper,
        }
    )
outcomes = pd.DataFrame(outcome_rows).set_index("policy")
print(outcomes.to_string(float_format=lambda value: f"{value:.3f}"))
print("\nWilson intervals are descriptive; fixed states nested in tasks are not established iid draws.")

# %%
per_task = (
    episodes.groupby(["task_id", "policy_type"])["success"]
    .sum()
    .astype(int)
    .unstack("policy_type")
    .sort_index()
)
print("Per-task successes (10 fixed initial states per cell):")
print(per_task.to_string())

paired = episodes.pivot(
    index=["task_id", "init_state_id"], columns="policy_type", values="success"
)
required_policies = {"openvla", "vla_jepa"}
if not required_policies.issubset(paired.columns) or paired[list(required_policies)].isna().any().any():
    raise ValueError("Expected one paired outcome for each policy at every task/init-state key")

paired_counts = {
    "both_success": int((paired["openvla"] & paired["vla_jepa"]).sum()),
    "vla_jepa_only": int((~paired["openvla"] & paired["vla_jepa"]).sum()),
    "openvla_only": int((paired["openvla"] & ~paired["vla_jepa"]).sum()),
    "both_failure": int((~paired["openvla"] & ~paired["vla_jepa"]).sum()),
}
print("\nPaired outcomes:", paired_counts)

# %% [markdown]
# ## 2. Raw labels and per-step signals
#
# The rollout did not assign behavioral failure modes. The raw terminal value for every failed
# episode is `unlabeled`; the analysis below creates a separate `candidate_label` column.

# %%
fail_steps = (
    df.where(df["success"] == False)  # noqa: E712
    .select(
        "episode_id",
        "policy_type",
        "instruction",
        "terminal_failure",
        "step_idx",
        "gripper_action",
        "gripper_state",
        "eef_pos",
        "video_path",
    )
    .to_pandas()
)
fail_steps["uid"] = fail_steps["policy_type"] + "@" + fail_steps["episode_id"]

raw_terminal = (
    fail_steps[["uid", "policy_type", "terminal_failure"]]
    .drop_duplicates()
    .groupby(["policy_type", "terminal_failure"], dropna=False)
    .size()
)
print(raw_terminal.to_string())

# Post-hoc thresholds selected from this pilot. `gripper_action[t]` is paired with the recorded
# post-action `gripper_state[t]`. Finger separation can suggest that something impeded closure,
# but without contact/object state or annotation it does not prove object hold.
HOLD_SIGNAL_M = 0.004
AIR_SIGNAL_M = 0.002


def episode_features(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("step_idx")
    gripper_action = group["gripper_action"].to_numpy(dtype=float)
    finger_gap = group["gripper_state"].to_numpy(dtype=float)
    eef_z = np.stack(group["eef_pos"].to_numpy())[:, 2]
    closes = np.flatnonzero((gripper_action[1:] > 0) & (gripper_action[:-1] <= 0)) + 1
    commanded_closed = gripper_action > 0
    hold_gap_signal = commanded_closed & (finger_gap > HOLD_SIGNAL_M)
    return pd.Series(
        {
            "policy_type": group["policy_type"].iloc[0],
            "instruction": group["instruction"].iloc[0],
            "video_path": group["video_path"].iloc[0],
            "steps": len(group),
            "close_transitions": int(len(closes)),
            "hold_gap_fraction": float(
                hold_gap_signal.sum() / max(commanded_closed.sum(), 1)
            ),
            "has_hold_gap_signal": bool(hold_gap_signal.any()),
            "max_lift_during_hold_signal": (
                float(eef_z[hold_gap_signal].max() - eef_z.min())
                if hold_gap_signal.any()
                else 0.0
            ),
            "air_gap_fraction": float(
                ((finger_gap < AIR_SIGNAL_M) & commanded_closed).sum()
                / max(commanded_closed.sum(), 1)
            ),
        }
    )


features = (
    fail_steps.groupby("uid").apply(episode_features, include_groups=False)
    if len(fail_steps)
    else pd.DataFrame()
)
features.head(10)

# %%
closed_steps = fail_steps[fail_steps["gripper_action"] > 0]
if len(closed_steps):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.hist(closed_steps["gripper_state"], bins=60, color="#4878a8")
    ax.axvline(
        AIR_SIGNAL_M,
        color="#d62728",
        ls="--",
        lw=1,
        label=f"post-hoc air-gap threshold (<{AIR_SIGNAL_M * 1000:.0f} mm)",
    )
    ax.axvline(
        HOLD_SIGNAL_M,
        color="#2ca02c",
        ls="--",
        lw=1,
        label=f"post-hoc hold-gap threshold (>{HOLD_SIGNAL_M * 1000:.0f} mm)",
    )
    ax.set_xlabel("measured finger separation while commanded closed (m)")
    ax.set_ylabel("failed-episode steps")
    ax.legend(fontsize=8)
    ax.set_title("Exploratory threshold selection on the analyzed failures", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gripper-thresholds-histogram.png", dpi=100)

# %% [markdown]
# ## 3. Automated candidate labels
#
# The names deliberately end in `_candidate`. The rules have not been manually validated, and
# several distinct physical events can produce the same recorded signal.

# %%
def candidate_label(feature: pd.Series) -> str:
    if feature.close_transitions >= 2 and feature.has_hold_gap_signal:
        return "repeated_close_candidate"
    if not feature.has_hold_gap_signal:
        return "no_hold_signal_candidate"
    if feature.max_lift_during_hold_signal < 0.02:
        return "low_lift_candidate"
    if feature.close_transitions == 1:
        return "single_close_other_failure"
    return "unclassified_candidate"


if len(features):
    features["candidate_label"] = features.apply(candidate_label, axis=1)
    candidate_mix = (
        features.groupby(["policy_type", "candidate_label"]).size().unstack(fill_value=0)
    )
    print(candidate_mix.to_string())

# %% [markdown]
# ## 4. A selected repeated-close candidate
#
# This trace is deliberately selected for having many close transitions. It is an illustrative
# diagnostic view, not a representative episode. Command transitions and finger separation do
# not establish object contact, a drop, or reacquisition.

# %%
if len(features):
    repeated = features[features["candidate_label"] == "repeated_close_candidate"]
    pool = repeated if len(repeated) else features
    top_policy = features["policy_type"].value_counts().idxmax()
    if (pool["policy_type"] == top_policy).any():
        pool = pool[pool["policy_type"] == top_policy]
    hero_id = pool.sort_values("close_transitions").index[-1]
    hero = fail_steps[fail_steps["uid"] == hero_id].sort_values("step_idx")
    gripper_action = hero["gripper_action"].to_numpy(dtype=float)
    finger_gap_mm = hero["gripper_state"].to_numpy(dtype=float) * 1000
    eef_z = np.stack(hero["eef_pos"].to_numpy())[:, 2]
    timestep = np.arange(len(gripper_action))
    closes = np.flatnonzero((gripper_action[1:] > 0) & (gripper_action[:-1] <= 0)) + 1
    opens = np.flatnonzero((gripper_action[1:] < 0) & (gripper_action[:-1] >= 0)) + 1

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestep, finger_gap_mm, lw=1.8, color="#1f77b4", label="finger gap (mm)")
    ax.axhline(HOLD_SIGNAL_M * 1000, color="#2ca02c", ls=":", lw=1)
    ax.axhline(AIR_SIGNAL_M * 1000, color="#d62728", ls=":", lw=1)
    for index, close_step in enumerate(closes, start=1):
        ax.axvline(close_step, color="#9467bd", alpha=0.6, lw=1)
        if index <= 14:
            ax.annotate(
                f"close {index}",
                (close_step, finger_gap_mm.max() * 0.98),
                rotation=90,
                fontsize=7,
                color="#9467bd",
                va="top",
            )
    for open_step in opens:
        ax.axvline(open_step, color="#ff7f0e", alpha=0.35, lw=1)
    secondary = ax.twinx()
    secondary.plot(timestep, eef_z, lw=1.2, color="gray", alpha=0.7, label="eef height")
    secondary.set_ylabel("eef z (m)", color="gray")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("finger separation (mm)")
    policy = features.loc[hero_id, "policy_type"]
    ax.set_title(
        f"Selected repeated-close candidate: {policy}, "
        f"{int(features.loc[hero_id, 'close_transitions'])} close transitions\n"
        f"“{features.loc[hero_id, 'instruction'][:80]}” ({hero_id})",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "regrasp-hero-trace.png", dpi=100)
    print("Optional media path recorded for selected episode:", features.loc[hero_id, "video_path"])

# %% [markdown]
# ## 5. Candidate-label distribution
#
# This is a descriptive view of 17 failures. It does not establish that the policies have
# different physical failure mechanisms.

# %%
if len(features):
    counts = (
        features.groupby("policy_type")["candidate_label"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(episodes["policy_type"].unique(), fill_value=0)
    )
    preferred_order = [
        "repeated_close_candidate",
        "no_hold_signal_candidate",
        "low_lift_candidate",
        "single_close_other_failure",
        "unclassified_candidate",
    ]
    order = [label for label in preferred_order if label in counts.columns]
    episode_counts = episodes.groupby("policy_type").size()
    rates = counts.div(episode_counts, axis=0)
    colors = {
        "repeated_close_candidate": "#9467bd",
        "no_hold_signal_candidate": "#d62728",
        "low_lift_candidate": "#ff7f0e",
        "single_close_other_failure": "#1f77b4",
        "unclassified_candidate": "gray",
    }
    ax = rates[order].plot.bar(
        figsize=(8, 3.6), width=0.75, color=[colors[label] for label in order]
    )
    ax.set_ylabel("candidate rate across all episodes")
    ax.set_xlabel("")
    ax.set_title("Post-hoc candidate labels; raw failures remain unlabeled", fontweight="bold")
    total_rate = rates[order].sum(axis=1)
    ax.set_ylim(0, max(float(total_rate.max()) * 1.28, 0.05))
    for index, policy in enumerate(rates.index):
        ax.text(
            index,
            float(total_rate.loc[policy]) + 0.004,
            f"n={int(counts.loc[policy].sum())} failed episodes",
            ha="center",
            fontsize=8,
        )
    legend_labels = {
        "repeated_close_candidate": "repeated close candidate",
        "no_hold_signal_candidate": "no hold-gap signal candidate",
        "low_lift_candidate": "low lift candidate",
        "single_close_other_failure": "single-close other failure",
        "unclassified_candidate": "unclassified candidate",
    }
    ax.legend(
        handles=[Patch(color=colors[label], label=legend_labels[label]) for label in order],
        fontsize=8,
        title=None,
        loc="upper right",
    )
    plt.xticks(rotation=0)
    plt.tight_layout()
    ax.figure.savefig(FIGURE_DIR / "failure-mix-comparison.png", dpi=100)

    failure_counts = (~episodes["success"]).groupby(episodes["policy_type"]).sum().astype(int)
    print("Recorded failures:", failure_counts.to_dict())
    print("Automated candidate labels are unvalidated and should not be read as causal modes.")

# %% [markdown]
# ## Running a new sweep
#
# These commands produce a new result; they do not exactly recreate the historical pilot
# without its unrecorded policy RNG and immutable checkpoint/code revisions.
#
# ```bash
# .venv/bin/modal run harness/cloud/vla_jepa_app.py \
#   --suites libero_spatial --episodes 10 --seed 7 \
#   --model-id lerobot/VLA-JEPA-LIBERO \
#   --model-revision 735d9f692981e286ade093b5046627eda876e5d0
# .venv/bin/modal run harness/cloud/openvla_app.py \
#   --suites libero_spatial --episodes 10 --seed 7 \
#   --model-id openvla/openvla-7b-finetuned-libero-spatial \
#   --model-revision 962318cec55ac10993ff0f5f43eda9a270b4c873
# ```
#
# Read [`docs/EVAL_PATTERNS.md`](../docs/EVAL_PATTERNS.md) for the upstream implementation
# differences and the minimum provenance checklist. For any behavioral-label study, validate a
# prespecified rule against manual annotation before applying it to held-out episodes.

# %%
if len(features):
    machine_summary = {
        "study_design": {
            "suite": "libero_spatial",
            "tasks": 10,
            "fixed_initial_states_per_task": 10,
            "policies": 2,
            "paired": True,
            "reference_protocol_reproduction": False,
        },
        "seed_provenance": {
            "stored_requested_seed": 7,
            "historical_simulator_seed": "0_or_unverified",
            "policy_rng_controlled": False,
        },
        "outcomes": {
            policy: {
                "successes": int(row.successes),
                "trials": int(row.trials),
                "success_rate": round(float(row.success_rate), 3),
                "wilson_95_descriptive": [
                    round(float(row.wilson_95_low), 4),
                    round(float(row.wilson_95_high), 4),
                ],
            }
            for policy, row in outcomes.iterrows()
        },
        "paired_outcomes": paired_counts,
        "raw_terminal_failure": {
            "label": "unlabeled",
            "episodes": int(len(features)),
        },
        "automated_candidate_mix": {
            policy: {
                str(label): int(count)
                for label, count in Counter(group["candidate_label"]).items()
            }
            for policy, group in features.groupby("policy_type")
        },
        "interpretation": (
            "Wilson intervals assume iid and are descriptive only; candidate labels are "
            "post-hoc and not manually validated. rollout-v1 rows align obs_t/action_t with "
            "post-action eef/gripper values rather than a same-instant snapshot."
        ),
    }
    print(json.dumps(machine_summary, indent=2))
