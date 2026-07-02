# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Why did my VLA fail? — comparative failure forensics on LIBERO
#
# **You can get a success rate; you can't easily answer *why* your VLA fails.**
#
# This notebook runs on the rollout parquet the harness writes — one row per **step**, episode
# metadata denormalized onto every row — and decomposes "the success rate dropped" into named,
# queryable failure modes. The headline comparison: **VLA-JEPA** (`lerobot/VLA-JEPA-LIBERO`,
# in-process via the lerobot port) vs **OpenVLA** (`openvla-7b-finetuned-libero-spatial`) on
# `libero_spatial`.
#
# Everything here is a query over per-step signal the schema preserves on purpose:
# `gripper_action` (commanded), `gripper_state` (measured finger separation), `eef_pos`.
# No video scrubbing, no re-simulation. Point it at your own `data/rollouts/` glob.

# %%
import json
from collections import Counter
from pathlib import Path

import daft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_GLOB = "data/rollouts/*/*.parquet"   # <- your rollout dir(s); one row per step

df = daft.read_parquet(DATA_GLOB)
print(f"{df.count_rows()} step rows")

# %% [markdown]
# ## 1. The commodity number: success rate
#
# This is where most eval stacks stop.

# %%
episodes = (
    df.groupby("episode_id", "policy_type")
      # success is denormalized onto every step row, so any_value() is exact
      .agg(daft.col("success").any_value().alias("success"),
           daft.col("step_idx").count().alias("steps"))
      .to_pandas()
)
sr = episodes.groupby("policy_type")["success"].agg(["mean", "count"])
sr.columns = ["success_rate", "episodes"]
print(sr.to_string(float_format=lambda v: f"{v:.0%}"))

# %% [markdown]
# ## 2. The wedge: isolate the failures and read their per-step signal
#
# One Daft filter gives us every step of every failed episode. From the per-step columns we
# compute a small set of **behavioral features** per failed episode:
#
# - `close_cycles` — commanded gripper close-transitions (`gripper_action` crossing to +1).
#   More than one = the policy re-attempted a grasp.
# - `held_frac` — fraction of closed-commanded steps where the fingers stopped **>4 mm apart**
#   (something between them) rather than closing on air (~1 mm; thresholds read off real
#   episodes — see the histogram below).
# - `max_lift` — end-effector rise above its episode-min z while commanding closed.
# - `steps` — episode length (cap-outs smell like timeouts or loops).

# %%
fail_steps = (
    df.where(df["success"] == False)  # noqa: E712
      .select("episode_id", "policy_type", "instruction", "step_idx",
              "gripper_action", "gripper_state", "eef_pos", "video_path")
      .to_pandas()
)

HOLD_MM = 0.004   # measured finger separation, while commanded closed: >4mm => holding something
AIR_MM = 0.002    # <2mm while closed => fingers met: closed on air


def episode_features(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("step_idx")
    ga = g["gripper_action"].to_numpy(dtype=float)     # commanded: -1 open / +1 close
    gs = g["gripper_state"].to_numpy(dtype=float)      # measured finger separation (m)
    z = np.stack(g["eef_pos"].to_numpy())[:, 2]
    closes = np.flatnonzero((ga[1:] > 0) & (ga[:-1] <= 0)) + 1
    closed = ga > 0
    held = closed & (gs > HOLD_MM)
    return pd.Series({
        "policy_type": g["policy_type"].iloc[0],
        "instruction": g["instruction"].iloc[0],
        "video_path": g["video_path"].iloc[0],
        "steps": len(g),
        "close_cycles": int(len(closes)),
        "held_frac": float(held.sum() / max(closed.sum(), 1)),
        "ever_held": bool(held.any()),
        "max_lift": float(z[held].max() - z.min()) if held.any() else 0.0,
        "closed_on_air_frac": float(((gs < AIR_MM) & closed).sum() / max(closed.sum(), 1)),
    })


feats = (
    fail_steps.groupby("episode_id").apply(episode_features, include_groups=False)
    if len(fail_steps) else pd.DataFrame()
)
feats.head(10)

# %%
# Ground the thresholds in the data: measured finger separation while commanding "close".
closed_steps = fail_steps[fail_steps["gripper_action"] > 0]
if len(closed_steps):
    plt.figure(figsize=(7, 2.6))
    plt.hist(closed_steps["gripper_state"], bins=60, color="#4878a8")
    plt.axvline(AIR_MM, color="#d62728", ls="--", lw=1, label=f"closed on air (<{AIR_MM*1000:.0f}mm)")
    plt.axvline(HOLD_MM, color="#2ca02c", ls="--", lw=1, label=f"holding (>{HOLD_MM*1000:.0f}mm)")
    plt.xlabel("measured finger separation while commanded closed (m)")
    plt.ylabel("steps")
    plt.legend(fontsize=8)
    plt.title("What the gripper actually held, across all failed episodes", fontsize=10)
    plt.tight_layout()

# %% [markdown]
# ## 3. Name the failure modes
#
# A tiny rule set over those features labels every failure. The vocabulary is the schema's
# `TERMINAL_FAILURE_LABELS` — the notebook writes what the rollout left as `unlabeled`.

# %%
def label_failure(f: pd.Series) -> str:
    if f.close_cycles >= 2 and f.ever_held:
        return "re_grasp"           # grasp -> lose -> re-attempt (the fumble loop)
    if not f.ever_held:
        return "no_grasp"           # never got the object between the fingers
    if f.max_lift < 0.02:
        return "grasp_no_lift"      # held it but never lifted
    if f.close_cycles == 1:
        return "missed_target"      # held + lifted + still failed => wrong placement
    return "timeout"


if len(feats):
    feats["failure_mode"] = feats.apply(label_failure, axis=1)
    mix = feats.groupby(["policy_type", "failure_mode"]).size().unstack(fill_value=0)
    print(mix.to_string())

# %% [markdown]
# ## 4. The hero: watch a re-grasp loop without watching anything
#
# The annotated per-step trace of the most fumble-prone failure — commanded close/open
# transitions against the *measured* finger separation, with end-effector height overlaid.
# Every marker is a grasp attempt; separations collapsing to ~1 mm are the gripper closing on
# air.

# %%
if len(feats):
    regrasps = feats[feats["failure_mode"] == "re_grasp"]
    hero_id = (regrasps if len(regrasps) else feats).sort_values("close_cycles").index[-1]
    hero = fail_steps[fail_steps["episode_id"] == hero_id].sort_values("step_idx")
    ga = hero["gripper_action"].to_numpy(dtype=float)
    gs = hero["gripper_state"].to_numpy(dtype=float) * 1000  # mm
    z = np.stack(hero["eef_pos"].to_numpy())[:, 2]
    t = np.arange(len(ga))
    closes = np.flatnonzero((ga[1:] > 0) & (ga[:-1] <= 0)) + 1
    opens = np.flatnonzero((ga[1:] < 0) & (ga[:-1] >= 0)) + 1

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, gs, lw=1.8, color="#1f77b4", label="finger separation (mm, measured)")
    ax.axhline(HOLD_MM * 1000, color="#2ca02c", ls=":", lw=1)
    ax.axhline(AIR_MM * 1000, color="#d62728", ls=":", lw=1)
    for i, c in enumerate(closes):
        ax.axvline(c, color="#9467bd", alpha=0.6, lw=1)
        if i < 14:
            ax.annotate("grasp" if i == 0 else f"re-grasp {i}", (c, gs.max() * 0.98),
                        rotation=90, fontsize=7, color="#9467bd", va="top")
    for o in opens:
        ax.axvline(o, color="#ff7f0e", alpha=0.35, lw=1)
    ax2 = ax.twinx()
    ax2.plot(t, z, lw=1.2, color="gray", alpha=0.7, label="eef height (m)")
    ax2.set_ylabel("eef z (m)", color="gray")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("finger separation (mm)")
    pol = feats.loc[hero_id, "policy_type"]
    ax.set_title(f"{pol} — {int(feats.loc[hero_id, 'close_cycles'])} grasp attempts in one failed episode\n"
                 f"“{feats.loc[hero_id, 'instruction'][:80]}”  ({hero_id})",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    print("video for this episode:", feats.loc[hero_id, "video_path"])

# %% [markdown]
# ## 5. The payoff: the two policies fail *differently*
#
# The success-rate gap says one model is better. The failure mix says **what to fix**.

# %%
if len(feats) and feats["policy_type"].nunique() >= 1:
    share = (feats.groupby("policy_type")["failure_mode"]
                  .value_counts(normalize=True).unstack(fill_value=0))
    order = [c for c in ["re_grasp", "no_grasp", "grasp_no_lift", "missed_target", "timeout"]
             if c in share.columns]
    ax = share[order].plot.bar(figsize=(8, 3.6), width=0.75,
                               color=["#9467bd", "#d62728", "#ff7f0e", "#1f77b4", "gray"])
    ax.set_ylabel("share of that policy's failures")
    ax.set_title("Failure-mode mix per policy — the why behind the success rate", fontweight="bold")
    ax.legend(fontsize=8, title=None)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if "re_grasp" in share.columns and share["re_grasp"].gt(0).sum() >= 2:
        r = share["re_grasp"].sort_values()
        print(f"{r.index[-1]}'s failures are {r.iloc[-1] / max(r.iloc[0], 1e-9):.1f}x more often "
              f"re-grasp fumble loops than {r.index[0]}'s.")

# %% [markdown]
# ## Reproduce this
#
# ```bash
# modal run harness/rollout/modal_vla_jepa_app.py --suites libero_spatial --episodes 10 --seed 7
# modal run harness/rollout/modal_app.py --policy-type openvla --suites libero_spatial --episodes 10 --seed 7
# modal volume get daft-model-outputs rollouts/ data/rollouts/
# ```
#
# Protocol notes: `libero_spatial`, all 10 tasks, deterministic init states 0–9, seed 7 — an
# honest 10-trial-per-task subset of the canonical 50-trial protocol (constants and the full
# evaluation grammar: [`docs/EVAL_PATTERNS.md`](../docs/EVAL_PATTERNS.md)). Thresholds
# (`HOLD_MM`, `AIR_MM`) were read off real episodes in §2's histogram — recheck them if your
# gripper differs. Per-episode mp4s sit next to the parquet (`video_path` column) when you do
# want to *see* the fumble the query found.

# %%
# Machine-readable summary (used by the README/blog):
if len(feats):
    print(json.dumps({
        "episodes": int(len(episodes)),
        "success_rate": {k: round(float(v), 3) for k, v in
                         episodes.groupby("policy_type")["success"].mean().items()},
        "failures": int(len(feats)),
        "failure_mix": {p: dict(Counter(g["failure_mode"]))
                        for p, g in feats.groupby("policy_type")},
    }, indent=2))
