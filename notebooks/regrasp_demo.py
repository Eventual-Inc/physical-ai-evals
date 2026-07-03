"""Re-grasp detection demo — proves the wedge on synthetic rollouts (no GPU).

Builds a handful of synthetic OpenVLA / VLA-JEPA *failure* rollouts in ROLLOUT_SCHEMA, reads
them back with Daft (the real "rollouts -> queryable DataFrame" flow), runs the re-grasp
detector off the per-step gripper + object-height signal, and renders the hero screenshot:
an annotated grasp -> lift -> slip/drop -> re-grasp trajectory + the OpenVLA-vs-VLA-JEPA
failure-mode mix.

⚠ SYNTHETIC DATA. This proves the DETECTION + VISUALIZATION; the real numbers come from real
rollouts. The figure is watermarked so it's never mistaken for measured results.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from harness.ingest.base import Episode, Step
from harness.writer import write_episode

TARGET = "akita_black_bowl"
rng = np.random.default_rng(0)


# --------------------------------------------------------------- synthetic rollouts

def build_signals(scenario: str):
    """Return (grip_closed[bool], obj_z[float]) for a failure scenario."""
    gc: list[bool] = []
    z: list[float] = []

    def add(n, closed, z0, z1=None):
        z1 = z0 if z1 is None else z1
        zs = np.linspace(z0, z1, n) + rng.normal(0, 0.002, n)
        gc.extend([closed] * n)
        z.extend(np.clip(zs, 0.0, None).tolist())

    if scenario in ("regrasp2", "regrasp1"):
        add(6, False, 0.0)               # approach
        add(1, True, 0.0)                # grasp
        add(6, True, 0.0, 0.15)          # lift
        add(3, True, 0.15, 0.02)         # slip / drop
        add(2, False, 0.02)              # gripper reopens
        add(1, True, 0.02)               # RE-GRASP
        add(7, True, 0.02, 0.16)         # lift again
        add(3, True, 0.16, 0.02)         # drop again
        if scenario == "regrasp2":
            add(2, False, 0.02)          # reopen
            add(1, True, 0.02)           # re-grasp #2
            add(6, True, 0.02, 0.15)
            add(3, True, 0.15, 0.02)
        add(3, False, 0.02)              # give up -> fail
    elif scenario == "drop":
        add(6, False, 0.0)
        add(1, True, 0.0)
        add(7, True, 0.0, 0.15)          # grasp + lift
        add(3, True, 0.15, 0.01)         # drop
        add(6, False, 0.01)              # never recovers
    elif scenario == "nograsp":
        add(20, False, 0.0)              # never closes on the object
    return gc, z


def make_episode(eid: str, policy: str, scenario: str) -> Episode:
    gc, z = build_signals(scenario)
    n = len(gc)
    steps = []
    for t in range(n):
        closed = gc[t]
        action = np.zeros(7, np.float32)
        action[-1] = 1.0 if closed else -1.0          # gripper command: +1 close / -1 open
        steps.append(
            Step(
                timestep=t, action=action, reward=0.0, done=(t == n - 1), is_terminal=(t == n - 1),
                eef_pos=np.array([0.0, 0.0, 0.2 + z[t]], np.float32),
                gripper_state=float(0.01 if closed else 0.08),   # finger separation: small=closed
                object_poses={TARGET: [0.0, 0.0, float(z[t]), 0.0, 0.0, 0.0, 1.0]},
            )
        )
    return Episode(
        eid, "libero", "put the bowl on the plate", tuple(steps),
        success=False, terminal_failure="unlabeled",
        model=f"{policy}-demo", policy_type=policy, suite="libero_spatial",
        task_id=0, task_name="put_bowl",
    )


# openvla fails mostly by re-grasping; vla_jepa mostly clean drops -> the wedge
SCENARIOS = {
    "openvla": ["regrasp2", "regrasp1", "regrasp2", "regrasp1", "drop"],   # 4/5 re_grasp
    "vla_jepa": ["drop", "drop", "regrasp1", "drop", "nograsp"],           # 1/5 re_grasp
}


# --------------------------------------------------------------- re-grasp detector

def detect_regrasp(obj_z, grip_closed, *, lift=0.05, drop=0.03):
    """Label an episode and return its event timeline from the per-step signal.

    grasp = gripper closes; lift = object rises above `lift`; drop = a lifted object falls below
    `drop`; re-grasp = the gripper closes AGAIN after a drop. >=1 re-grasp -> 're_grasp'.
    """
    events, lifted, drops, regrasps, prev = [], False, 0, 0, False
    for t, (z, c) in enumerate(zip(obj_z, grip_closed)):
        if c and not prev:
            events.append((t, "re-grasp" if drops > 0 else "grasp"))
            regrasps += drops > 0
        if z > lift and not lifted:
            lifted = True
            events.append((t, "lift"))
        if lifted and z < drop:
            lifted, drops = False, drops + 1
            events.append((t, "drop"))
        prev = c
    if regrasps >= 1:
        label = "re_grasp"
    elif drops >= 1:
        label = "drop_no_recover"
    elif any(grip_closed):
        label = "missed_target"
    else:
        label = "no_grasp"
    return label, events, regrasps


# --------------------------------------------------------------- run it through Daft

def main():
    import daft

    outdir = Path(tempfile.mkdtemp()) / "rollouts"
    for policy, scns in SCENARIOS.items():
        for i, s in enumerate(scns):
            write_episode(make_episode(f"libero_spatial/0/{i}/{policy}", policy, s), outdir,
                          run_id=f"demo-{policy}")

    # The wedge query: one glob -> filter to failures -> mine them. (Daft.)
    df = daft.read_parquet(f"{outdir}/*.parquet")
    fail = df.where(df["success"] == False).sort(["episode_id", "step_idx"])
    d = fail.to_pydict()

    ep_rows: OrderedDict[str, list[int]] = OrderedDict()
    for i in range(len(d["episode_id"])):
        ep_rows.setdefault(d["episode_id"][i], []).append(i)

    results = []
    for eid, idxs in ep_rows.items():
        obj_z = [json.loads(d["object_poses"][i])[TARGET][2] for i in idxs]
        gc = [d["gripper_state"][i] < 0.04 for i in idxs]
        label, events, regrasps = detect_regrasp(obj_z, gc)
        results.append({"episode_id": eid, "policy": d["policy_type"][idxs[0]],
                        "label": label, "obj_z": obj_z, "gc": gc, "events": events})

    by_policy: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        by_policy[r["policy"]][r["label"]] += 1

    print(f"\n{len(results)} failure episodes mined with Daft:\n")
    for p in ("openvla", "vla_jepa"):
        tot = sum(by_policy[p].values())
        rg = by_policy[p]["re_grasp"]
        print(f"  {p:9s}  re-grasp rate {rg}/{tot} = {rg/tot:.0%}   mix={dict(by_policy[p])}")
    rate = {p: by_policy[p]["re_grasp"] / sum(by_policy[p].values()) for p in ("openvla", "vla_jepa")}
    ratio = rate["openvla"] / max(rate["vla_jepa"], 1e-9)
    print(f"\n  => OpenVLA's failures are {ratio:.0f}x more often slip-then-fumble re-grasps.\n")

    _plot(results, by_policy, rate, ratio)


# --------------------------------------------------------------- the hero screenshot

def _plot(results, by_policy, rate, ratio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hero = next(r for r in results if r["policy"] == "openvla" and r["label"] == "re_grasp")
    z, gc, events = hero["obj_z"], hero["gc"], hero["events"]
    color = {"grasp": "#2ca02c", "lift": "#1f77b4", "drop": "#d62728", "re-grasp": "#9467bd"}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})

    ax.plot(range(len(z)), z, lw=2.4, color="#1f77b4", label="object height")
    ax.axhline(0.0, ls="--", lw=1, color="gray", label="table")
    # shade gripper-closed spans
    in_c, start = False, 0
    for t, c in enumerate(gc + [False]):
        if c and not in_c:
            in_c, start = True, t
        elif not c and in_c:
            in_c = False
            ax.axvspan(start, t, color="#ffce6b", alpha=0.25, label="_grip closed")
    ax.axvspan(0, 0, color="#ffce6b", alpha=0.25, label="gripper closed")  # legend proxy
    for t, kind in events:
        ax.scatter([t], [z[t]], s=55, zorder=5, color=color[kind], edgecolor="white", lw=0.8)
        ax.annotate(kind, (t, z[t]), textcoords="offset points",
                    xytext=(0, -16 if kind == "drop" else 13), ha="center",
                    fontsize=9, fontweight="bold", color=color[kind])
    ax.set_title("Automatic re-grasp detection — OpenVLA, libero_spatial",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("object height (m)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.margins(x=0.02)

    pols = ["openvla", "vla_jepa"]
    ax2.bar(pols, [rate[p] for p in pols], color=["#d62728", "#2ca02c"], width=0.6)
    for i, p in enumerate(pols):
        ax2.text(i, rate[p] + 0.03, f"{rate[p]:.0%}", ha="center", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("share of failures that are re-grasp loops")
    ax2.set_title(f"OpenVLA re-grasps {ratio:.0f}× more often", fontweight="bold", fontsize=12)

    fig.suptitle("You can get a success rate; this tells you WHY it fails", fontsize=13, y=1.02)
    fig.text(0.5, -0.02, "synthetic demo data — proves detection + viz; real numbers from real rollouts",
             ha="center", fontsize=8, style="italic", color="gray")
    out = Path(__file__).parent / "regrasp_demo.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"  hero screenshot -> {out}")


if __name__ == "__main__":
    main()
