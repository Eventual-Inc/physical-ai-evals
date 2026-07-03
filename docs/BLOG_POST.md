# Our VLA scored 0% and the model was innocent

*Reproducing a VLA eval end-to-end — OpenVLA vs VLA-JEPA on LIBERO, on rented GPUs — and
finding the bug with one DataFrame query instead of an afternoon of scrubbing video.*

<!-- FIGURES: hero trace + failure-mix bar are baked in notebooks/failure_modes.ipynb;
     the gripper-query terminal shot is reproducible from the parquet. -->

---

OpenVLA's published number on LIBERO-Spatial is 84.7%.

Our first sweep scored **0 for 7**. Every episode ran to the 250-step cap and died. No
exception, no warning. The robot moved like it meant it — approach, descend, close, lift,
nothing. Plausible-looking failure, two hundred and fifty steps at a time.

The usual next move is to pour a coffee and start scrubbing rollout videos. We didn't have to,
because our harness writes every rollout to parquet — one row per step, with the commanded
action and the measured gripper state on every row. So instead of watching robots fail in real
time, we asked the data:

```
commanded gripper over one episode: min=0.000  max=0.996
LIBERO expects -1 (open) .. +1 (close)
```

**The hand could never open.**

OpenVLA's `predict_action` returns actions in the RLDS convention: gripper in [0, 1], where ~1
means open. LIBERO's controller wants [-1, +1], where -1 means open. Fed raw, the command is
never negative — so the gripper physically cannot open — and the polarity is backwards on top.
The model was innocent. The harness was lying to it.

The fix is three lines, and it's not even novel — it's exactly what OpenVLA's own eval utils
do (normalize [0,1] → [-1,1], binarize, invert). After the fix: **10/10 on task 0**, episodes
finishing in ~80 steps instead of capping out at 250.

One query. That's the post, really. Everything below is the receipts.

## This is everyone's Tuesday

If you work on VLAs, some version of this has happened to you, because the entire genre of
VLA evaluation fails this way:

- There's a starVLA issue where someone gets **0.00 across 24 tasks** with the official
  checkpoint and the official README. The cause: one missing `unnorm_key`.
- A researcher reports that a mere HuggingFace processor *warning* makes their "success rate
  drop dramatically."
- There's a LIBERO issue where the target bowl **teleports** during env init. Tasks fail and
  the policy never had a chance.
- openpi and starVLA both ship loud TRAIN/TEST CONSISTENCY banners in their client code —
  because eval-time preprocessing has to byte-match training, and when it doesn't, nothing
  raises. Your success rate just quietly isn't real.

We mapped the eval stacks of openpi, starVLA, OpenVLA, LeRobot, and AllenAI's
vla-evaluation-harness against each other (the full grammar is in
[`docs/EVAL_PATTERNS.md`](EVAL_PATTERNS.md)). Every one of them decomposes into the same nine
components — checkpoint, normalization stats, policy wrapper, serving split, env construction,
chunk handling, success criterion, trials and seeds, recording. And the failure mode of the
whole genre is the same: **plumbing errors masquerade as model quality.** One scalar out, no
forensics.

You can't grep a success rate.

## So we made the eval explain itself

We built a harness with two jobs.

**Job one: reproduce.** Run VLA-JEPA and OpenVLA on LIBERO, on your own GPU, with the
canonical protocol (50 trials per task, seed 7 — the constants OpenVLA's eval established and
everyone else inherited). Each policy runs **in one Python process**: policy, simulator, and
data capture together. No policy server, no WebSocket, no cloning four repos. The ecosystem's
server-client split exists for real robots and dependency conflicts — neither applies to sim
eval in a container, so we deleted the socket. VLA-JEPA loads through its LeRobot port
(`lerobot/VLA-JEPA-LIBERO`), LIBERO comes from the `hf-libero` wheel, and the whole thing runs
as a [Daft](https://daft.ai) UDF on a Modal GPU: episode specs go in as DataFrame rows,
parquet comes out.

**Job two: understand.** Every rollout streams to a canonical schema — one row per step,
commanded action, measured gripper separation, end-effector pose, success, a path to the mp4.
That's the layer none of the five frameworks record, and it's the difference between "the
number dropped" and knowing *why*. When our 0% happened, the diagnosis was a filter and two
aggregations:

```python
df = daft.read_parquet("data/rollouts/*/*.parquet")
failures = df.where(df["success"] == False)
```

## What the comparison actually found

With the harness honest, we ran both policies on LIBERO-Spatial: 10 tasks × 10 trials each,
seed 7, 200 episodes total.

|             | success rate | failures |
|-------------|--------------|----------|
| VLA-JEPA    | **99/100**   | 1        |
| OpenVLA     | **84/100**   | 16       |

Two things about that table.

First: OpenVLA's 84% lands within a point of its published 84.7% — from our code, on rented
GPUs, an hour after that same pipeline was scoring zero. Reproduction isn't a vibe; it's a
number you can hit.

Second: the table is still just success rates, and success rates don't tell you what to fix.
The per-step data does. We labeled every failure from three columns — commanded gripper
cycles, measured finger separation, end-effector height — and the picture is one-sided:

**OpenVLA fails 16× more often than VLA-JEPA (16% vs 1%) — and 15 of its 16 failures are the
same failure: a re-grasp fumble loop.**

The hero episode makes it visceral. "Pick up the black bowl next to the ramekin and place it
on the plate": OpenVLA commands **23 grasp attempts** in one episode. The finger-separation
trace shows the whole story — fingers snap shut to ~2mm (that's the width of *air*, not a
bowl), open, reposition, snap shut on air again, twenty-some times until the clock runs out.
We never watched the video to learn this. The trace *is* the video, minus the waiting.

(VLA-JEPA's single failure across 100 episodes? Also a fumble loop. When strong policies fail,
they apparently fail the same way — a hypothesis we now get to test on more suites.)

That's the actionable difference between 99% and 84%: not "be better," but *this policy loses
objects on the regrasp and can't recover*. That's a data-curation target and a fine-tuning
target, not a shrug.

## The field guide: 18 landmines, so you don't step on them

We logged every failure we hit getting this running cleanly — because getting it running
cleanly was the point. The full log is
[`NOTES.md`](https://github.com/Eventual-Inc/VLA-JEPA/blob/main/NOTES.md); the highlights:

**Environment and build.** LIBERO's famous "dependency conflict" is a myth — its `setup.py`
installs nothing; the scary `requirements.txt` is for training, not rollouts. CUDA runtime
images ship no compiler and no cmake (ask `evdev` and `egl_probe` how that goes). LIBERO
prompts for *interactive input on first import* — it EOFErrors inside a container, and yes,
that survived into the packaged wheel.

**Silent success-rate killers.** The RLDS↔LIBERO gripper convention (the 0% above). Skipping
center-crop when the checkpoint trained with crop augmentation. `unnorm_key` for the LIBERO
fine-tunes is the *suite name*, not the dataset name the docs suggest. numpy 2 sneaking in
through a dependency bump and breaking torch 2.2. Images that must be float [0,1] because a
processor downstream sets `do_rescale=False` — feed it 0-255 and nothing raises, ever.

**Sweep killers.** `env.reset()` every episode: restoring init state alone leaves robosuite's
internal step counter running *across episodes*, and around cumulative step 1000 the env
poisons itself and every episode after — invisible in short runs, fatal in sweeps. (openpi's
loop resets every episode. Now we know why.) The 180° de-rotated image view breaks
`torch.from_numpy` — negative strides. And a bonus from the analysis side: `episode_id` names
the episode *spec*, so two policies produce the same id — group by policy too, or you'll
chimera two trajectories into one 500-step phantom in a 250-step suite. Ours got caught
because 500 > 250. Yours might not.

None of these raise. All of them read as "the model is bad."

## Run it on your data

Everything above is one repo: the harness, the executed notebook with the figures, the eval
grammar, and the full landmine log.

```bash
modal run harness/rollout/modal_app.py --policy-type openvla --suites libero_spatial --episodes 10
modal run harness/rollout/modal_vla_jepa_app.py --suites libero_spatial --episodes 10
# then: notebooks/failure_modes.ipynb over data/rollouts/
```

The notebook doesn't care where the parquet came from. The same schema ingests DROID, LeRobot,
robomimic HDF5, ALOHA, EgoDex, and ABC — so if you've got demonstrations or rollouts in any of
those, the failure forensics run on *your* data with a path change. If you try it, we genuinely
want to hear what breaks: the landmine log grows by contribution.

We're Eventual, we build [Daft](https://daft.ai), and we're building data tooling for physical
AI. If success rates are hiding your failures too — come find us.
