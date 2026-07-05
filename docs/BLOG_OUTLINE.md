# Blog outline — "Our VLA scored 0% and the model was innocent"

Deliverable 3 of the contract (blog + social derivatives, carrying the reproducibility story
from NOTES.md). Audience per the champion research: the LIBERO/starVLA/openvla/vla-eval
cluster — researchers whose improvement loop is bottlenecked on eval pain. Voice: practitioner
depth, receipts over claims. **Daft is the engine that appears in the results; the eval pain is
the pitch.** Everything cited below exists in the repo (NOTES.md · docs/EVAL_PATTERNS.md ·
notebooks/failure_modes.py · git history).

**Working titles** (pick at draft time):
1. *Our VLA scored 0% and the model was innocent* ← lead candidate
2. *You can get a success rate. You can't get the why.*
3. *Reproducing a VLA eval is broken — here are the receipts*

---

## 1. Cold open — the 0% that wasn't (~300 words)

OpenVLA's published LIBERO-Spatial number: **84.7%**. Our first sweep: **0/7, every episode
dying at the 250-step cap**. No exception, no warning — plausible-looking robot flailing.

The usual next step is an afternoon of scrubbing rollout videos. Instead, one query over the
per-step parquet the harness writes:

```
commanded gripper over one episode: min=0.000  max=0.996
LIBERO expects -1 (open) .. +1 (close)
```

**The hand could never open.** `predict_action` returns the RLDS gripper convention
(0..1, ~1=open); LIBERO wants −1..+1 with the opposite polarity. The model was innocent; the
harness was lying. One fix later (normalize → binarize → invert, straight from OpenVLA's own
eval utils): **10/10 on task 0, episodes finishing in ~80 steps.** [FIGURE: the two-line query +
before/after table. Receipt: NOTES.md gotcha; commits `df6c8ee` → validation.]

Beat to land: *a silent harness bug is indistinguishable from a bad model unless your eval
records enough to interrogate.*

## 2. This is everyone's Tuesday (~250 words)

Zoom out: this failure mode is the VLA ecosystem's default experience.

- starVLA #355: 0.00 SR across 24 tasks — official checkpoint + official README, one missing
  `unnorm_key`.
- A ShanghaiTech researcher reporting a mere HF processor *warning* makes "success rate drop
  dramatically."
- LIBERO #141: the bowl **teleports** at eval init — tasks failing for non-policy reasons.
- openpi's docs and starVLA's client both ship loud TRAIN/TEST-CONSISTENCY banners because
  eval-time preprocessing must *byte-match* training — and mismatches surface only as SR.

The structural claim (from docs/EVAL_PATTERNS.md): every VLA eval is the same nine components
— checkpoint, norm stats, policy wrapper, serving split, env construction, chunk handling,
success criterion, trials×seeds, recording — and **the failure mode of the whole genre is that
component #2/#5 errors masquerade as model quality.** One number out, no forensics.

## 3. What we built: the eval that can explain itself (~400 words)

The two-part thesis, matching the repo's acts:

**Reproduce.** VLA-JEPA (the lerobot port + `lerobot/VLA-JEPA-LIBERO`) and OpenVLA, on LIBERO,
on your own GPU, in ONE process each — no policy server, no WebSocket, no bespoke eval repo.
Canonical protocol constants (50 trials/seed 7 — the OpenVLA-origin numbers openpi/starVLA/
vla-eval all inherit). The stack is deliberately opinionated Python: Daft data plane, Modal GPU
placement, PyTorch in-process, MuJoCo/robosuite (`hf-libero` wheel — no git clone). Contrast
honestly with allenai/vla-eval: they buy tech-agnostic generality with a mandatory network
boundary + Docker per benchmark; we buy reproducibility with one process you can read.
[Sidebar: why the ecosystem's WebSocket exists (real robots, env conflicts, batched serving)
and why none of those bind a sim eval on Modal — `@daft.cls` IS the model server.]

**Understand.** Every rollout streams to one-row-per-step parquet: commanded vs measured
gripper, EEF pose, per-step actions, success, video path. "SR dropped" then decomposes with
a DataFrame query into *policy* failures vs *harness* failures. That's the layer none of the
five frameworks record, and it's what §1's diagnosis fell out of. Show the schema in ~10
lines; show the wedge query (`where(success == False)`).

## 4. The payoff — failure forensics on a real comparison (~400 words)

The notebook (notebooks/failure_modes.py) on the real sweep:
[NUMBERS IN — 2026-07-02 sweep, 200 episodes:]
- Success rates: **VLA-JEPA 99%** vs **OpenVLA 84%** on libero_spatial (10 trials × 10 tasks,
  seed 7 — an honest subset of the 50-trial canon, stated as such). Bonus beat: OpenVLA's 84%
  lands **within a point of its published 84.7%** — the reproduction claim, demonstrated,
  *after* fixing our own 0%.
- **The hero figure** (baked in the executed notebook): OpenVLA commanding **23 grasp attempts
  in one failed episode** ("pick up the black bowl next to the ramekin and place it on the
  plate") — fingers collapsing to ~2 mm "closed on air" over and over, eef sawtoothing.
  Caption: *watch a fumble loop without watching anything.*
- The failure-mix bar + the money line: **"OpenVLA fails 16× more often than VLA-JEPA (16% vs
  1%) — and 15 of its 16 failures are re-grasp fumble loops."** (VLA-JEPA's single failure is
  also a fumble loop.) The actionable *why* a scalar can't give you.
- Fresh §5 material from the sweep itself: the episode_id chimera (a 500-step phantom episode
  in a 250-cap suite — group by (policy_type, episode_id)) and the env.reset() outcome-flip
  (the same episode fails pre-fix, succeeds post-fix: contamination degrades before it crashes).
- The behavioral features are 5 lines of pandas over 3 schema columns — cheap on purpose;
  thresholds read off the data (holding >4 mm vs air <2 mm histogram).

## 5. Field guide: 19 landmines between you and a reproducible VLA eval (~500 words)

The condensed table now lives at `docs/FRICTION_POINTS.md` (symptom → fix, grouped by when it
bites); the chronological story is `docs/FRICTION_LOG.md`. This section IS the reproducibility
story the contract promised, and it's skimmable link-bait for practitioners:

- **Env & build:** LIBERO's "dependency conflict" is a myth (its setup.py installs nothing);
  CUDA runtime bases ship no compiler/cmake (evdev, egl_probe); LIBERO's `input()` prompt
  EOFErrors containers — twice (raw clone AND the hf-libero wheel); MUJOCO_GL before import.
- **Silent SR killers:** the RLDS↔LIBERO gripper convention (§1); missing center-crop when the
  checkpoint trained with crop aug; `unnorm_key` is the SUITE name for the fine-tunes;
  numpy 2 sneaking in via a dependency bump and breaking torch 2.2 ("_ARRAY_API not found");
  images must be float-[0,1] because a downstream processor sets `do_rescale=False`.
- **Sweep killers:** `env.reset()` every episode — `set_init_state` alone accumulates
  robosuite's step counter across episodes toward the horizon and poisons the env
  ("executing action in terminated episode"; invisible in short runs — OpenPI's loop does this
  and now we know why); the 180° de-rotation view vs `torch.from_numpy` (negative strides);
  `modal run --detach` survives network drops but not client teardown → deploy+spawn +
  resumable sweeps (part filenames = deterministic episode ids).

Close the section: *none of these raise; all of them read as "the model is bad."*

## 6. Close + CTA (~150 words)

- Repo + PR link; `modal run` three-liner; the notebook runs on YOUR parquet — the schema is
  six ingest adapters wide (DROID/LeRobot/HDF5/ALOHA/EgoDex/ABC), so "run it on your own
  dataset" is real. ← the contract's gold signal, stated as the explicit ask.
- What's next honestly: full 50-trial canon, more suites, π0 via openpi's Python API.
- Thanks/credits: lerobot's vla_jepa port + hf-libero (we're early external consumers),
  OpenVLA's eval utils, LIBERO.

---

## Social derivatives

**X thread (7 posts):** 1) OpenVLA published 84.7%; ours said 0/7. The model was innocent. 🧵
2) the two-line gripper query (screenshot) 3) the convention mismatch explained in one image
4) fix → 10/10 (before/after) 5) the re-grasp hero figure — "watch a fumble loop without
watching anything" 6) the failure-mix figure + "OpenVLA fails 16× more often — 15 of 16 failures are fumble loops" 7) repo + "bring your own parquet".

**Standalone image posts:** the re-grasp trace; the 16-landmines list as a card.

**LeRobot Discord:** "we ran the lerobot VLA-JEPA port + hf-libero end-to-end on Modal as maybe
its first external consumers — here's what we hit" (gotchas #10–12) + notebook link. Genuine
contribution tone, not promo.

**Named-researcher sends (the distribution third):** the ~25 from the champion brief
(CK1201, llong-cs, Wushr-Lance, haibao-yu, + the vla-eval starrers). One-line pitch: *"you
star eval harnesses — we made one that explains its failures; 15 min for feedback?"* Wushr-Lance
gets the §1 story specifically (his issue IS this post).

## Assets checklist

- [x] Hero: re-grasp trace PNG (baked: openvla@libero_spatial/1/0/7, 23 attempts)
- [x] Failure-mix comparison bar (baked in notebook)
- [ ] The 0/7 gripper-query terminal screenshot (recreate: `min/max gripper_action`)
- [ ] Before/after table (0/7 @250-cap → 10/10 @~80 steps)
- [ ] Nine-components grammar table (from EVAL_PATTERNS, simplified)
- [ ] 16-landmines card

---

# Post 2 — "The dependency hell was so famous, even the AI believed it"

The second post, born from a real event: an AI agent auditing this repo asserted the VLA
dependency-hell lore as fact ("the stacks can never live in a pip-installable package…
LIBERO needs Python 3.8"), then read the receipts — NOTES.md, the verified images, the
one-resolver-pass pip spec — and **retracted, in writing**. The retraction itself is the
content. Receipts strong enough to flip a skeptic that was trained on the lore.

**Working titles:**
1. *The dependency hell was so famous, even the AI believed it*
2. *You don't need three conda envs to eval a VLA*
3. *An AI read our repo and retracted the LIBERO myth*

**The arc (~900 words):**
1. **The lore** — "LIBERO needs Python 3.8; numpy 1.22.4; keep sim in its own conda env;
   VLA eval = three environments minimum." Quoted everywhere, including (verbatim) by an AI
   agent reviewing this repo. We believed it too — our own June NOTES carried a
   "you need FOUR environments" section.
2. **The receipt** — LIBERO's `setup.py` is `install_requires=[]`. The scary pins are its
   *training* requirements, never installed for rollouts. The verified images: sim + policy +
   Daft in ONE resolver pass — OpenVLA stack (torch 2.2 / transformers 4.40.1 / robosuite
   1.4.1 / mujoco 3.9.0 / numpy 1.26.4) and the even cleaner VLA-JEPA path
   (`lerobot[vla_jepa,libero] @ SHA` — the hf-libero wheel ships LIBERO's code + bddl +
   assets; no clone, no config patch, no PYTHONPATH, no policy server). **One Python (3.12)
   across the repo.**
3. **The retraction** — quote the agent's correction directly ("What I got wrong vs. the
   kernel that's true…"). The kernel: the only real fragmentation left is
   **per-policy-checkpoint** — frozen checkpoints freeze transformers pins (4.40.1 vs
   5.4–5.6), so *policies* need separate images from each other. It was never sim-vs-policy,
   never the Python version. Dependency hell is an artifact of checkpoint freezing, not of
   robotics.
4. **The moral** — two audiences read your repo now: researchers and their agents. Stale
   claims don't just confuse people, they get *asserted as fact* by the models reading on
   their behalf (our own stale pyproject comment — "dedicated 3.8 conda env" — survived two
   weeks past its falsification and did exactly that). Keep the receipts consistent;
   they're load-bearing.
5. **CTA** — the pip spec + `modal run` lines ARE the proof; run them.

**Assets:** the agent's correction (quoted, anonymized as "an AI agent reviewing the repo");
the before/after pyproject `libero` extra; the one-pip-layer image spec from
`modal_vla_jepa_app.py`; the NOTES "Environments — ONE Python, TWO images, and the 3.8 myth"
section.

**Relationship to Post 1:** Post 1 = the forensics story (0% → one query → 84.7%). Post 2 =
the environment-myth story. Independent reads, same repo, same thesis: receipts over lore.
Post 2 can run first (lighter, spicier hook) or second (Post 1's §5 tees it up).

**Channel notes:** the X version leads with the agent quote screenshot — "an AI reviewed our
repo and issued a correction" is the scroll-stopper. LeRobot Discord angle: hf-libero as the
unsung hero (the wheel that killed the conda env).
