# notebooks/ — comparative failure-mode notebook (OUTLINE ONLY)

> This is **deliverable 2 of 3** and a separate third of the month's budget. It is **not
> built in this scaffold** — this file is the outline so the harness skeleton knows what
> shape of parquet the notebook will consume. Do not implement the notebook here.

**Planned file:** `notebooks/failure_modes.ipynb` (one notebook; keep it runnable
top-to-bottom on a researcher's own `data/rollouts/`).

## The story the notebook tells (15 minutes of a researcher's attention)

"You ran two VLAs on LIBERO. One scores higher. **But _why_ does the loser fail?**
Here's how Daft turns 800 rollouts into a clustered map of failure modes — and surfaces a
re-grasp loop neither success rate nor a glance at a video would have told you about."

The headliner is **VLA-JEPA vs OpenVLA**; the hero screenshot is **automatic re-grasp
detection**.

## Outline

1. **Load** — `df = daft.read_parquet("data/rollouts/*.parquet")`. One row per step;
   `model`/`policy_type` distinguish the two policies. Show row count + the two success
   rates side by side (the commodity number — establish it, then move past it).
2. **Isolate the failures** — `df.where(df["success"] == False)`. This is the wedge: we
   only mine the failures. Quick `groupby("policy_type").count()` to frame the gap.
3. **Embed** — one representation per failed episode for clustering. Two paths (config’d
   via `EmbedConfig`):
   - text: `embed_text(col("instruction"))` (fast, CPU-friendly default for CI),
   - image: a `@daft.cls` encoder over `frame_path` at the failure step (richer).
   Materialize: `pdf = df.select("episode_id","embedding").to_pandas()`.
4. **Cluster** — Daft has no built-in KMeans: `X = np.stack(pdf["embedding"])` →
   `sklearn.cluster.KMeans/HDBSCAN` → attach `cluster_id` back via `daft.from_pydict` +
   join. Write `clusters/` partitioned by `cluster_id`.
5. **Name the clusters** — per cluster, show 3–4 representative rollout videos
   (`video_path`) and the modal `terminal_failure`. This is the "wrong-object pile vs the
   drop pile vs the re-grasp pile" reveal.
6. **HERO MOMENT — re-grasp detection.** Without watching a single video, query the
   per-step signal that the schema preserves precisely for this:
   - gripper closes (`gripper_state` crosses to closed) →
   - manipuland lifts (`object_poses[target].z` rises above table) →
   - manipuland falls back / gripper-object distance jumps →
   - gripper reopens then recloses.
   Emit a per-episode event timeline `(grasp, lift, slip/drop, regrasp_attempt)`, label
   those episodes `terminal_failure="re_grasp"`, and screenshot the count + one annotated
   trajectory. **This is the blog/social image.**
7. **Comparison payoff** — re-grasp rate VLA-JEPA vs OpenVLA. The headline isn't "X% vs
   Y% success"; it's "OpenVLA's failures are N× more often slip-then-fumble re-grasps" —
   an actionable _why_ a success rate can't give you.

## Contract the notebook relies on (from `harness/schema.py`)

- One row per step; episode fields denormalized → no joins to filter failures.
- `gripper_state`, `gripper_action`, `eef_pos`, `object_poses` (JSON) present at every
  step → re-grasp detection is a pure query, no re-simulation.
- `frame_path` / `video_path` are paths (decoded lazily by Daft), not inline bytes.
- `embedding` is a portable `list<float32>`; `.cast()` to `embedding(float32, DIM)` before
  `cosine_distance`. `DIM` = `harness.schema.EMBEDDING_DIM` (shared with the harness).
