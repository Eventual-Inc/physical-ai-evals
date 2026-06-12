"""Canonical rollout parquet schema — the contract everything in the harness emits and reads.

This module is CONCRETE (not a stub). It is the single source of truth for the
on-disk layout that:

  * ``harness/writer.py`` writes (one row per *step*),
  * ``harness/ingest/*`` adapters normalize into (via ``Episode.to_step_rows``),
  * the comparative failure-mode notebook reads back with ``daft.read_parquet``.

Design principle: **one row per step**, not per episode. Failure-mode clustering
needs sub-episode granularity — you cluster on *what the gripper/eef were doing at
the moment of slip*, not on an episode-level summary. The re-grasp hero moment
(grasp -> lift -> slip/drop -> re-grasp) is only recoverable if every step carries
``gripper_state``, ``eef_pos`` and the manipuland ``object_poses``. Episode-level
fields (success, terminal_failure, instruction, ...) are denormalized onto every
step row so the notebook can ``filter(col('success') == False)`` without a join,
and so a single ``read_parquet`` glob is self-describing.

Storage choices that keep the file Daft-readable AND portable (see NOTES.md):
  * Only plain Arrow primitives / lists / strings — NO Daft-only extension types,
    so the fixture loads in pandas/pyarrow/duckdb too.
  * Frames are carried as *paths/URLs* (``frame_path``, ``video_path``), never raw
    image bytes inline. Daft decodes media lazily from the path column; inlining
    images would bloat the parquet and defeat lazy multimodal I/O.
  * The optional ``embedding`` column is a fixed-width ``list<float32>`` of
    ``EMBEDDING_DIM``. Daft casts ``fixed_size_list(float32, DIM)`` -> ``embedding``
    for ``cosine_distance``; we store the portable list form on disk and let the
    notebook ``.cast()`` it. Left null by the rollout writer; populated by the
    embedding pass.
"""

from __future__ import annotations

import pyarrow as pa

# --- shared constants (the writer, ingestors, tests, and notebook all import these) ---

#: 7-DoF end-effector delta action: (x, y, z, roll, pitch, yaw, gripper). See
#: rollout/policy.py for the action contract. Fixed across LIBERO / OpenVLA / VLA-JEPA.
ACTION_DIM = 7

#: Normalized proprio state width when mimicking LeRobot's 8-dim convention:
#: eef position (3) + eef orientation as axis-angle (3) + gripper qpos (2).
#: Stored as a variable ``list<float32>`` so DROID's wider state still fits without
#: schema churn; this constant is the *expected* LIBERO width for validation.
STATE_DIM = 8

#: Embedding width for the failure-mode clustering pass. Kept as a module constant so
#: the harness writer, the notebook, and CI fixtures agree on one number. The default
#: matches a small sentence-transformers / SigLIP-class encoder; override via config.
EMBEDDING_DIM = 1024

#: Schema version stamped into every row so the notebook can branch if the layout
#: evolves. Bump on any breaking column change.
SCHEMA_VERSION = "rollout-v1"

#: Canonical terminal-failure labels. ``None``/empty == success or not-yet-labeled.
#: ``re_grasp`` is the wedge's headline class (the notebook hero moment).
TERMINAL_FAILURE_LABELS = (
    "re_grasp",          # grasp -> slip/drop -> re-attempt loop (the hero moment)
    "no_grasp",          # never closed gripper on the object
    "drop_no_recover",   # grasped, dropped, never recovered
    "wrong_object",      # manipulated the wrong object (object/spatial suites)
    "missed_target",     # correct object, wrong placement / goal predicate unmet
    "timeout",           # ran out of steps with no clear single failure
    "collision",         # arm/object collision aborted the trajectory
    "unlabeled",         # captured but not yet triaged
)


def rollout_schema() -> pa.Schema:
    """Return the canonical one-row-per-step pyarrow schema.

    Column groups
    -------------
    Identity / reproducibility (denormalized per step):
        ``schema_version``  str   — SCHEMA_VERSION stamp.
        ``episode_id``      str   — deterministic id = f"{suite}/{task_id}/{init_state_id}/{seed}".
        ``run_id``          str   — groups episodes from one ``harness rollout`` invocation.
        ``model``           str   — policy id, e.g. 'openvla/openvla-7b-finetuned-libero-spatial'.
        ``policy_type``     str   — 'openvla' | 'vla_jepa' | ingest source.
        ``source``          str   — 'libero' (rollout) | 'lerobot' | 'droid' | 'hdf5'.
        ``suite``           str   — LIBERO suite key, e.g. 'libero_goal'.
        ``task_id``         int32 — task index within the suite.
        ``task_name``       str   — LIBERO task name.
        ``instruction``     str   — natural-language command fed to the policy.
        ``bddl_file``       str   — LIBERO BDDL path (nullable for ingest sources).
        ``init_state_id``   int32 — which of the 50 init layouts (nullable for ingest).
        ``seed``            int64 — env seed (nullable for ingest).
        ``control_mode``    str   — 'relative' | 'absolute' (action parameterization).

    Episode outcome (denormalized per step — enables join-free failure queries):
        ``success``           bool  — episode-level success (LIBERO check_success / derived).
        ``terminal_failure``  str   — one of TERMINAL_FAILURE_LABELS, null/'' if success.
        ``num_steps``         int32 — total steps in the episode.

    Per-step signal (the granularity failure clustering needs):
        ``step_idx``      int32          — 0-based step index within the episode.
        ``action``        list<float32>  — ACTION_DIM EEF delta (un-normalized robot units).
        ``reward``        float32        — sparse robosuite reward {0,1} (nullable).
        ``done``          bool           — env done flag at this step.
        ``state``         list<float32>  — proprio (LeRobot 8-dim convention, nullable).
        ``eef_pos``       list<float32>  — end-effector xyz (3), for slip/lift detection.
        ``gripper_state`` float32        — scalar gripper opening (qpos-derived), nullable.
        ``gripper_action``float32        — commanded gripper (action[-1]), nullable.
        ``object_poses``  str            — JSON {object_name: [x,y,z,qx,qy,qz,qw]} of tracked
                                           manipulands; JSON (not struct) because the object set
                                           varies per task — keeps the schema task-agnostic.

    Media references (paths/URLs, decoded lazily by Daft — never inline bytes):
        ``frame_path``    str  — per-step agentview RGB frame on disk (nullable).
        ``wrist_path``    str  — per-step wrist-cam RGB frame (nullable).
        ``video_path``    str  — per-episode mp4 (denormalized; nullable).

    Optional embedding (populated by the clustering pass, null at rollout time):
        ``embedding``     list<float32>  — fixed width EMBEDDING_DIM; cast to
                                           ``embedding(float32, DIM)`` in the notebook.
    """
    f32 = pa.float32()
    return pa.schema(
        [
            # --- identity / reproducibility ---
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("episode_id", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("policy_type", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("suite", pa.string(), nullable=True),
            pa.field("task_id", pa.int32(), nullable=True),
            pa.field("task_name", pa.string(), nullable=True),
            pa.field("instruction", pa.string(), nullable=False),
            pa.field("bddl_file", pa.string(), nullable=True),
            pa.field("init_state_id", pa.int32(), nullable=True),
            pa.field("seed", pa.int64(), nullable=True),
            pa.field("control_mode", pa.string(), nullable=True),
            # --- episode outcome (denormalized per step) ---
            pa.field("success", pa.bool_(), nullable=False),
            pa.field("terminal_failure", pa.string(), nullable=True),
            pa.field("num_steps", pa.int32(), nullable=False),
            # --- per-step signal ---
            pa.field("step_idx", pa.int32(), nullable=False),
            pa.field("action", pa.list_(f32), nullable=True),
            pa.field("reward", f32, nullable=True),
            pa.field("done", pa.bool_(), nullable=True),
            pa.field("state", pa.list_(f32), nullable=True),
            pa.field("eef_pos", pa.list_(f32), nullable=True),
            pa.field("gripper_state", f32, nullable=True),
            pa.field("gripper_action", f32, nullable=True),
            pa.field("object_poses", pa.string(), nullable=True),  # JSON
            # --- media references (paths/urls; decoded lazily) ---
            pa.field("frame_path", pa.string(), nullable=True),
            pa.field("wrist_path", pa.string(), nullable=True),
            pa.field("video_path", pa.string(), nullable=True),
            # --- optional embedding (null at rollout time) ---
            pa.field("embedding", pa.list_(f32), nullable=True),
        ],
        metadata={
            b"schema_version": SCHEMA_VERSION.encode(),
            b"action_dim": str(ACTION_DIM).encode(),
            b"state_dim": str(STATE_DIM).encode(),
            b"embedding_dim": str(EMBEDDING_DIM).encode(),
        },
    )


#: Materialized once at import — most callers want the schema object, not a fresh build.
ROLLOUT_SCHEMA: pa.Schema = rollout_schema()

#: Column order as a plain list (handy for building dicts/validating row keys).
COLUMNS: tuple[str, ...] = tuple(ROLLOUT_SCHEMA.names)


def empty_step_row() -> dict[str, object]:
    """Return a dict with every schema column set to ``None``.

    The intended idiom: spread this, then overwrite the columns you have, so callers
    can never silently drop a column or reorder fields.

        row = {**empty_step_row(), "episode_id": ep_id, "step_idx": i, ...}
    """
    return {name: None for name in COLUMNS}


def validate_rows(rows: list[dict[str, object]]) -> pa.Table:
    """Build a pyarrow Table from step-row dicts under ``ROLLOUT_SCHEMA``.

    Raises ``pa.lib.ArrowInvalid``/``ArrowTypeError`` if a row violates the schema —
    this is the cheap correctness gate the writer and tests both go through. Returns
    the validated table ready for ``pyarrow.parquet.write_table``.
    """
    return pa.Table.from_pylist(rows, schema=ROLLOUT_SCHEMA)
