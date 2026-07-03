"""vla_jepa_harness — reproducible VLA rollouts -> parquet for failure-mode mining.

The wedge: a success rate is commodity; this turns thousands of rollouts into a
queryable Daft DataFrame so you can answer *why* a VLA fails (failure-mode clustering),
not just how often. Deliverable 1 of 3 (harness / notebook / blog).

Public surface kept import-light on purpose: importing ``harness`` must NOT drag in
torch / robosuite / lerobot / tensorflow / daft. Those live behind lazy imports in the
policy, runner, and ingest adapters. Only the schema/config/contracts are re-exported.
"""

from __future__ import annotations

from harness.config import EmbedConfig, IngestConfig, RolloutConfig
from harness.schema import (
    ACTION_DIM,
    COLUMNS,
    EMBEDDING_DIM,
    ROLLOUT_SCHEMA,
    SCHEMA_VERSION,
    STATE_DIM,
    TERMINAL_FAILURE_LABELS,
    empty_step_row,
    rollout_schema,
    validate_rows,
)

__version__ = "0.1.0"

__all__ = [
    "ROLLOUT_SCHEMA",
    "SCHEMA_VERSION",
    "COLUMNS",
    "ACTION_DIM",
    "STATE_DIM",
    "EMBEDDING_DIM",
    "TERMINAL_FAILURE_LABELS",
    "rollout_schema",
    "empty_step_row",
    "validate_rows",
    "RolloutConfig",
    "IngestConfig",
    "EmbedConfig",
    "__version__",
]
