"""Core contract: schema, episode types, writer, config."""

from physical_ai_evals.core.config import (
    CORE_SUITES,
    SUITE_MAX_STEPS,
    EmbedConfig,
    IngestConfig,
    RolloutConfig,
)
from physical_ai_evals.core.episode import PRIMARY, WRIST, Episode, Step
from physical_ai_evals.core.schema import (
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
from physical_ai_evals.core.writer import (
    RolloutWriter,
    assert_emits_schema,
    write_episode,
    write_rows,
)

__all__ = [
    "ACTION_DIM",
    "COLUMNS",
    "CORE_SUITES",
    "EMBEDDING_DIM",
    "Episode",
    "IngestConfig",
    "EmbedConfig",
    "PRIMARY",
    "ROLLOUT_SCHEMA",
    "RolloutConfig",
    "RolloutWriter",
    "SCHEMA_VERSION",
    "STATE_DIM",
    "Step",
    "SUITE_MAX_STEPS",
    "TERMINAL_FAILURE_LABELS",
    "WRIST",
    "assert_emits_schema",
    "empty_step_row",
    "rollout_schema",
    "validate_rows",
    "write_episode",
    "write_rows",
]
