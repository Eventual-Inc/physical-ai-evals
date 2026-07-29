"""BDDL task-definition parsing shared by the LIBERO benchmark catalogs."""

from __future__ import annotations

from daft import DataFrame, col
from daft.datatype import DataType
from daft.functions import regexp_extract


def add_instructions(tasks: DataFrame, *, io_config=None) -> DataFrame:
    """Add BDDL language instructions, reading only rows retained by the query plan."""
    content = col("bddl_path").download(io_config=io_config).cast(DataType.string)
    return tasks.with_column(
        "instruction", regexp_extract(content, r"\(:language\s+([^\r\n)]+)", 1)
    )
