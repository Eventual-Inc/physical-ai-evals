"""Authenticated Hugging Face access for Daft reads."""

from __future__ import annotations

import os


def hf_io_config(io_config=None):
    """Return ``io_config``, or one carrying ``HF_TOKEN`` when it is set.

    Daft does not read ``HF_TOKEN`` itself, and anonymous Hub listing is
    rate-limited (HTTP 429) after a few catalog queries.
    """
    token = os.environ.get("HF_TOKEN")
    if io_config is not None or not token:
        return io_config
    from daft.io import HuggingFaceConfig, IOConfig

    return IOConfig(hf=HuggingFaceConfig(token=token))
