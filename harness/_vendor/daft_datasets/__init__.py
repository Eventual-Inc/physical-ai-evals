"""TEMPORARY vendored copies of unmerged Daft dataset readers — DELETE WHEN MERGED.

Source (Apache-2.0, github.com/Eventual-Inc/Daft), copied VERBATIM at the PR head so they
diff cleanly against upstream:
  * ``lerobot.py``  <- PR #7090 (slade/lerobot @ 0f84f25) — ``daft.datasets.lerobot``, LeRobot v3
  * ``droid.py``    <- PR #7089 (slade/droid   @ 18d28ea) — ``daft.datasets.droid``, raw DROID

Why this exists: the PRs are unmerged, so our pinned ``daft==0.7.15`` does not ship
``daft.datasets.{lerobot,droid}`` yet. Every *internal* Daft API they rely on (``video_file``,
``unnest``, ``from_glob_path``, ``try_deserialize``, ``lpad``, ``GCSConfig``, ...) IS present
in 0.7.15, so they import and run as-is. ``install()`` monkey-patches them onto
``daft.datasets`` so the harness adapters AND the notebook can call
``daft.datasets.lerobot.read(...)`` / ``daft.datasets.droid.raw(...)`` exactly as they will
once the PRs land.

REMOVAL: once a Daft release ships these, ``install()`` sees the real modules and becomes a
no-op. At that point delete this whole package and the two ``install()`` call sites
(``harness/ingest/lerobot.py``, ``harness/ingest/droid.py``).

Tracking: https://github.com/Eventual-Inc/Daft/pull/7090  ·  https://github.com/Eventual-Inc/Daft/pull/7089
"""

from __future__ import annotations

import sys


def install() -> list[str]:
    """Register vendored lerobot/droid under ``daft.datasets`` IF the installed Daft lacks them.

    Idempotent. Returns the names injected this call (empty once Daft ships its own, or on a
    second call). PREFERS the real modules: if ``daft.datasets.lerobot`` already exists (a
    future Daft release), it is left untouched.
    """
    import daft.datasets as _dd

    injected: list[str] = []
    if not hasattr(_dd, "lerobot"):
        from harness._vendor.daft_datasets import lerobot as _lerobot

        _dd.lerobot = _lerobot
        sys.modules.setdefault("daft.datasets.lerobot", _lerobot)
        injected.append("lerobot")
    if not hasattr(_dd, "droid"):
        from harness._vendor.daft_datasets import droid as _droid

        _dd.droid = _droid
        sys.modules.setdefault("daft.datasets.droid", _droid)
        injected.append("droid")
    return injected
