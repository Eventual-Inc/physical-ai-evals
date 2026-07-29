from __future__ import annotations

import daft
import pytest

from physical_ai_evals.core import hub


def test_hub_catalog_discovers_files_with_daft_and_checks_revision(monkeypatch):
    calls = {}
    revisions = iter(["abc123", "abc123"])
    monkeypatch.setattr(hub, "current_repo_revision", lambda _repo_id: next(revisions))

    def fake_glob(paths, *, io_config):
        calls["paths"] = paths
        calls["io_config"] = io_config
        return daft.from_pydict(
            {
                "path": [
                    "hf://datasets/org/data/bddl_files/second.bddl",
                    "hf://datasets/org/data/bddl_files/first.bddl",
                ],
                "size": [2, 1],
                "num_rows": [None, None],
            }
        )

    monkeypatch.setattr(daft, "from_glob_path", fake_glob)

    files = hub.glob_repo_files(
        "org/data", "abc123", ("bddl_files/**/*.bddl",), io_config="custom-io"
    )

    assert calls == {
        "paths": ["hf://datasets/org/data/bddl_files/**/*.bddl"],
        "io_config": "custom-io",
    }
    # the result is a lazy plan, not a materialized list
    assert isinstance(files, daft.DataFrame)
    assert files.column_names == ["path"]
    assert sorted(files.to_pydict()["path"]) == [
        "bddl_files/first.bddl",
        "bddl_files/second.bddl",
    ]


def test_hub_catalog_rejects_upstream_drift_before_globbing(monkeypatch):
    monkeypatch.setattr(hub, "current_repo_revision", lambda _repo_id: "new-head")

    def unexpected_glob(*_args, **_kwargs):
        pytest.fail("Daft glob should not run after revision drift")

    monkeypatch.setattr(daft, "from_glob_path", unexpected_glob)

    with pytest.raises(RuntimeError, match="moved from expected revision"):
        hub.glob_repo_files("org/data", "recorded-head", ("**/*.bddl",))


def test_hub_uri_pins_a_path_column_to_an_immutable_revision():
    frame = daft.from_pydict({"path": ["bddl_files/first.bddl"]})
    pinned = frame.select(hub.hf_uri("org/data", "abc123", daft.col("path")).alias("uri"))

    assert pinned.to_pydict()["uri"] == [
        "hf://datasets/org/data@abc123/bddl_files/first.bddl"
    ]
