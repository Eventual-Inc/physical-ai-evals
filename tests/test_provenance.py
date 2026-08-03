"""Evaluation fingerprints and atomic manifest behavior."""

from __future__ import annotations

import json

import pytest

import physical_ai_evals.provenance as provenance


def test_evaluation_manifest_is_canonical_and_config_sensitive():
    first_id, first = provenance.evaluation_manifest(
        policy={"id": "policy", "revision": "abc"},
        benchmark={"name": "mock", "revision": "def"},
        rollouts_sha256="rollouts",
    )
    second_id, second = provenance.evaluation_manifest(
        policy={"revision": "abc", "id": "policy"},
        benchmark={"revision": "def", "name": "mock"},
        rollouts_sha256="rollouts",
    )
    changed_id, _ = provenance.evaluation_manifest(
        policy={"id": "policy", "revision": "changed"},
        benchmark={"name": "mock", "revision": "def"},
        rollouts_sha256="rollouts",
    )

    assert first_id == second_id
    assert first == second
    assert first_id != changed_id


def test_manifest_is_atomic_and_rejects_conflicts(tmp_path):
    config = {"schema_version": "test", "policy": {"id": "p"}}
    provenance.write_manifest(tmp_path, "evaluation", config)

    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload == {"evaluation_id": "evaluation", "config": config}
    provenance.write_manifest(tmp_path, "evaluation", config)

    with pytest.raises(RuntimeError, match="conflict"):
        provenance.write_manifest(tmp_path, "other", config)
    assert list(tmp_path.glob(".manifest.*.json")) == []


def test_implementation_fingerprint_fails_closed_on_moved_source(tmp_path, monkeypatch):
    fake_module = tmp_path / "provenance.py"
    fake_module.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(fake_module))

    with pytest.raises(RuntimeError, match="source is missing"):
        provenance.implementation_fingerprint()
