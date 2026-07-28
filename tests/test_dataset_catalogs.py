from __future__ import annotations

import daft
import pytest

from physical_ai_evals.datasets import _lerobot, abc, aloha, egodex, libero_para, libero_pro


def test_libero_para_catalog_is_pinned_and_separates_environment(monkeypatch):
    monkeypatch.setattr(
        libero_para,
        "list_repo_files",
        lambda _repo_id, _revision: [
            "README.md",
            "bddl_files/act_lexical_addition_deletion_eval3_ver7.bddl",
            "bddl_files/obj_lexical_same_polarity_habitual_eval8_ver2.bddl",
        ],
    )

    data = libero_para.raw("org/para", "abc123").to_pydict()

    assert data["suite"] == ["libero_para", "libero_para"]
    assert data["environment_suite"] == ["libero_goal", "libero_goal"]
    assert data["environment_task_id"] == [3, 8]
    assert data["paraphrase_type"] == ["act", "obj"]
    assert data["variant_id"] == [7, 2]
    assert data["dataset_revision"] == ["abc123", "abc123"]
    assert data["bddl_path"][0].startswith("hf://datasets/org/para@abc123/")


def test_libero_pro_catalog_pairs_published_init_files(monkeypatch):
    files = [
        "bddl_files/01_visual_noise_glare/bddl/libero_goal/turn_on_the_stove.bddl",
        "bddl_files/libero_spatial_lan/pick_up_the_bowl.bddl",
        "init_files/libero_goal/turn_on_the_stove.pruned_init",
        "init_files/libero_spatial_lan/pick_up_the_bowl.pruned_init",
    ]
    monkeypatch.setattr(
        libero_pro,
        "list_repo_files",
        lambda _repo_id, _revision: files,
    )

    data = libero_pro.raw("org/pro", "def456").to_pydict()

    assert data["suite"] == ["libero_goal", "libero_spatial"]
    assert data["suite_variant"] == [
        "01_visual_noise_glare",
        "libero_spatial_lan",
    ]
    assert data["perturbation"] == ["01_visual_noise_glare", "lan"]
    assert data["init_path"] == [
        "hf://datasets/org/pro@def456/init_files/libero_goal/"
        "turn_on_the_stove.pruned_init",
        "hf://datasets/org/pro@def456/init_files/libero_spatial_lan/"
        "pick_up_the_bowl.pruned_init",
    ]


def test_instruction_payload_is_loaded_after_catalog_selection(tmp_path):
    first = tmp_path / "first.bddl"
    second = tmp_path / "second.bddl"
    first.write_text("(:language put the bowl on the plate)\n", encoding="utf-8")
    second.write_text("(:language turn on the stove)\n", encoding="utf-8")
    catalog = daft.from_pydict(
        {
            "task_name": ["first", "second"],
            "bddl_path": [str(first), str(second)],
        }
    )

    selected = catalog.where(daft.col("task_name") == "second")
    data = libero_para.instructions(selected).to_pydict()

    assert data["task_name"] == ["second"]
    assert data["instruction"] == ["turn on the stove"]


def test_revision_check_rejects_upstream_drift(monkeypatch):
    monkeypatch.setattr(_lerobot, "current_repo_revision", lambda _repo_id: "new-head")

    with pytest.raises(RuntimeError, match="moved from expected revision"):
        _lerobot.verified_lerobot_uri("org/data", "recorded-head")


def test_aloha_raw_uses_revision_checked_native_reader(monkeypatch):
    from daft.datasets import lerobot

    calls = {}
    monkeypatch.setattr(
        aloha,
        "verified_lerobot_uri",
        lambda repo_id, revision: f"verified://{repo_id}@{revision}",
    )

    def fake_read(uri, **kwargs):
        calls["uri"] = uri
        calls["kwargs"] = kwargs
        return "aloha-frame-plan"

    monkeypatch.setattr(lerobot, "read", fake_read)

    result = aloha.raw(load_video_frames="observation.images.cam_high")

    assert result == "aloha-frame-plan"
    assert calls["uri"].endswith(f"@{aloha.DEFAULT_REVISION}")
    assert calls["kwargs"]["load_video_frames"] == "observation.images.cam_high"


def test_abc_episode_reader_can_target_smoke_dataset(monkeypatch):
    from daft.datasets import lerobot

    calls = {}
    monkeypatch.setattr(
        abc,
        "verified_lerobot_uri",
        lambda repo_id, revision: f"verified://{repo_id}@{revision}",
    )

    def fake_read_episodes(uri, **kwargs):
        calls["uri"] = uri
        calls["kwargs"] = kwargs
        return "abc-episode-plan"

    monkeypatch.setattr(lerobot, "read_episodes", fake_read_episodes)

    result = abc.episodes(
        repo_id=abc.SMOKE_REPO_ID,
        revision=abc.SMOKE_REVISION,
        include_stats=True,
    )

    assert result == "abc-episode-plan"
    assert abc.SMOKE_REPO_ID in calls["uri"]
    assert calls["kwargs"]["include_stats"] is True


def test_egodex_catalog_lists_nested_activity_datasets(monkeypatch):
    monkeypatch.setattr(
        egodex,
        "list_repo_files",
        lambda _repo_id, _revision: [
            "README.md",
            "test/add_remove_lid/meta/info.json",
            "test/add_remove_lid/meta/tasks.parquet",
            "train/write/meta/info.json",
        ],
    )

    data = egodex.catalog("org/egodex", "abc123", split="test").to_pydict()

    assert data == {
        "dataset": ["egodex"],
        "dataset_revision": ["abc123"],
        "split": ["test"],
        "task_name": ["add_remove_lid"],
        "dataset_uri": ["hf://datasets/org/egodex/test/add_remove_lid"],
    }


def test_egodex_reader_uses_selected_activity(monkeypatch):
    from daft.datasets import lerobot

    calls = {}
    monkeypatch.setattr(
        egodex,
        "verified_lerobot_uri",
        lambda repo_id, revision, subpath: f"verified://{repo_id}@{revision}/{subpath}",
    )

    def fake_read_episodes(uri, **kwargs):
        calls["uri"] = uri
        return "egodex-episode-plan"

    monkeypatch.setattr(lerobot, "read_episodes", fake_read_episodes)

    result = egodex.episodes("add_remove_lid", split="test")

    assert result == "egodex-episode-plan"
    assert calls["uri"].endswith("/test/add_remove_lid")
