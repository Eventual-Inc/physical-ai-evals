from __future__ import annotations

import daft

from physical_ai_evals.datasets import libero_para, libero_pro


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
