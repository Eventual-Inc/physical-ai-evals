from __future__ import annotations

import daft

from physical_ai_evals.bench.libero import libero_para, libero_pro


def _listing(paths: list[str]):
    """Stand in for glob_repo_files, which returns a lazy one-column plan."""
    return lambda _repo_id, _revision, _patterns, **_kwargs: daft.from_pydict({"path": paths})


def test_libero_para_catalog_is_pinned_and_separates_environment(monkeypatch):
    monkeypatch.setattr(
        libero_para,
        "glob_repo_files",
        _listing([
            "bddl_files/act_lexical_addition_deletion_eval3_ver7.bddl",
            "bddl_files/obj_lexical_same_polarity_habitual_eval8_ver2.bddl",
        ]),
    )

    data = libero_para.raw("org/para", "abc123").sort("bddl_path").to_pydict()

    assert data["suite"] == ["libero_para", "libero_para"]
    assert data["environment_suite"] == ["libero_goal", "libero_goal"]
    assert data["environment_task_id"] == [3, 8]
    assert data["paraphrase_type"] == ["act", "obj"]
    assert data["paraphrase_key"] == [
        "lexical_addition_deletion",
        "lexical_same_polarity_habitual",
    ]
    assert data["variant_id"] == [7, 2]
    assert data["dataset_revision"] == ["abc123", "abc123"]
    assert data["bddl_path"][0].startswith("hf://datasets/org/para@abc123/")


def test_libero_para_catalog_stays_lazy_until_collected(monkeypatch):
    monkeypatch.setattr(
        libero_para,
        "glob_repo_files",
        _listing([f"bddl_files/act_key_eval{i}_ver0.bddl" for i in range(50)]),
    )

    catalog = libero_para.raw("org/para", "abc123")

    assert isinstance(catalog, daft.DataFrame)
    selected = catalog.where(daft.col("environment_task_id") == 7).to_pydict()
    assert selected["environment_task_id"] == [7]


def test_libero_pro_catalog_pairs_published_init_files(monkeypatch):
    monkeypatch.setattr(
        libero_pro,
        "glob_repo_files",
        _listing([
            "bddl_files/01_visual_noise_glare/bddl/libero_goal/turn_on_the_stove.bddl",
            "bddl_files/libero_spatial_lan/pick_up_the_bowl.bddl",
            "init_files/libero_goal/turn_on_the_stove.pruned_init",
            "init_files/libero_spatial_lan/pick_up_the_bowl.pruned_init",
        ]),
    )

    data = libero_pro.raw("org/pro", "def456").sort("bddl_path").to_pydict()

    assert data["suite"] == ["libero_goal", "libero_spatial"]
    assert data["suite_variant"] == ["01_visual_noise_glare", "libero_spatial_lan"]
    assert data["perturbation"] == ["01_visual_noise_glare", "lan"]
    assert data["init_path"] == [
        # the configured variant has no init_files/01_visual_noise_glare/ entry,
        # so it falls back to the base suite
        "hf://datasets/org/pro@def456/init_files/libero_goal/turn_on_the_stove.pruned_init",
        "hf://datasets/org/pro@def456/init_files/libero_spatial_lan/pick_up_the_bowl.pruned_init",
    ]


def test_libero_pro_init_path_is_null_when_no_init_file_is_published(monkeypatch):
    monkeypatch.setattr(
        libero_pro,
        "glob_repo_files",
        _listing(["bddl_files/libero_goal_swap/unpublished_task.bddl"]),
    )

    data = libero_pro.raw("org/pro", "def456").to_pydict()

    assert data["task_name"] == ["unpublished_task"]
    assert data["init_path"] == [None]


def test_instruction_payload_is_loaded_after_catalog_selection(tmp_path):
    first = tmp_path / "first.bddl"
    second = tmp_path / "second.bddl"
    first.write_text("(:language put the bowl on the plate)\n", encoding="utf-8")
    second.write_text("(:language turn on the stove)\n", encoding="utf-8")
    catalog = daft.from_pydict(
        {"task_name": ["first", "second"], "bddl_path": [str(first), str(second)]}
    )

    selected = catalog.where(daft.col("task_name") == "second")
    data = libero_para.instructions(selected).to_pydict()

    assert data["task_name"] == ["second"]
    assert data["instruction"] == ["turn on the stove"]
