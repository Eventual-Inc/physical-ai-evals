"""CPU-only tests for standard, Para, and Pro benchmark specifications."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import daft
import numpy as np
import torch

from physical_ai_evals import libero, libero_para, libero_pro
from physical_ai_evals.libero import LiberoRuntime

benchmarks = importlib.import_module("physical_ai_evals.libero")


def _listing(paths: list[str]):
    return lambda *_args, **_kwargs: daft.from_pydict({"path": paths})


def test_standard_libero_builds_lazy_episode_grid_without_importing_simulator():
    benchmark = libero(
        "libero_spatial",
        task_ids=(1, 3),
        episodes=2,
        seed=11,
    )

    assert isinstance(benchmark.specs, daft.DataFrame)
    data = benchmark.specs.sort(["task_id", "init_state_id"]).to_pydict()
    assert data["task_id"] == [1, 1, 3, 3]
    assert data["init_state_id"] == [0, 1, 0, 1]
    assert data["seed"] == [11] * 4
    assert data["max_steps"] == [250] * 4
    assert len(set(data["episode_key"])) == 4
    assert all(key.startswith("e") for key in data["episode_key"])


def test_standard_runtime_loads_trusted_init_states_across_torch_versions(
    tmp_path,
    monkeypatch,
):
    init_root = tmp_path / "init"
    init_path = init_root / "libero_spatial" / "task.pruned_init"
    init_path.parent.mkdir(parents=True)
    expected = np.arange(4, dtype=np.float32)
    torch.save([expected], init_path)

    task = SimpleNamespace(
        problem_folder="libero_spatial",
        bddl_file="task.bddl",
        init_states_file="task.pruned_init",
    )
    suite_object = SimpleNamespace(
        get_task=lambda _task_id: task,
        get_task_init_states=lambda _task_id: (_ for _ in ()).throw(
            AssertionError("version-sensitive upstream loader must not be used")
        ),
    )
    benchmark_module = SimpleNamespace(
        get_benchmark_dict=lambda: {"libero_spatial": lambda: suite_object}
    )
    libero_package = ModuleType("libero")
    libero_module = ModuleType("libero.libero")
    libero_module.benchmark = benchmark_module
    libero_module.get_libero_path = lambda key: (
        init_root if key == "init_states" else tmp_path / "bddl"
    )
    libero_package.libero = libero_module
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)

    runtime = LiberoRuntime()
    monkeypatch.setattr(
        runtime,
        "_replace_environment",
        lambda *_args, **_kwargs: setattr(runtime, "_environment", "environment"),
    )
    environment, loaded_task, states = runtime._standard(
        "libero_spatial",
        0,
        7,
    )

    assert environment == "environment"
    assert loaded_task is task
    np.testing.assert_array_equal(states[0], expected)


def test_para_catalog_and_specs_keep_goal_environment_separate(tmp_path, monkeypatch):
    bddl = tmp_path / "act_lexical_eval3_ver2.bddl"
    bddl.write_text("(:language place the bowl beside the plate)\n", encoding="utf-8")
    tasks = daft.from_pydict(
        {
            "benchmark": ["libero_para"],
            "benchmark_revision": ["para-revision"],
            "suite": ["libero_goal"],
            "task_id": [3],
            "task_key": ["act_lexical_eval3_ver2"],
            "task_name": ["act_lexical_eval3_ver2"],
            "perturbation": ["act"],
            "paraphrase_key": ["lexical"],
            "variant_id": [2],
            "bddl_path": [str(bddl)],
        }
    )

    benchmark = libero_para(tasks=tasks, episodes=2)
    data = benchmark.specs.sort("init_state_id").to_pydict()

    assert data["benchmark"] == ["libero_para", "libero_para"]
    assert data["suite"] == ["libero_goal", "libero_goal"]
    assert data["suite_variant"] == ["libero_para", "libero_para"]
    assert data["task_id"] == [3, 3]
    assert data["instruction"] == ["place the bowl beside the plate"] * 2
    assert data["init_path"] == [None, None]

    monkeypatch.setattr(
        benchmarks,
        "_glob_repo_files",
        _listing(
            [
                "bddl_files/act_lexical_addition_eval3_ver7.bddl",
                "bddl_files/obj_same_polarity_eval8_ver2.bddl",
            ]
        ),
    )
    catalog = benchmarks.libero_para_tasks("org/para", "abc123").sort("task_id").to_pydict()
    assert catalog["task_id"] == [3, 8]
    assert catalog["perturbation"] == ["act", "obj"]
    assert catalog["variant_id"] == [7, 2]
    assert all(path.startswith("hf://datasets/org/para@abc123/") for path in catalog["bddl_path"])


def test_pro_catalog_pairs_variant_then_suite_initial_states(monkeypatch):
    monkeypatch.setattr(
        benchmarks,
        "_glob_repo_files",
        _listing(
            [
                "bddl_files/01_visual_noise_glare/bddl/libero_goal/turn_on_the_stove.bddl",
                "bddl_files/libero_spatial_lan/pick_up_the_bowl.bddl",
                "init_files/libero_goal/turn_on_the_stove.pruned_init",
                "init_files/libero_spatial_lan/pick_up_the_bowl.pruned_init",
            ]
        ),
    )

    data = benchmarks.libero_pro_tasks("org/pro", "def456").sort("bddl_path").to_pydict()

    assert data["suite"] == ["libero_goal", "libero_spatial"]
    assert data["suite_variant"] == ["01_visual_noise_glare", "libero_spatial_lan"]
    assert data["perturbation"] == ["01_visual_noise_glare", "lan"]
    assert data["init_path"] == [
        "hf://datasets/org/pro@def456/init_files/libero_goal/turn_on_the_stove.pruned_init",
        "hf://datasets/org/pro@def456/init_files/libero_spatial_lan/pick_up_the_bowl.pruned_init",
    ]


def test_pro_specs_use_published_files(tmp_path):
    bddl = tmp_path / "task.bddl"
    init = tmp_path / "task.pruned_init"
    bddl.write_text("(:language put the cup on the plate)\n", encoding="utf-8")
    init.write_bytes(b"fixture")
    tasks = daft.from_pydict(
        {
            "benchmark": ["libero_pro"],
            "benchmark_revision": ["pro-revision"],
            "suite": ["libero_spatial"],
            "suite_variant": ["libero_spatial_lan"],
            "perturbation": ["lan"],
            "task_name": ["put_the_cup_on_the_plate"],
            "task_key": ["libero_spatial_lan:put_the_cup_on_the_plate"],
            "bddl_path": [str(bddl)],
            "init_path": [str(init)],
        }
    )

    benchmark = libero_pro(
        "libero_spatial",
        tasks=tasks,
        episodes=3,
        max_steps=17,
    )
    data = benchmark.specs.sort("init_state_id").to_pydict()

    assert data["benchmark"] == ["libero_pro"] * 3
    assert data["task_id"] == [None] * 3
    assert data["init_state_id"] == [0, 1, 2]
    assert data["max_steps"] == [17] * 3
    assert data["bddl_path"] == [str(bddl)] * 3
    assert data["init_path"] == [str(init)] * 3
