"""CPU-only tests for standard, Para, and Pro benchmark specifications."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import daft
import numpy as np
import pytest
import torch
from daft.functions import format

from physical_ai_evals import libero, libero_para, libero_pro
from physical_ai_evals.libero import LiberoRuntime

benchmarks = importlib.import_module("physical_ai_evals.libero")


def _listing(paths: list[str]):
    return lambda *_args, **_kwargs: daft.from_pydict({"path": paths})


def _install_fake_libero(monkeypatch, tmp_path, suites):
    benchmark_module = SimpleNamespace(
        get_benchmark_dict=lambda: {
            suite: (lambda tasks=tasks: SimpleNamespace(get_task=tasks.__getitem__))
            for suite, tasks in suites.items()
        }
    )
    libero_package = ModuleType("libero")
    libero_module = ModuleType("libero.libero")
    libero_module.benchmark = benchmark_module
    libero_module.get_libero_path = lambda key: tmp_path / key
    libero_package.libero = libero_module
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)


def _task(suite: str, task_id: int):
    return SimpleNamespace(
        problem_folder=suite,
        bddl_file=f"task_{task_id}.bddl",
        init_states_file=f"task_{task_id}.pruned_init",
        language=f"perform {suite} task {task_id}",
        name=f"{suite}_task_{task_id}",
    )


def _local_hf_uris(monkeypatch, tmp_path):
    monkeypatch.setattr(
        benchmarks,
        "_hf_uri",
        lambda _repo, _revision, path: format(f"{tmp_path}/{{}}", path),
    )


def test_standard_libero_builds_lazy_executable_episode_grid(tmp_path, monkeypatch):
    _install_fake_libero(
        monkeypatch,
        tmp_path,
        {"libero_spatial": {task_id: _task("libero_spatial", task_id) for task_id in range(10)}},
    )

    benchmark = libero(
        "libero_spatial",
        task_ids=(1, 3),
        episodes=2,
        seed=11,
    )

    assert isinstance(benchmark.rollouts, daft.DataFrame)
    data = benchmark.rollouts.sort(["task_id", "init_state_id"]).to_pydict()
    assert data["task_id"] == [1, 1, 3, 3]
    assert data["init_state_id"] == [0, 1, 0, 1]
    assert data["seed"] == [11] * 4
    assert data["max_steps"] == [250] * 4
    assert data["instruction"] == [
        "perform libero_spatial task 1",
        "perform libero_spatial task 1",
        "perform libero_spatial task 3",
        "perform libero_spatial task 3",
    ]
    assert data["bddl_path"] == [
        "libero://bddl_files/libero_spatial/task_1.bddl",
        "libero://bddl_files/libero_spatial/task_1.bddl",
        "libero://bddl_files/libero_spatial/task_3.bddl",
        "libero://bddl_files/libero_spatial/task_3.bddl",
    ]
    assert len(set(data["episode_key"])) == 4
    assert all(key.startswith("e") for key in data["episode_key"])


def test_runtime_consumes_resolved_asset_references(tmp_path, monkeypatch):
    bddl = tmp_path / "bddl_files" / "libero_spatial" / "task_0.bddl"
    init = tmp_path / "init_states" / "libero_spatial" / "task_0.pruned_init"
    bddl.parent.mkdir(parents=True)
    init.parent.mkdir(parents=True)
    bddl.write_text("(:language perform task zero)\n", encoding="utf-8")
    expected = np.arange(4, dtype=np.float32)
    torch.save([expected], init)
    _install_fake_libero(monkeypatch, tmp_path, {"libero_spatial": {}})

    runtime = LiberoRuntime()
    monkeypatch.setattr(
        runtime,
        "_replace_environment",
        lambda *_args, **_kwargs: setattr(runtime, "_environment", "environment"),
    )
    environment, instruction, state, task_name = runtime.open(
        {
            "task_key": "0",
            "task_name": "task_zero",
            "instruction": "perform task zero",
            "bddl_path": "libero://bddl_files/libero_spatial/task_0.bddl",
            "init_path": "libero://init_states/libero_spatial/task_0.pruned_init",
            "init_state_id": 0,
            "seed": 7,
        }
    )

    assert environment == "environment"
    assert instruction == "perform task zero"
    assert task_name == "task_zero"
    np.testing.assert_array_equal(state, expected)


def test_runtime_normalizes_libero_observations():
    runtime = LiberoRuntime()
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    wrist = image + 20

    observation = runtime.normalize_observation(
        {
            "agentview_image": image,
            "robot0_eye_in_hand_image": wrist,
            "robot0_eef_pos": [0.1, 0.2, 0.3],
            "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
            "robot0_gripper_qpos": [0.04, -0.04],
        }
    )

    np.testing.assert_array_equal(observation["image"], image[::-1, ::-1])
    np.testing.assert_array_equal(observation["wrist_image"], wrist[::-1, ::-1])
    np.testing.assert_allclose(
        observation["state"],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04],
    )
    np.testing.assert_allclose(observation["eef_position"], [0.1, 0.2, 0.3])
    assert observation["gripper"] == pytest.approx(0.08)


def test_runtime_normalizes_vector_observation_rows():
    runtime = LiberoRuntime()
    images = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)

    observations = runtime.normalize_observations(
        {
            "agentview_image": images,
            "robot0_eef_pos": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "robot0_eef_quat": np.array(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
            ),
            "robot0_gripper_qpos": np.array([[0.04, -0.04], [0.03, -0.03]]),
        },
        count=2,
    )

    assert len(observations) == 2
    np.testing.assert_array_equal(observations[1]["image"], images[1, ::-1, ::-1])
    np.testing.assert_allclose(observations[1]["eef_position"], [0.4, 0.5, 0.6])
    assert observations[1]["gripper"] == pytest.approx(0.06)


def test_para_plan_maps_eval_ids_to_sorted_goal_environments(tmp_path, monkeypatch):
    paths = [
        "bddl_files/act_lexical_addition_eval3_ver7.bddl",
        "bddl_files/obj_same_polarity_eval8_ver2.bddl",
    ]
    instructions = {
        paths[0]: "place the bowl beside the plate",
        paths[1]: "put the mug on the table",
    }
    for path, instruction in instructions.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"(:language {instruction})\n", encoding="utf-8")

    monkeypatch.setattr(benchmarks, "_glob_repo_files", _listing(paths))
    _local_hf_uris(monkeypatch, tmp_path)

    benchmark = libero_para(task_ids=[3, 8], episodes=2)
    data = benchmark.rollouts.sort(["task_id", "init_state_id"]).to_pydict()

    assert data["benchmark"] == ["libero_para"] * 4
    assert data["suite"] == ["libero_goal"] * 4
    assert data["suite_variant"] == ["libero_para"] * 4
    assert data["task_id"] == [3, 3, 8, 8]
    assert data["instruction"] == [
        "place the bowl beside the plate",
        "place the bowl beside the plate",
        "put the mug on the table",
        "put the mug on the table",
    ]
    assert data["task_name"] == [
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "put_the_wine_bottle_on_top_of_the_cabinet",
    ]
    assert data["bddl_path"] == [
        "libero://bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl",
        "libero://bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl",
        "libero://bddl_files/libero_goal/put_the_wine_bottle_on_top_of_the_cabinet.bddl",
        "libero://bddl_files/libero_goal/put_the_wine_bottle_on_top_of_the_cabinet.bddl",
    ]
    assert data["init_path"] == [
        "libero://init_states/libero_goal/put_the_bowl_on_the_plate.pruned_init",
        "libero://init_states/libero_goal/put_the_bowl_on_the_plate.pruned_init",
        "libero://init_states/libero_goal/put_the_wine_bottle_on_top_of_the_cabinet.pruned_init",
        "libero://init_states/libero_goal/put_the_wine_bottle_on_top_of_the_cabinet.pruned_init",
    ]


def test_pro_plan_pairs_variant_then_suite_initial_states(tmp_path, monkeypatch):
    paths = [
        "bddl_files/01_visual_noise_glare/bddl/libero_goal/turn_on_the_stove.bddl",
        "bddl_files/libero_spatial_lan/pick_up_the_bowl.bddl",
        "init_files/libero_goal/turn_on_the_stove.pruned_init",
        "init_files/libero_spatial_lan/pick_up_the_bowl.pruned_init",
    ]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".bddl"):
            target.write_text(f"(:language execute {target.stem})\n", encoding="utf-8")
        else:
            target.write_bytes(b"fixture")

    monkeypatch.setattr(benchmarks, "_glob_repo_files", _listing(paths))
    _local_hf_uris(monkeypatch, tmp_path)

    spatial = libero_pro("libero_spatial", episodes=3, max_steps=17)
    spatial_data = spatial.rollouts.sort("init_state_id").to_pydict()
    assert spatial_data["suite_variant"] == ["libero_spatial_lan"] * 3
    assert spatial_data["perturbation"] == ["lan"] * 3
    assert spatial_data["init_state_id"] == [0, 1, 2]
    assert spatial_data["max_steps"] == [17] * 3
    assert spatial_data["bddl_path"] == [str(tmp_path / paths[1])] * 3
    assert spatial_data["init_path"] == [str(tmp_path / paths[3])] * 3

    goal = libero_pro("libero_goal", episodes=1)
    goal_data = goal.rollouts.to_pydict()
    assert goal_data["suite_variant"] == ["01_visual_noise_glare"]
    assert goal_data["perturbation"] == ["01_visual_noise_glare"]
    assert goal_data["init_path"] == [str(tmp_path / paths[2])]
