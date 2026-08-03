"""LIBERO task plans and one simulator runtime."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import daft
import numpy as np
from daft import DataFrame, DataType, Expression, col, lit
from daft.functions import coalesce, format, hash, regexp, regexp_extract, when

from physical_ai_evals.geometry import quat_xyzw_to_axis_angle
from physical_ai_evals.rollout import Benchmark, RuntimeObservation
from physical_ai_evals.schema import EEF_POS_DIM, STATE_DIM

LIBERO_PRO_CODE_REPOSITORY = "https://github.com/Zxy-MLlab/LIBERO-PRO.git"
LIBERO_PRO_CODE_REVISION = "eafdb809426b13153aa1e4c42d6601844217dfec"
LIBERO_PARA_REPO_ID = "HAI-Lab/LIBERO-Para"
LIBERO_PARA_REVISION = "d306f66f8b441cad1155b21a3f69e440079c81c9"
LIBERO_PRO_REPO_ID = "zhouxueyang/LIBERO-Pro"
LIBERO_PRO_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"

SUITE_TASKS: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
# LIBERO-Para's eval IDs follow lexicographically sorted Goal BDDL names,
# not LIBERO's benchmark task order. Its renamed eval init files are exact
# copies of these tasks' standard Goal init files.
_PARA_GOAL_TASKS = (
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
)

_PARA_TASK = r"bddl_files/((act|obj|comp)_(.+)_eval(\d+)_ver(\d+))\.bddl"
_PRO_CONFIGURED = r"bddl_files/(\d+_[^/]+)/bddl/(libero_(?:10|goal|object|spatial))/([^/]+)\.bddl"
_PRO_PERTURBED = (
    r"bddl_files/(libero_(?:10|goal|object|spatial))_(lan|object|swap|task)/([^/]+)\.bddl"
)
_HF_DATASET_URI = re.compile(
    r"^hf://datasets/(?P<repo>[^/]+/[^/@]+)@(?P<revision>[^/]+)/(?P<path>.+)$"
)
_LIBERO_ASSET_URI = re.compile(r"^libero://(?P<root>bddl_files|init_states)/(?P<path>.+)$")

_INSTALLED_TASK = DataType.struct(
    {
        "standard_bddl_path": DataType.string(),
        "standard_init_path": DataType.string(),
        "standard_instruction": DataType.string(),
        "standard_task_name": DataType.string(),
    }
)


# Installed LIBERO task rows -------------------------------------------------


@daft.func.batch(
    return_dtype=_INSTALLED_TASK,
    unnest=True,
    use_process=False,
    batch_size=128,
)
def _installed_task(
    suites: daft.Series,
    task_ids: daft.Series,
) -> list[dict[str, str]]:
    """Resolve installed tasks once per suite and Daft batch."""
    from libero.libero import benchmark

    names = suites.to_pylist()
    ids = task_ids.to_pylist()
    benchmark_types = benchmark.get_benchmark_dict()
    suite_objects = {name: benchmark_types[name]() for name in set(names)}
    resolved = []
    for name, task_id in zip(names, ids, strict=True):
        task = suite_objects[name].get_task(task_id)
        problem = str(task.problem_folder)
        resolved.append(
            {
                "standard_bddl_path": f"libero://bddl_files/{problem}/{task.bddl_file}",
                "standard_init_path": (f"libero://init_states/{problem}/{task.init_states_file}"),
                "standard_instruction": str(task.language),
                "standard_task_name": str(task.name),
            }
        )
    return resolved


def _current_repo_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(repo_id=repo_id).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face did not return a revision for {repo_id}")
    return revision


def _check_repo_revision(repo_id: str, revision: str) -> None:
    current = _current_repo_revision(repo_id)
    if current != revision:
        raise RuntimeError(
            f"{repo_id} moved from expected revision {revision} to {current}; "
            "review the upstream change and update the recorded revision"
        )


def _glob_repo_files(
    repo_id: str,
    revision: str,
    patterns: Iterable[str],
    *,
    io_config=None,
) -> DataFrame:
    """Return lazy paths after verifying the dataset's pinned revision."""
    _check_repo_revision(repo_id, revision)
    root = f"hf://datasets/{repo_id}"
    listing = daft.from_glob_path(
        [f"{root}/{pattern.lstrip('/')}" for pattern in patterns],
        io_config=io_config,
    )
    _check_repo_revision(repo_id, revision)
    return listing.select(col("path").substr(len(root) + 1).alias("path"))


def _hf_uri(repo_id: str, revision: str, repo_path: Expression | str) -> Expression:
    return format(f"hf://datasets/{repo_id}@{revision}/{{}}", repo_path)


# Executable episode rows ---------------------------------------------------


def _rollouts(
    tasks: DataFrame,
    *,
    benchmark: str,
    revision: str,
    episodes: int,
    seed: int,
    max_steps: int | None,
) -> DataFrame:
    """Expand resolved task rows into the common executable episode grid."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if episodes > 50:
        raise ValueError(
            "LIBERO publishes 50 fixed initial states per task; episodes must be <= 50"
        )

    expanded = tasks.join(
        daft.from_pydict({"init_state_id": list(range(episodes))}),
        how="cross",
    ).with_columns(
        {
            "benchmark": lit(benchmark),
            "benchmark_revision": lit(revision),
            "seed": lit(seed),
            "max_steps": (
                lit(max_steps)
                if max_steps is not None
                else when(col("suite") == lit("libero_spatial"), lit(250))
                .when(col("suite") == lit("libero_object"), lit(280))
                .when(col("suite") == lit("libero_10"), lit(520))
                .when(col("suite") == lit("libero_90"), lit(400))
                .otherwise(lit(300))
            ),
        }
    )
    episode_id = format(
        "{}/{}/{}/{}/{}",
        col("benchmark"),
        col("suite"),
        col("task_key"),
        col("init_state_id"),
        col("seed"),
    )
    return expanded.with_columns(
        {
            "episode_id": episode_id,
            # Prefix prevents Hive partition inference from parsing the hash as
            # a number and losing resume identity precision.
            "episode_key": format(
                "e{}",
                hash(episode_id, hash_function="sha1").cast(DataType.string()),
            ),
        }
    ).select(
        "episode_key",
        "episode_id",
        "benchmark",
        "benchmark_revision",
        "suite",
        "suite_variant",
        "perturbation",
        "task_id",
        "task_key",
        "task_name",
        "instruction",
        "bddl_path",
        "init_path",
        "init_state_id",
        "seed",
        "max_steps",
    )


# Standard LIBERO -----------------------------------------------------------


def libero(
    suite: str = "libero_spatial",
    *,
    task_ids: Sequence[int] | None = None,
    episodes: int = 50,
    seed: int = 7,
    max_steps: int | None = None,
) -> Benchmark:
    """Standard LIBERO, resolved from the installed simulator checkout."""
    if suite not in SUITE_TASKS:
        raise ValueError(f"unknown LIBERO suite {suite!r}; choose one of {sorted(SUITE_TASKS)}")
    ids = list(task_ids) if task_ids is not None else list(range(SUITE_TASKS[suite]))
    if any(task_id < 0 or task_id >= SUITE_TASKS[suite] for task_id in ids):
        raise ValueError(f"task_ids must be valid indices for {suite}")

    installed = daft.from_pydict({"task_id": ids}).select(
        "task_id",
        _installed_task(lit(suite), col("task_id")),
    )
    tasks = installed.select(
        lit(suite).alias("suite"),
        lit(None).cast(DataType.string()).alias("suite_variant"),
        lit(None).cast(DataType.string()).alias("perturbation"),
        "task_id",
        col("task_id").cast(DataType.string()).alias("task_key"),
        col("standard_task_name").alias("task_name"),
        col("standard_instruction").alias("instruction"),
        col("standard_bddl_path").alias("bddl_path"),
        col("standard_init_path").alias("init_path"),
    )
    return Benchmark(
        name="libero",
        revision=LIBERO_PRO_CODE_REVISION,
        rollouts=_rollouts(
            tasks,
            benchmark="libero",
            revision=LIBERO_PRO_CODE_REVISION,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
        ),
        runtime_factory=LiberoRuntime,
        metadata={"suite": suite, "simulator": LIBERO_PRO_CODE_REPOSITORY},
    )


# LIBERO-Para ---------------------------------------------------------------


def libero_para(
    *,
    task_ids: Sequence[int] | None = None,
    task_keys: Sequence[str] | None = None,
    paraphrase_types: Sequence[str] | None = None,
    episodes: int = 1,
    seed: int = 7,
    max_steps: int | None = None,
    io_config=None,
) -> Benchmark:
    """LIBERO-Para instructions on the official sorted Goal environments."""
    files = _glob_repo_files(
        LIBERO_PARA_REPO_ID,
        LIBERO_PARA_REVISION,
        ("bddl_files/*.bddl",),
        io_config=io_config,
    )
    path = col("path")
    paraphrases = files.where(regexp(path, _PARA_TASK)).select(
        lit("libero_goal").alias("suite"),
        regexp_extract(path, _PARA_TASK, 4).cast(DataType.int64()).alias("task_id"),
        regexp_extract(path, _PARA_TASK, 1).alias("task_key"),
        regexp_extract(path, _PARA_TASK, 2).alias("perturbation"),
        _hf_uri(LIBERO_PARA_REPO_ID, LIBERO_PARA_REVISION, path).alias("_instruction_path"),
    )
    if task_ids is not None:
        paraphrases = paraphrases.where(col("task_id").is_in(list(task_ids)))
    if task_keys is not None:
        paraphrases = paraphrases.where(col("task_key").is_in(list(task_keys)))
    if paraphrase_types is not None:
        paraphrases = paraphrases.where(col("perturbation").is_in(list(paraphrase_types)))

    environments = daft.from_pydict(
        {
            "task_id": list(range(len(_PARA_GOAL_TASKS))),
            "standard_task_name": list(_PARA_GOAL_TASKS),
            "standard_bddl_path": [
                f"libero://bddl_files/libero_goal/{name}.bddl" for name in _PARA_GOAL_TASKS
            ],
            "standard_init_path": [
                f"libero://init_states/libero_goal/{name}.pruned_init" for name in _PARA_GOAL_TASKS
            ],
        }
    )
    selected = paraphrases.join(environments, on="task_id", how="inner")
    instruction = regexp_extract(
        col("_instruction_path").download(io_config=io_config).cast(DataType.string()),
        r"(?s)\(:language\s+(.+?)\s*\)",
        1,
    )
    tasks = selected.select(
        "suite",
        lit("libero_para").alias("suite_variant"),
        "perturbation",
        "task_id",
        "task_key",
        col("standard_task_name").alias("task_name"),
        instruction.alias("instruction"),
        col("standard_bddl_path").alias("bddl_path"),
        col("standard_init_path").alias("init_path"),
    )
    return Benchmark(
        name="libero_para",
        revision=LIBERO_PARA_REVISION,
        rollouts=_rollouts(
            tasks,
            benchmark="libero_para",
            revision=LIBERO_PARA_REVISION,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
        ),
        runtime_factory=LiberoRuntime,
        metadata={
            "suite": "libero_goal",
            "simulator_revision": LIBERO_PRO_CODE_REVISION,
        },
    )


# LIBERO-Pro ----------------------------------------------------------------


def libero_pro(
    suite: str = "libero_spatial",
    *,
    perturbations: Sequence[str] | None = None,
    task_keys: Sequence[str] | None = None,
    episodes: int = 50,
    seed: int = 7,
    max_steps: int | None = None,
    io_config=None,
) -> Benchmark:
    """LIBERO-Pro, resolved from its published BDDL and initial states."""
    if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
        raise ValueError("LIBERO-Pro suite must be spatial, object, goal, or 10")

    files = _glob_repo_files(
        LIBERO_PRO_REPO_ID,
        LIBERO_PRO_REVISION,
        ("bddl_files/**/*.bddl", "init_files/**/*.pruned_init"),
        io_config=io_config,
    )
    path = col("path")
    configured = regexp(path, _PRO_CONFIGURED)
    bddls = (
        files.where(path.startswith("bddl_files/") & (configured | regexp(path, _PRO_PERTURBED)))
        .select(
            when(configured, regexp_extract(path, _PRO_CONFIGURED, 2))
            .otherwise(regexp_extract(path, _PRO_PERTURBED, 1))
            .alias("suite"),
            when(configured, regexp_extract(path, _PRO_CONFIGURED, 1))
            .otherwise(
                format(
                    "{}_{}",
                    regexp_extract(path, _PRO_PERTURBED, 1),
                    regexp_extract(path, _PRO_PERTURBED, 2),
                )
            )
            .alias("suite_variant"),
            when(configured, regexp_extract(path, _PRO_CONFIGURED, 1))
            .otherwise(regexp_extract(path, _PRO_PERTURBED, 2))
            .alias("perturbation"),
            when(configured, regexp_extract(path, _PRO_CONFIGURED, 3))
            .otherwise(regexp_extract(path, _PRO_PERTURBED, 3))
            .alias("task_name"),
            _hf_uri(LIBERO_PRO_REPO_ID, LIBERO_PRO_REVISION, path).alias("bddl_path"),
        )
        .with_column(
            "task_key",
            format("{}:{}", col("suite_variant"), col("task_name")),
        )
    )

    init_files = files.where(path.startswith("init_files/"))
    paired = bddls.with_columns(
        {
            "_variant_key": format(
                "init_files/{}/{}.pruned_init",
                col("suite_variant"),
                col("task_name"),
            ),
            "_suite_key": format(
                "init_files/{}/{}.pruned_init",
                col("suite"),
                col("task_name"),
            ),
        }
    )
    for key, hit in (
        ("_variant_key", "_variant_hit"),
        ("_suite_key", "_suite_hit"),
    ):
        paired = paired.join(
            init_files.select(path.alias(key), path.alias(hit)),
            on=key,
            how="left",
        )
    selected = paired.select(
        "suite",
        "suite_variant",
        "perturbation",
        "task_name",
        "task_key",
        "bddl_path",
        _hf_uri(
            LIBERO_PRO_REPO_ID,
            LIBERO_PRO_REVISION,
            coalesce(col("_variant_hit"), col("_suite_hit")),
        ).alias("init_path"),
    ).where(col("suite") == lit(suite))
    if perturbations is not None:
        selected = selected.where(col("perturbation").is_in(list(perturbations)))
    if task_keys is not None:
        selected = selected.where(col("task_key").is_in(list(task_keys)))

    instruction = regexp_extract(
        col("bddl_path").download(io_config=io_config).cast(DataType.string()),
        r"(?s)\(:language\s+(.+?)\s*\)",
        1,
    )
    tasks = selected.select(
        "suite",
        "suite_variant",
        "perturbation",
        lit(None).cast(DataType.int64()).alias("task_id"),
        "task_key",
        "task_name",
        instruction.alias("instruction"),
        "bddl_path",
        "init_path",
    )
    return Benchmark(
        name="libero_pro",
        revision=LIBERO_PRO_REVISION,
        rollouts=_rollouts(
            tasks,
            benchmark="libero_pro",
            revision=LIBERO_PRO_REVISION,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
        ),
        runtime_factory=LiberoRuntime,
        metadata={
            "suite": suite,
            "simulator_revision": LIBERO_PRO_CODE_REVISION,
        },
    )


# Simulator runtime ---------------------------------------------------------


def _set_mujoco_gl() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


def _offscreen_environment(
    bddl_path: str,
    camera_height: int,
    camera_width: int,
):
    """Construct MuJoCo only inside its simulator worker."""
    _set_mujoco_gl()
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )


def _local_path(path_or_uri: str) -> Path:
    path = Path(path_or_uri)
    if path.is_file():
        return path

    installed = _LIBERO_ASSET_URI.fullmatch(path_or_uri)
    if installed is not None:
        from libero.libero import get_libero_path

        path = Path(get_libero_path(installed.group("root"))) / installed.group("path")
        if path.is_file():
            return path
        raise FileNotFoundError(f"installed LIBERO asset not found: {path}")

    remote = _HF_DATASET_URI.fullmatch(path_or_uri)
    if remote is not None:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=remote.group("repo"),
                repo_type="dataset",
                revision=remote.group("revision"),
                filename=remote.group("path"),
            )
        )
    raise FileNotFoundError(f"benchmark file not found: {path_or_uri}")


@dataclass
class LiberoRuntime:
    """One cached scalar or subprocess-vector environment per rollout actor."""

    camera_height: int = 256
    camera_width: int = 256

    def __post_init__(self) -> None:
        self._environment: Any = None
        self._environment_key: tuple[Any, ...] | None = None
        self._init_cache: dict[str, Any] = {}

    def prepare(self) -> None:
        """Select spawn before policy construction initializes CUDA."""
        import multiprocessing

        method = multiprocessing.get_start_method(allow_none=True)
        if method is None:
            multiprocessing.set_start_method("spawn")
        elif method != "spawn":
            raise RuntimeError(
                "LIBERO subprocess environments require multiprocessing start method "
                f"'spawn', but {method!r} is already active"
            )

    def _derotate(self, image: Any) -> np.ndarray:
        return np.asarray(image)[::-1, ::-1]

    def _eef_position(self, observation: Mapping[str, Any]) -> np.ndarray | None:
        value = observation.get("robot0_eef_pos")
        if value is None:
            return None
        return np.asarray(value, np.float32).ravel()[:EEF_POS_DIM]

    def _gripper(self, observation: Mapping[str, Any]) -> float | None:
        value = observation.get("robot0_gripper_qpos")
        if value is None:
            return None
        qpos = np.asarray(value, np.float32).ravel()
        return float(qpos[0] - qpos[1]) if qpos.size >= 2 else float(qpos[0])

    def _proprio(self, observation: Mapping[str, Any]) -> np.ndarray | None:
        eef = observation.get("robot0_eef_pos")
        gripper = observation.get("robot0_gripper_qpos")
        if eef is None or gripper is None:
            return None

        parts = [np.asarray(eef, np.float32).ravel()[:3]]
        quaternion = observation.get("robot0_eef_quat")
        if quaternion is not None:
            parts.append(quat_xyzw_to_axis_angle(np.asarray(quaternion).reshape(1, 4))[0])
        gripper_array = np.asarray(gripper, np.float32).ravel()
        parts.append(gripper_array[:2] if gripper_array.size >= 2 else gripper_array)
        state = np.concatenate(parts).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"LIBERO proprioception must have shape {(STATE_DIM,)}, got {state.shape}"
            )
        return state

    def normalize_observation(self, observation: Mapping[str, Any]) -> RuntimeObservation:
        """Translate one raw robosuite observation into the evaluation schema."""
        return {
            "image": self._derotate(observation["agentview_image"]),
            "wrist_image": (
                self._derotate(observation["robot0_eye_in_hand_image"])
                if "robot0_eye_in_hand_image" in observation
                else None
            ),
            "state": self._proprio(observation),
            "eef_position": self._eef_position(observation),
            "gripper": self._gripper(observation),
        }

    def _observation_rows(
        self,
        observations: Any,
        count: int,
    ) -> list[Mapping[str, Any]]:
        if isinstance(observations, np.ndarray) and observations.dtype == object:
            rows = observations.tolist()
            if len(rows) == count and all(isinstance(row, Mapping) for row in rows):
                return rows
        if isinstance(observations, Sequence) and not isinstance(
            observations,
            (str, bytes, Mapping),
        ):
            rows = list(observations)
            if len(rows) == count and all(isinstance(row, Mapping) for row in rows):
                return cast(list[Mapping[str, Any]], rows)
        if isinstance(observations, Mapping):

            def take(value: Any, index: int) -> Any:
                if isinstance(value, Mapping):
                    return {name: take(child, index) for name, child in value.items()}
                array = np.asarray(value)
                if array.ndim > 0 and array.shape[0] == count:
                    return array[index]
                return value

            return [
                {name: take(value, index) for name, value in observations.items()}
                for index in range(count)
            ]
        raise TypeError(
            "LIBERO vector observations must be an object array, row sequence, "
            "or dictionary of batched arrays"
        )

    def normalize_observations(
        self,
        observations: Any,
        count: int,
    ) -> list[RuntimeObservation]:
        """Translate a LIBERO vector observation batch into evaluation rows."""
        return [
            self.normalize_observation(observation)
            for observation in self._observation_rows(observations, count)
        ]

    def _resolve(self, rollout: Mapping[str, Any]) -> dict[str, Any]:
        bddl_path = _local_path(str(rollout["bddl_path"]))
        init_ref = rollout.get("init_path")
        if init_ref is None:
            raise ValueError(f"task {rollout['task_key']!r} has no initial-state file")
        init_path = _local_path(str(init_ref))
        cache_key = str(init_path)
        if cache_key not in self._init_cache:
            # These are trusted simulator states from pinned benchmark sources,
            # not model weights. PyTorch 2.6 defaults weights_only to True.
            import torch

            self._init_cache[cache_key] = torch.load(init_path, weights_only=False)

        instruction = rollout.get("instruction")
        if instruction is None:
            raise ValueError(f"task {rollout['task_key']!r} has no instruction")
        init_state_id = int(rollout["init_state_id"])
        return {
            "bddl_path": bddl_path,
            "instruction": str(instruction),
            "init_state": self._init_cache[cache_key][init_state_id],
            "task_name": (
                str(rollout["task_name"]) if rollout.get("task_name") is not None else None
            ),
            "seed": int(rollout["seed"]),
        }

    def _replace_environment(self, key: tuple[Any, ...], bddl_path: Path, seed: int) -> None:
        if self._environment_key == key:
            return
        self.close()
        self._environment = _offscreen_environment(
            str(bddl_path),
            self.camera_height,
            self.camera_width,
        )
        self._environment.seed(seed)
        self._environment_key = key

    def open(self, rollout: Mapping[str, Any]) -> tuple[Any, str, Any, str | None]:
        task = self._resolve(rollout)
        key = ("scalar", str(task["bddl_path"]), task["seed"])
        self._replace_environment(key, task["bddl_path"], task["seed"])
        return (
            self._environment,
            task["instruction"],
            task["init_state"],
            task["task_name"],
        )

    def open_batch(
        self,
        rollouts: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, list[str], list[Any], list[str | None]]:
        """Open or reuse LIBERO's native CPU subprocess vector environment."""
        tasks = [self._resolve(rollout) for rollout in rollouts]
        environment_key = (
            "vector",
            tuple((str(task["bddl_path"]), task["seed"]) for task in tasks),
        )
        if self._environment_key != environment_key:
            self.close()
            _set_mujoco_gl()
            from libero.libero.envs import SubprocVectorEnv

            self._environment = SubprocVectorEnv(
                [
                    partial(
                        _offscreen_environment,
                        str(task["bddl_path"]),
                        self.camera_height,
                        self.camera_width,
                    )
                    for task in tasks
                ]
            )
            self._environment_key = environment_key

        self._environment.seed([task["seed"] for task in tasks])
        return (
            self._environment,
            [task["instruction"] for task in tasks],
            [task["init_state"] for task in tasks],
            [task["task_name"] for task in tasks],
        )

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
        self._environment = None
        self._environment_key = None
