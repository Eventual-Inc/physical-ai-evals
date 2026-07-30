"""The LIBERO benchmark family: lazy specs and one simulator runtime."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import daft
from daft import DataFrame, DataType, Expression, col, lit
from daft.functions import coalesce, format, hash, regexp, regexp_extract, when

from physical_ai_evals.rollout import Benchmark

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
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 250,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

_PARA_TASK = r"bddl_files/((act|obj|comp)_(.+)_eval(\d+)_ver(\d+))\.bddl"
_PRO_CONFIGURED = r"bddl_files/(\d+_[^/]+)/bddl/(libero_(?:10|goal|object|spatial))/([^/]+)\.bddl"
_PRO_PERTURBED = (
    r"bddl_files/(libero_(?:10|goal|object|spatial))_(lan|object|swap|task)/([^/]+)\.bddl"
)
_HF_DATASET_URI = re.compile(
    r"^hf://datasets/(?P<repo>[^/]+/[^/@]+)@(?P<revision>[^/]+)/(?P<path>.+)$"
)


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
    """Return one lazy, revision-relative path column."""
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


def _with_instructions(tasks: DataFrame, *, io_config=None) -> DataFrame:
    content = col("bddl_path").download(io_config=io_config).cast(DataType.string())
    instruction = regexp_extract(content, r"(?s)\(:language\s+(.+?)\s*\)", 1).alias("instruction")
    return tasks.with_column("instruction", instruction)


def libero_para_tasks(
    repo_id: str = LIBERO_PARA_REPO_ID,
    revision: str = LIBERO_PARA_REVISION,
    *,
    io_config=None,
) -> DataFrame:
    """Lazy LIBERO-Para catalog with revision-pinned BDDL references."""
    files = _glob_repo_files(repo_id, revision, ("bddl_files/*.bddl",), io_config=io_config)
    path = col("path")
    return files.select(
        lit("libero_para").alias("benchmark"),
        lit(revision).alias("benchmark_revision"),
        lit("libero_goal").alias("suite"),
        regexp_extract(path, _PARA_TASK, 4).cast(DataType.int64()).alias("task_id"),
        regexp_extract(path, _PARA_TASK, 1).alias("task_key"),
        regexp_extract(path, _PARA_TASK, 1).alias("task_name"),
        regexp_extract(path, _PARA_TASK, 2).alias("perturbation"),
        regexp_extract(path, _PARA_TASK, 3).alias("paraphrase_key"),
        regexp_extract(path, _PARA_TASK, 5).cast(DataType.int64()).alias("variant_id"),
        _hf_uri(repo_id, revision, path).alias("bddl_path"),
    )


def libero_pro_tasks(
    repo_id: str = LIBERO_PRO_REPO_ID,
    revision: str = LIBERO_PRO_REVISION,
    *,
    io_config=None,
) -> DataFrame:
    """Lazy LIBERO-Pro catalog with paired BDDL and initial-state files."""
    files = _glob_repo_files(
        repo_id,
        revision,
        ("bddl_files/**/*.bddl", "init_files/**/*.pruned_init"),
        io_config=io_config,
    )
    path = col("path")
    configured = regexp(path, _PRO_CONFIGURED)
    tasks = (
        files.where(path.startswith("bddl_files/"))
        .select(
            lit("libero_pro").alias("benchmark"),
            lit(revision).alias("benchmark_revision"),
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
            _hf_uri(repo_id, revision, path).alias("bddl_path"),
        )
        .with_column(
            "task_key",
            format("{}:{}", col("suite_variant"), col("task_name")),
        )
    )

    inits = files.where(path.startswith("init_files/"))
    keyed = tasks.with_columns(
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
        keyed = keyed.join(
            inits.select(path.alias(key), path.alias(hit)),
            on=key,
            how="left",
        )
    return keyed.select(
        *tasks.column_names,
        _hf_uri(
            repo_id,
            revision,
            coalesce(col("_variant_hit"), col("_suite_hit")),
        ).alias("init_path"),
    )


def _validate_episodes(episodes: int) -> None:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if episodes > 50:
        raise ValueError(
            "LIBERO publishes 50 fixed initial states per task; episodes must be <= 50"
        )


def _episode_specs(
    tasks: DataFrame,
    *,
    benchmark: str,
    revision: str,
    episodes: int,
    seed: int,
    max_steps: int | None,
) -> DataFrame:
    _validate_episodes(episodes)
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
            # Prefix keeps Hive partition inference from parsing a 64-bit key as
            # a float and losing resume identity precision.
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


def libero(
    suite: str = "libero_spatial",
    *,
    task_ids: Sequence[int] | None = None,
    episodes: int = 50,
    seed: int = 7,
    max_steps: int | None = None,
) -> Benchmark:
    """Standard LIBERO benchmark specs."""
    if suite not in SUITE_TASKS:
        raise ValueError(f"unknown LIBERO suite {suite!r}; choose one of {sorted(SUITE_TASKS)}")
    ids = list(task_ids) if task_ids is not None else list(range(SUITE_TASKS[suite]))
    if any(task_id < 0 or task_id >= SUITE_TASKS[suite] for task_id in ids):
        raise ValueError(f"task_ids must be valid indices for {suite}")
    tasks = daft.from_pydict(
        {
            "suite": [suite] * len(ids),
            "suite_variant": [None] * len(ids),
            "perturbation": [None] * len(ids),
            "task_id": ids,
            "task_key": [str(task_id) for task_id in ids],
            "task_name": [None] * len(ids),
            "instruction": [None] * len(ids),
            "bddl_path": [None] * len(ids),
            "init_path": [None] * len(ids),
        }
    )
    revision = LIBERO_PRO_CODE_REVISION
    return Benchmark(
        name="libero",
        revision=revision,
        specs=_episode_specs(
            tasks,
            benchmark="libero",
            revision=revision,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
        ),
        runtime_factory=LiberoRuntime,
        metadata={"suite": suite, "simulator": LIBERO_PRO_CODE_REPOSITORY},
    )


def libero_para(
    *,
    tasks: DataFrame | None = None,
    task_ids: Sequence[int] | None = None,
    task_keys: Sequence[str] | None = None,
    paraphrase_types: Sequence[str] | None = None,
    episodes: int = 1,
    seed: int = 7,
    max_steps: int | None = None,
    io_config=None,
) -> Benchmark:
    """LIBERO-Para specs; environments/init states stay standard LIBERO-Goal."""
    selected = tasks if tasks is not None else libero_para_tasks(io_config=io_config)
    if task_ids is not None:
        selected = selected.where(col("task_id").is_in(list(task_ids)))
    if task_keys is not None:
        selected = selected.where(col("task_key").is_in(list(task_keys)))
    if paraphrase_types is not None:
        selected = selected.where(col("perturbation").is_in(list(paraphrase_types)))
    selected = _with_instructions(selected, io_config=io_config).select(
        "suite",
        lit("libero_para").alias("suite_variant"),
        "perturbation",
        "task_id",
        "task_key",
        "task_name",
        "instruction",
        "bddl_path",
        lit(None).cast(DataType.string()).alias("init_path"),
    )
    return Benchmark(
        name="libero_para",
        revision=LIBERO_PARA_REVISION,
        specs=_episode_specs(
            selected,
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


def libero_pro(
    suite: str = "libero_spatial",
    *,
    perturbations: Sequence[str] | None = None,
    task_keys: Sequence[str] | None = None,
    tasks: DataFrame | None = None,
    episodes: int = 50,
    seed: int = 7,
    max_steps: int | None = None,
    io_config=None,
) -> Benchmark:
    """LIBERO-Pro specs backed by published BDDL and initial-state files."""
    if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
        raise ValueError("LIBERO-Pro suite must be spatial, object, goal, or 10")
    selected = tasks if tasks is not None else libero_pro_tasks(io_config=io_config)
    selected = selected.where(col("suite") == lit(suite))
    if perturbations is not None:
        selected = selected.where(col("perturbation").is_in(list(perturbations)))
    if task_keys is not None:
        selected = selected.where(col("task_key").is_in(list(task_keys)))
    selected = _with_instructions(selected, io_config=io_config).select(
        "suite",
        "suite_variant",
        "perturbation",
        lit(None).cast(DataType.int64()).alias("task_id"),
        "task_key",
        "task_name",
        "instruction",
        "bddl_path",
        "init_path",
    )
    return Benchmark(
        name="libero_pro",
        revision=LIBERO_PRO_REVISION,
        specs=_episode_specs(
            selected,
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


def _set_mujoco_gl() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


def _offscreen_environment(
    bddl_path: str,
    camera_height: int,
    camera_width: int,
):
    """Construct MuJoCo only inside its spawned simulator worker."""
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
    match = _HF_DATASET_URI.fullmatch(path_or_uri)
    if match is None:
        raise FileNotFoundError(f"benchmark file not found: {path_or_uri}")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=match.group("repo"),
            repo_type="dataset",
            revision=match.group("revision"),
            filename=match.group("path"),
        )
    )


@dataclass
class LiberoRuntime:
    """One cached scalar or subprocess-vector LIBERO environment per actor."""

    camera_height: int = 256
    camera_width: int = 256

    def __post_init__(self) -> None:
        self._environment: Any = None
        self._environment_key: tuple[Any, ...] | None = None
        self._task: Any = None
        self._init_cache: dict[tuple[Any, ...], Any] = {}

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

    def _replace_environment(self, key: tuple[Any, ...], bddl_path: Path, seed: int) -> None:
        if self._environment_key == key:
            return
        self.close()
        _set_mujoco_gl()
        from libero.libero.envs import OffScreenRenderEnv

        self._environment = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )
        self._environment.seed(seed)
        self._environment_key = key

    def _standard_assets(self, suite: str, task_id: int) -> tuple[Path, Any, Any]:
        _set_mujoco_gl()
        from libero.libero import benchmark, get_libero_path

        suite_object = benchmark.get_benchmark_dict()[suite]()
        task = suite_object.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        cache_key = ("standard", suite, task_id)
        if cache_key not in self._init_cache:
            # These are trusted simulator states from the pinned LIBERO-Pro
            # checkout, not model weights. PyTorch 2.6 changed torch.load's
            # default to weights_only=True, which rejects their NumPy objects.
            import torch

            init_path = (
                Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            )
            self._init_cache[cache_key] = torch.load(
                init_path,
                weights_only=False,
            )
        return bddl, task, self._init_cache[cache_key]

    def _standard(self, suite: str, task_id: int, seed: int) -> tuple[Any, Any, Any]:
        bddl, task, init_states = self._standard_assets(suite, task_id)
        key = ("standard", suite, task_id, seed)
        self._replace_environment(key, bddl, seed)
        return self._environment, task, init_states

    def _resolve(
        self,
        spec: Mapping[str, Any],
    ) -> tuple[Path, str, Any, str | None, int]:
        benchmark = str(spec["benchmark"])
        seed = int(spec["seed"])
        init_state_id = int(spec["init_state_id"])
        if benchmark in {"libero", "libero_para"}:
            suite = "libero_goal" if benchmark == "libero_para" else str(spec["suite"])
            task_id = int(spec["task_id"])
            bddl, task, init_states = self._standard_assets(suite, task_id)
            instruction = (
                str(spec["instruction"])
                if spec.get("instruction") is not None
                else str(getattr(task, "language", ""))
            )
            return (
                bddl,
                instruction,
                init_states[init_state_id],
                getattr(task, "name", None),
                seed,
            )

        if benchmark != "libero_pro":
            raise ValueError(f"unsupported benchmark runtime: {benchmark!r}")
        bddl_path = _local_path(str(spec["bddl_path"]))
        init_uri = spec.get("init_path")
        if init_uri is None:
            raise ValueError(f"LIBERO-Pro task {spec['task_key']!r} has no published init_path")
        init_path = _local_path(str(init_uri))
        cache_key = ("pro", str(init_path))
        if cache_key not in self._init_cache:
            import torch

            self._init_cache[cache_key] = torch.load(init_path, weights_only=False)
        return (
            bddl_path,
            str(spec["instruction"]),
            self._init_cache[cache_key][init_state_id],
            str(spec["task_name"]),
            seed,
        )

    def open(self, spec: Mapping[str, Any]) -> tuple[Any, str, Any, str | None]:
        bddl_path, instruction, init_state, task_name, seed = self._resolve(spec)
        key = ("scalar", str(bddl_path), seed)
        self._replace_environment(key, bddl_path, seed)
        return self._environment, instruction, init_state, task_name

    def open_batch(
        self,
        specs: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, list[str], list[Any], list[str | None]]:
        """Open or reuse LIBERO's native CPU subprocess vector environment."""
        resolved = [self._resolve(spec) for spec in specs]
        environment_key = (
            "vector",
            tuple((str(bddl_path), seed) for bddl_path, _, _, _, seed in resolved),
        )
        if self._environment_key != environment_key:
            self.close()
            _set_mujoco_gl()
            from libero.libero.envs import SubprocVectorEnv

            self._environment = SubprocVectorEnv(
                [
                    partial(
                        _offscreen_environment,
                        str(bddl_path),
                        self.camera_height,
                        self.camera_width,
                    )
                    for bddl_path, _, _, _, _ in resolved
                ]
            )
            self._environment_key = environment_key

        seeds = [seed for _, _, _, _, seed in resolved]
        self._environment.seed(seeds)
        return (
            self._environment,
            [instruction for _, instruction, _, _, _ in resolved],
            [init_state for _, _, init_state, _, _ in resolved],
            [task_name for _, _, _, task_name, _ in resolved],
        )

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
        self._environment = None
        self._environment_key = None
        self._task = None
