"""End-to-end conformance tests for the public evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import daft
import pytest

import physical_ai_evals.rollout as rollout
from physical_ai_evals import canonical_signature, evaluate, read_evaluation
from physical_ai_evals.testing import mock_benchmark, mock_policy

EPISODE_SIGNATURE_COLUMNS = (
    "episode_index",
    "episode_id",
    "task_id",
    "init_state_id",
    "success",
    "length",
    "reward",
)
STEP_SIGNATURE_COLUMNS = (
    "episode_index",
    "episode_id",
    "frame_index",
    "timestamp",
    "action",
    "reward",
    "next.done",
    "observation.state",
    "observation.eef_position",
    "observation.gripper",
)

EPISODE_SIGNATURE = "b4d7cd10f80d2dca273f3806e0ac801bdaee6b807d6ee01409ad846aaf208e61"
STEP_SIGNATURE = "ca63c19601b22f71a34f148485cfb2b6b6b6fa2d8c8c09bc0aadad2e7d7e721a"


def _signatures(evaluation) -> tuple[str, str]:
    return (
        canonical_signature(
            evaluation.episodes,
            sort_by=("episode_index",),
            columns=EPISODE_SIGNATURE_COLUMNS,
        ),
        canonical_signature(
            evaluation.steps,
            sort_by=("episode_index", "frame_index"),
            columns=STEP_SIGNATURE_COLUMNS,
        ),
    )


def _only_evaluation_root(out: Path) -> Path:
    roots = [path for path in out.iterdir() if path.is_dir()]
    assert len(roots) == 1
    return roots[0]


def test_evaluate_is_deterministic_lazy_and_resumable(tmp_path):
    counter = tmp_path / "policy-loads.txt"
    out = tmp_path / "runs"
    policy = mock_policy(counter_path=counter)
    benchmark = mock_benchmark()

    evaluation = evaluate(policy, benchmark, out=out, write_video=False)

    assert evaluation.episodes.count_rows() == 6
    assert evaluation.steps.count_rows() == 36
    assert evaluation.success_rate() == pytest.approx(2 / 3)
    assert evaluation.metrics("task_id").sort("task_id").to_pydict() == {
        "task_id": [0, 1, 2],
        "episodes": [2, 2, 2],
        "successes": [2, 0, 2],
        "mean_steps": [5.0, 6.0, 7.0],
    }
    assert _signatures(evaluation) == (EPISODE_SIGNATURE, STEP_SIGNATURE)
    assert counter.read_text(encoding="utf-8").splitlines() == ["initialized"]

    resumed = evaluate(policy, benchmark, out=out, write_video=False)

    assert resumed.path == evaluation.path
    assert _signatures(resumed) == (EPISODE_SIGNATURE, STEP_SIGNATURE)
    # A fully complete resume does not instantiate the model or runtime.
    assert counter.read_text(encoding="utf-8").splitlines() == ["initialized"]


def test_episode_failure_lands_prior_work_and_resume_repairs(tmp_path, monkeypatch):
    out = tmp_path / "runs"
    original = rollout._run_episode
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated rollout crash")
        return original(*args, **kwargs)

    monkeypatch.setattr(rollout, "_run_episode", fail_on_second)
    with pytest.raises(RuntimeError, match="rollout crash"):
        evaluate(mock_policy(), mock_benchmark(), out=out, write_video=False)

    partial = read_evaluation(_only_evaluation_root(out))
    assert partial.episodes.count_rows() == 1
    assert partial.steps.count_rows() in {5, 6, 7}

    monkeypatch.setattr(rollout, "_run_episode", original)
    repaired = evaluate(mock_policy(), mock_benchmark(), out=out, write_video=False)

    assert repaired.episodes.count_rows() == 6
    assert repaired.steps.count_rows() == 36
    assert repaired.episodes.select("episode_key").distinct().count_rows() == 6
    assert _signatures(repaired) == (EPISODE_SIGNATURE, STEP_SIGNATURE)


def test_steps_without_completion_are_overwritten_on_resume(tmp_path, monkeypatch):
    out = tmp_path / "runs"
    original = rollout._write_partition
    failed = False

    def fail_first_episode_write(frame, root):
        nonlocal failed
        if root.name == "episodes" and not failed:
            failed = True
            raise RuntimeError("simulated completion write crash")
        return original(frame, root)

    monkeypatch.setattr(rollout, "_write_partition", fail_first_episode_write)
    with pytest.raises(RuntimeError, match="completion write crash"):
        evaluate(mock_policy(), mock_benchmark(), out=out, write_video=False)

    partial = read_evaluation(_only_evaluation_root(out))
    assert partial.episodes.count_rows() == 0
    assert partial.steps.count_rows() in {5, 6, 7}

    monkeypatch.setattr(rollout, "_write_partition", original)
    repaired = evaluate(mock_policy(), mock_benchmark(), out=out, write_video=False)

    assert repaired.episodes.count_rows() == 6
    assert repaired.steps.count_rows() == 36
    assert (
        repaired.steps.select("episode_key", "frame_index").distinct().count_rows()
        == repaired.steps.count_rows()
    )
    assert _signatures(repaired) == (EPISODE_SIGNATURE, STEP_SIGNATURE)


def test_corrupt_step_partition_is_detected_and_repaired(tmp_path):
    policy = mock_policy()
    benchmark = mock_benchmark()
    evaluation = evaluate(
        policy,
        benchmark,
        out=tmp_path / "runs",
        write_video=False,
    )
    step_file = next((evaluation.path / "steps").rglob("*.parquet"))
    step_file.write_bytes(b"truncated parquet")

    repaired = evaluate(
        policy,
        benchmark,
        out=tmp_path / "runs",
        write_video=False,
    )

    assert repaired.episodes.count_rows() == 6
    assert repaired.steps.count_rows() == 36
    assert _signatures(repaired) == (EPISODE_SIGNATURE, STEP_SIGNATURE)


def test_wrong_schema_file_is_ignored(tmp_path):
    evaluation = evaluate(
        mock_policy(),
        mock_benchmark(),
        out=tmp_path / "runs",
        write_video=False,
    )
    daft.from_pydict({"junk": ["not a step"]}).write_parquet(
        evaluation.path / "steps" / "rogue"
    )

    reopened = read_evaluation(evaluation.path)

    assert reopened.steps.count_rows() == 36
    assert _signatures(reopened) == (EPISODE_SIGNATURE, STEP_SIGNATURE)


@pytest.mark.parametrize(
    ("benchmark", "message"),
    [
        (mock_benchmark(task_ids=()), "produced no episodes"),
        (mock_benchmark(task_ids=(0, 0), episodes=1), "duplicate episode keys"),
    ],
)
def test_invalid_benchmark_grids_fail_closed(tmp_path, benchmark, message):
    with pytest.raises(ValueError, match=message):
        evaluate(
            mock_policy(),
            benchmark,
            out=tmp_path / "runs",
            write_video=False,
        )


def test_video_paths_are_atomic_and_readable(tmp_path):
    evaluation = evaluate(
        mock_policy(),
        mock_benchmark(task_ids=(0,), episodes=1),
        out=tmp_path / "runs",
    )
    episode = evaluation.episodes.to_pydict()

    for column in ("primary_video_path", "wrist_video_path"):
        path = evaluation.path / episode[column][0]
        assert path.is_file()
        assert path.suffix == ".mp4"
        assert path.stat().st_size > 0
    assert list(evaluation.path.rglob(".*.mp4")) == []
