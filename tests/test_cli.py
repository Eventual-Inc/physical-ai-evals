from physical_ai_evals.cli import main


def test_cli_dry_run_resolves_without_policy_or_simulator_import(capsys, tmp_path):
    result = main(
        [
            "evaluate",
            "--policy",
            "openvla",
            "--benchmark",
            "libero",
            "--suite",
            "libero_spatial",
            "--tasks",
            "0",
            "--episodes",
            "2",
            "--out",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "openvla-7b-finetuned-libero-spatial" in output
    assert "'episodes_per_task': 2" in output
