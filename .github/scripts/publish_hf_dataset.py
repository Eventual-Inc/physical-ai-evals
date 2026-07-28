from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SOURCE_DIR = "results/libero-spatial-pilot-2026-07-02"
SOURCE_FILES = (
    "README.md",
    "build_bundle.py",
    "episodes.csv",
    "failure_signatures.csv",
    "manifest.json",
    "steps.parquet",
    "summary.json",
)


def git_file(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_card(repo_id: str, source_commit: str) -> str:
    return f"""---
license: apache-2.0
task_categories:
  - robotics
tags:
  - libero
  - openvla
  - vla-jepa
configs:
  - config_name: steps
    data_files: steps.parquet
  - config_name: episodes
    data_files: episodes.csv
---

# LIBERO-Spatial paired pilot

Historical rollout records for OpenVLA and VLA-JEPA on 100 paired LIBERO-Spatial episode
specifications. The dataset contains 200 policy/episode records and 23,283 transition rows.

This is an exploratory trace, not a LIBERO reference evaluation or a confirmatory model
comparison. The artifact label `2026-07-02` is not a recorded execution timestamp.

## Files

- `steps.parquet`: normalized `rollout-v1` transition rows.
- `episodes.csv`: one row per policy and episode specification.
- `failure_signatures.csv`: post-hoc candidate signals for failed episodes.
- `summary.json`: descriptive aggregate counts.
- `manifest.json`: recorded configuration and provenance limitations.
- `build_bundle.py`: deterministic validation and bundle construction.
- `ARTIFACT_README.md`: the original artifact documentation.
- `SOURCE_SHA256SUMS`: checksums recorded in source control.
- `SHA256SUMS`: checksums for the files published here.

## Query with Daft

```python
import daft

steps = daft.read_parquet("hf://datasets/{repo_id}/steps.parquet")
failures = steps.where(steps["success"] == False)
failures.select("policy_type", "episode_id", "task_id", "step_idx").show()
```

## Provenance limits

- The traces store requested seed 7, but the simulator seed is 0/unverified.
- Policy random-number generator state was not recorded.
- The OpenVLA checkpoint identity is inferred rather than established by the stored model field.
- Model revisions, exact evaluation code revision, dependency lock, hardware, container image,
  and execution timestamps were not recorded.
- The trace uses 10 fixed initial states per task rather than the 50-trial reference protocol.
- Referenced frame and video files are not included.
- Failure signatures are unvalidated candidate labels, not causal diagnoses.

The source artifact was extracted from
[`{source_commit}`](https://github.com/Eventual-Inc/physical-ai-evals/commit/{source_commit}).
"""


def main() -> None:
    repo_id = os.environ["HF_DATASET_REPO"]
    source_commit = os.environ["SOURCE_COMMIT"]

    with tempfile.TemporaryDirectory() as temp_dir:
        upload_dir = Path(temp_dir)
        source_checksums = git_file(source_commit, f"{SOURCE_DIR}/SHA256SUMS")

        for name in SOURCE_FILES:
            output_name = "ARTIFACT_README.md" if name == "README.md" else name
            (upload_dir / output_name).write_bytes(
                git_file(source_commit, f"{SOURCE_DIR}/{name}")
            )

        expected = {}
        for line in source_checksums.decode().splitlines():
            digest, name = line.split("  ", 1)
            expected[name] = digest
        for name in SOURCE_FILES:
            output_name = "ARTIFACT_README.md" if name == "README.md" else name
            actual = sha256(upload_dir / output_name)
            if actual != expected[name]:
                raise RuntimeError(f"Checksum mismatch for {name}: {actual} != {expected[name]}")

        (upload_dir / "SOURCE_SHA256SUMS").write_bytes(source_checksums)
        (upload_dir / "README.md").write_text(
            dataset_card(repo_id, source_commit),
            encoding="utf-8",
        )

        published_files = sorted(
            path for path in upload_dir.iterdir() if path.name != "SHA256SUMS"
        )
        checksums = "".join(
            f"{sha256(path)}  {path.name}\n" for path in published_files
        )
        (upload_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")

        if os.environ.get("HF_DATASET_DRY_RUN") == "1":
            print(f"Verified {len(published_files)} source files for {repo_id}")
            return

        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=upload_dir,
            commit_message=f"Publish trace from physical-ai-evals@{source_commit}",
        )
        print(f"Published https://huggingface.co/datasets/{repo_id}")
        print(f"Commit: {commit}")


if __name__ == "__main__":
    main()
