from __future__ import annotations

import os
import re
from pathlib import Path

APP_DIR = "/workspace"
MODEL_CACHE_DIR = "/models"
OUTPUT_DIR = "/outputs"

MODAL_LOCAL_DIR_IGNORE = (
    ".context/**", ".git/**", ".ruff_cache/**", ".pytest_cache/**", ".venv/**",
    ".env", ".envrc", "**/.DS_Store", "**/__pycache__/**", "**/*.py[cod]", "data/**",
)

_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def hf_cache_env(model_cache_dir: str = MODEL_CACHE_DIR) -> dict[str, str]:
    """Point all Hugging Face caches at the model-cache Volume (Xet acceleration on)."""
    return {
        "HF_HOME": f"{model_cache_dir}/huggingface",
        "HF_HUB_CACHE": f"{model_cache_dir}/huggingface/hub",
        "TRANSFORMERS_CACHE": f"{model_cache_dir}/huggingface/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
    }


def normalize_hf_token_env() -> str | None:
    for key in _HF_TOKEN_ENV_VARS:
        token = os.environ.get(key)
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            return token
    return None


def _safe_dir_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", value.strip("/")) or "default"


def resolve_hf_model_path(
    repo_or_path: str,
    model_cache_dir: str | Path = MODEL_CACHE_DIR,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Local path for a model: an existing dir as-is, else an HF snapshot into the cache Volume."""
    model_path = Path(repo_or_path).expanduser()
    if model_path.exists():
        return model_path
    if repo_or_path.startswith(("/", ".", "~")):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    from huggingface_hub import snapshot_download

    local_dir = Path(model_cache_dir) / "huggingface" / "repos" / _safe_dir_name(repo_or_path) / _safe_dir_name(revision or "main")
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_or_path, revision=revision or None,
            token=token or normalize_hf_token_env(), local_dir=str(local_dir),
        )
    )
