# Common workflows. `make help` lists targets; CI runs the same commands (.github/workflows/ci.yml).

VENV    ?= .venv
SUITES  ?= libero_spatial
EPISODES ?= 10
RUFF_VERSION ?= 0.15.21
TY_VERSION ?= 0.0.56
MKDOCS_VERSION ?= 1.6.1
MKDOCS_MATERIAL_VERSION ?= 9.7.6
TWINE_VERSION ?= 6.2.0
CHECK_WHEEL_CONTENTS_VERSION ?= 0.6.3

.DEFAULT_GOAL := help
.PHONY: help lock lock-check setup lint fmt typecheck test check docs docs-build build clean \
        smoke-openvla smoke-vla-jepa rollout-openvla rollout-vla-jepa

help: ## List available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

lock: ## Refresh the Python 3.12 dependency lock
	uv lock --python 3.12

lock-check: ## Fail if pyproject.toml and uv.lock disagree
	uv lock --check

setup: ## Create the frozen dev venv (Python 3.12 + CPU Torch), mirroring CI
	uv sync --frozen --extra ingest_hdf5 --extra modal
	# CPU wheel keeps all policy-adapter tests active without resolving a CUDA stack.
	@if [ "$$(uname -s)" = "Linux" ]; then \
		uv pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu; \
	else \
		uv pip install torch==2.2.0; \
	fi

lint: ## ruff check
	uvx ruff==$(RUFF_VERSION) check physical_ai_evals/ tests/ .github/scripts/

fmt: ## ruff format + autofix
	uvx ruff==$(RUFF_VERSION) format physical_ai_evals/ tests/ .github/scripts/
	uvx ruff==$(RUFF_VERSION) check --fix physical_ai_evals/ tests/ .github/scripts/

typecheck: ## ty check (unresolved lazy imports downgraded to warnings)
	uvx ty@$(TY_VERSION) check physical_ai_evals/ --exit-zero-on-warning

test: ## Run the CPU-only test suite
	# The venv's python, not `uv run pytest`: the dev env is hand-composed (CPU torch on
	# top of the editable install); `uv run` would re-lock and sync over it.
	$(VENV)/bin/python -m pytest tests/ -q

check: lock-check lint typecheck test ## Everything CI gates on

docs: ## Serve the docs site locally
	uvx --from mkdocs==$(MKDOCS_VERSION) --with mkdocs-material==$(MKDOCS_MATERIAL_VERSION) mkdocs serve

docs-build: ## Build the docs site (strict, as CI does)
	uvx --from mkdocs==$(MKDOCS_VERSION) --with mkdocs-material==$(MKDOCS_MATERIAL_VERSION) mkdocs build --strict

build: ## Build sdist + wheel
	rm -rf build dist *.egg-info
	uv build --sdist --wheel
	uvx twine==$(TWINE_VERSION) check dist/*
	uvx check-wheel-contents==$(CHECK_WHEEL_CONTENTS_VERSION) dist/*.whl

clean: ## Remove build/test artifacts
	rm -rf build dist site *.egg-info .pytest_cache .ruff_cache

# --- Modal (one-time: `modal token new` + `modal secret create hf-token HF_TOKEN=...`) ---
# The venv's modal, not a global CLI: `modal run` imports the app file locally, which pulls
# in the physical_ai_evals package and its dependencies.

smoke-openvla: ## CPU image smoke test for the OpenVLA Modal app
	$(VENV)/bin/modal run physical_ai_evals/modal_app.py --smoke-test

smoke-vla-jepa: ## CPU image smoke test for the VLA-JEPA Modal app
	$(VENV)/bin/modal run physical_ai_evals/modal_vla_jepa_app.py --smoke-test

rollout-openvla: ## OpenVLA sweep on Modal (SUITES=..., EPISODES=...)
	$(VENV)/bin/modal run physical_ai_evals/modal_app.py --policy-type openvla --suites $(SUITES) --episodes $(EPISODES)

rollout-vla-jepa: ## VLA-JEPA sweep on Modal (SUITES=..., EPISODES=...)
	$(VENV)/bin/modal run physical_ai_evals/modal_vla_jepa_app.py --suites $(SUITES) --episodes $(EPISODES)
