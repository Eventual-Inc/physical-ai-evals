# Notebooks

The examples are stored as Jupytext-compatible Python files so their code is reviewable in a
diff.

`query_libero_catalogs.py` queries the published LIBERO-Para and LIBERO-Pro
repository manifests and summarizes task variants.

`query_lerobot_datasets.py` queries ALOHA, EgoDex, and the ABC-130K smoke conversion without
decoding video.

Run it from the repository root:

```bash
uv run python examples/notebooks/query_libero_catalogs.py
uv run python examples/notebooks/query_lerobot_datasets.py
```
