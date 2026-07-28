# Notebooks

The examples are stored as Jupytext-compatible Python files so their code is reviewable in a
diff.

`query_libero_catalogs.py` queries the published LIBERO-Para and LIBERO-PRO repository
manifests, summarizes their task variants, and reads a small filtered set of BDDL instructions.

Run it from the repository root:

```bash
uv run python examples/notebooks/query_libero_catalogs.py
```
