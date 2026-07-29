# physical-ai-evals

`physical-ai-evals` provides:

- revision-checked Daft readers for ALOHA, EgoDex, and ABC-130K;
- revision-pinned task catalogs for LIBERO-Para and LIBERO-PRO;
- selective robomimic/LIBERO HDF5 ingest through `daft.file.Hdf5File`;
- a normalized `rollout-v2` Parquet schema with Daft tensor columns for episode and step records;
- OpenVLA and VLA-JEPA policy adapters; and
- local and Modal execution paths for LIBERO.

## Install

```bash
make setup
make check
```

The supported runtime is Python 3.12. Policy dependencies are optional and installed
separately because OpenVLA and VLA-JEPA require incompatible environments.

## Start here

- [Dataset catalogs](datasets.md)
- [Evaluation protocol](evaluation.md)
- [Troubleshooting](troubleshooting.md)

The package API and command-line examples are in the
[repository README](https://github.com/Eventual-Inc/physical-ai-evals#readme).
