# physical-ai-evals

`physical-ai-evals` provides:

- revision-pinned Daft catalogs for LIBERO-Para and LIBERO-PRO;
- selective robomimic/LIBERO HDF5 ingest through `daft.file.Hdf5File`;
- a normalized Parquet schema for episode and step records;
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
