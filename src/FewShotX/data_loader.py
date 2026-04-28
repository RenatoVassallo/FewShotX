from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import pandas as pd

_DATASET_PACKAGE = "FewShotX.datasets"
_SUPPORTED_SUFFIXES = {".csv", ".parquet"}


def available_datasets() -> list[str]:
    """Return bundled dataset filenames shipped with the package."""
    dataset_root = files(_DATASET_PACKAGE)
    return sorted(
        entry.name
        for entry in dataset_root.iterdir()
        if entry.is_file() and entry.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def load_dataset(filename: str) -> pd.DataFrame:
    """Load a bundled dataset from ``FewShotX.datasets``.

    Parameters
    ----------
    filename:
        Dataset filename, including the extension. Supported formats are
        ``.csv`` and ``.parquet``.
    """
    dataset_name = Path(filename).name
    suffix = Path(dataset_name).suffix.lower()

    if suffix not in _SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported file format {suffix!r}. Supported formats: {supported}.")

    dataset_root = files(_DATASET_PACKAGE)
    dataset_path = dataset_root.joinpath(dataset_name)

    if not dataset_path.is_file():
        available = ", ".join(available_datasets()) or "none"
        raise FileNotFoundError(
            f"Dataset {dataset_name!r} not found in {_DATASET_PACKAGE}. Available datasets: {available}."
        )

    with as_file(dataset_path) as resolved_path:
        if suffix == ".parquet":
            return pd.read_parquet(resolved_path)
        return pd.read_csv(resolved_path)
