"""Public API for FewShotX.

The top-level package exposes the small teaching-oriented surface used in the
course notebooks while deferring heavyweight imports until they are actually
needed.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]

__all__ = [
    "Embeddings",
    "DictionaryScorer",
    "ZeroShotLearner",
    "ZeroShotNLI",
    "FewShotLearner",
    "FewShotLinearRegression",
    "BayesianMSELoss",
    "evaluate_predictions",
    "load_dataset",
    "available_datasets",
    "configure_notebook",
]

_LAZY_IMPORTS = {
    "Embeddings": ("FewShotX.embeddings.embed", "Embeddings"),
    "DictionaryScorer": ("FewShotX.scoring.dictionary", "DictionaryScorer"),
    "ZeroShotLearner": ("FewShotX.scoring.zeroshot", "ZeroShotLearner"),
    "ZeroShotNLI": ("FewShotX.scoring.zeroshot", "ZeroShotNLI"),
    "FewShotLearner": ("FewShotX.scoring.fewshot", "FewShotLearner"),
    "FewShotLinearRegression": ("FewShotX.scoring.fewshot", "FewShotLinearRegression"),
    "BayesianMSELoss": ("FewShotX.scoring.fewshot", "BayesianMSELoss"),
    "evaluate_predictions": ("FewShotX.scoring.eval_plots", "evaluate_predictions"),
    "load_dataset": ("FewShotX.data_loader", "load_dataset"),
    "available_datasets": ("FewShotX.data_loader", "available_datasets"),
    "configure_notebook": ("FewShotX.notebook", "configure_notebook"),
}

def _source_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if tomllib is None:
        return "0.0.0"
    try:
        with pyproject_path.open("rb") as handle:
            return tomllib.load(handle)["project"]["version"]
    except Exception:
        return "0.0.0"


try:
    __version__ = version("fewshotx")
except PackageNotFoundError:
    __version__ = _source_version()


def __getattr__(name: str) -> Any:
    """Lazily resolve public names to keep import cost low."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
