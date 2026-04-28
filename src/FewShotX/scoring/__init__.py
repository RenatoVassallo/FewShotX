from importlib import import_module
from typing import Any

__all__ = [
    "DictionaryScorer",
    "evaluate_predictions",
    "ZeroShotLearner",
    "ZeroShotNLI",
    "FewShotLearner",
    "FewShotLinearRegression",
    "BayesianMSELoss",
]

_LAZY_IMPORTS = {
    "DictionaryScorer": ("FewShotX.scoring.dictionary", "DictionaryScorer"),
    "evaluate_predictions": ("FewShotX.scoring.eval_plots", "evaluate_predictions"),
    "ZeroShotLearner": ("FewShotX.scoring.zeroshot", "ZeroShotLearner"),
    "ZeroShotNLI": ("FewShotX.scoring.zeroshot", "ZeroShotNLI"),
    "FewShotLearner": ("FewShotX.scoring.fewshot", "FewShotLearner"),
    "FewShotLinearRegression": ("FewShotX.scoring.fewshot", "FewShotLinearRegression"),
    "BayesianMSELoss": ("FewShotX.scoring.fewshot", "BayesianMSELoss"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
