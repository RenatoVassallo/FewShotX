from __future__ import annotations

import os
import warnings
from logging import ERROR, getLogger
from typing import Optional


def get_tqdm(progress_bar: Optional[str] = None):
    """Return a tqdm implementation based on the configured progress style."""
    selected = (progress_bar or os.environ.get("FEWSHOTX_PROGRESS_BAR", "auto")).lower()

    if selected == "rich":
        try:
            from tqdm.rich import tqdm, trange

            return tqdm, trange
        except Exception:
            selected = "auto"

    if selected == "notebook":
        from tqdm.notebook import tqdm, trange
    elif selected == "console":
        from tqdm import tqdm, trange
    else:
        from tqdm.auto import tqdm, trange

    return tqdm, trange


def configure_notebook(
    *,
    theme: Optional[str] = "whitegrid",
    progress_bar: str = "auto",
    suppress_warnings: bool = True,
):
    """Configure a notebook session for cleaner output and modern progress bars.

    Returns a tuple ``(tqdm, trange)`` so notebooks can opt into a consistent
    progress-bar implementation without repeating setup code.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["FEWSHOTX_PROGRESS_BAR"] = progress_bar

    if suppress_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r"`clean_up_tokenization_spaces` was not set.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*pin_memory.*no accelerator is found.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*pin_memory.*not supported on MPS.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
            category=Warning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Using `tqdm\.autonotebook\.tqdm` in notebook mode.*",
            category=Warning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"rich is experimental/alpha",
            category=Warning,
        )
        for logger_name in ("transformers", "sentence_transformers", "setfit", "huggingface_hub"):
            getLogger(logger_name).setLevel(ERROR)

    if theme:
        try:
            import seaborn as sns

            sns.set_theme(style=theme)
        except Exception:
            pass

    return get_tqdm(progress_bar)
