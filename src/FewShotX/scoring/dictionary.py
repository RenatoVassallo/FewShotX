from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional

import pandas as pd
import spacy

from FewShotX.notebook import get_tqdm


class DictionaryScorer:
    """Dictionary-based text scorer backed by spaCy tokenization."""

    def __init__(
        self,
        dictionaries: dict[str, Sequence[str]],
        model_name: str = "en_core_web_sm",
        *,
        use_lemma: bool = False,
        verbose: bool = True,
        batch_size: int = 256,
        store_tokens: bool = True,
        progress_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.dictionaries = {label: {str(term).lower() for term in terms} for label, terms in dictionaries.items()}
        self.model_name = model_name
        self.use_lemma = use_lemma
        self.verbose = verbose
        self.batch_size = batch_size
        self.store_tokens = store_tokens
        self.progress_kwargs = {
            "dynamic_ncols": True,
            "leave": False,
            "smoothing": 0.05,
            "colour": "#2f6db3",
            **(progress_kwargs or {}),
        }

        try:
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
        except OSError as exc:
            raise OSError(
                f"spaCy model {model_name!r} is not installed. "
                f"Install it before using DictionaryScorer."
            ) from exc

        self.nlp.max_length = 2_000_000

    def _preprocess_doc(self, doc) -> list[str]:
        token_attr = "lemma_" if self.use_lemma else "text"
        tokens = []
        for token in doc:
            if not token.is_alpha or token.is_stop:
                continue
            tokens.append(getattr(token, token_attr).lower())
        return tokens

    def preprocess_texts(self, texts: Sequence[str]) -> list[list[str]]:
        iterator: Iterable = self.nlp.pipe(texts, batch_size=self.batch_size)
        if self.verbose:
            tqdm, _ = get_tqdm()
            iterator = tqdm(
                iterator,
                total=len(texts),
                desc="Dictionary scoring with spaCy",
                **self.progress_kwargs,
            )
        return [self._preprocess_doc(doc) for doc in iterator]

    def classify(self, tokens: Sequence[str]) -> dict[str, int]:
        return {
            label: sum(token in keyword_set for token in tokens)
            for label, keyword_set in self.dictionaries.items()
        }

    def score_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Return a new DataFrame with per-dictionary match counts."""
        if df.empty:
            raise ValueError("DataFrame is empty.")
        if text_col not in df.columns:
            raise ValueError(f"Column {text_col!r} not found in DataFrame.")

        texts = df[text_col].fillna("").astype(str).tolist()
        token_lists = self.preprocess_texts(texts)
        score_rows = [self.classify(tokens) for tokens in token_lists]

        score_df = pd.DataFrame(score_rows, index=df.index)
        if self.store_tokens:
            score_df[f"preprocessed_{text_col}"] = token_lists

        return df.copy().join(score_df)
