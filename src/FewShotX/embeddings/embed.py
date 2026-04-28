from __future__ import annotations

from collections.abc import Sequence
from typing import Union
import warnings

import numpy as np
import pandas as pd


class Embeddings:
    """Embed text with a SentenceTransformer model.

    The class keeps the teaching-friendly API used in the notebooks while
    adding a few practical controls around batching, device placement, and
    output handling.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dtype: str = "float32",
        verbose: bool = True,
        device: Union[str, None] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.max_length = self.model.get_max_seq_length()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.dtype = np.dtype(dtype)
        self.verbose = verbose
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def _coerce_texts(self, texts: Union[Sequence[str], str]) -> list[str]:
        if isinstance(texts, str):
            return [texts]
        return ["" if text is None else str(text) for text in texts]

    def _estimate_truncation(self, texts: list[str]) -> list[bool]:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return [len(text) > self.max_length for text in texts]

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        return [len(token_ids) > self.max_length for token_ids in encoded["input_ids"]]

    def embed_text(self, texts: Union[Sequence[str], str]) -> tuple[np.ndarray, list[bool]]:
        """Embed one text or a sequence of texts."""
        text_list = self._coerce_texts(texts)
        truncated_flags = self._estimate_truncation(text_list)

        embeddings = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            show_progress_bar=self.verbose,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings, dtype=self.dtype), truncated_flags

    def embed_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Return a new DataFrame with embedding columns appended."""
        if df.empty:
            raise ValueError("DataFrame is empty.")
        if text_col not in df.columns:
            raise ValueError(f"Column {text_col!r} not found in DataFrame.")

        texts = df[text_col].fillna("").astype(str).tolist()
        embeddings, truncated_flags = self.embed_text(texts)

        if any(truncated_flags) and self.verbose:
            warnings.warn(
                "Some texts exceed the model maximum length and were probably truncated.",
                stacklevel=2,
            )

        emb_df = pd.DataFrame(
            embeddings,
            columns=[f"emb_{i}" for i in range(self.embedding_dim)],
            index=df.index,
        )
        emb_df["probably_truncated"] = truncated_flags

        return df.copy().join(emb_df)
