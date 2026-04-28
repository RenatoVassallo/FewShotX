from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from FewShotX.notebook import get_tqdm


@runtime_checkable
class EncoderLike(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs) -> Any:
        ...

    def get_sentence_embedding_dimension(self) -> int:
        ...


class ZeroShotLearner:
    """Embedding-similarity zero-shot classifier."""

    def __init__(self, model: EncoderLike, dtype: str = "float32", similarity: str = "cosine"):
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be 'cosine' or 'dot'")

        self.model = model
        self.embedding_dim = model.get_sentence_embedding_dimension()
        self.dtype = np.dtype(dtype)
        self.similarity = similarity

    def _embed_labels(self, labels: Sequence[str]) -> np.ndarray:
        if not labels:
            raise ValueError("labels must contain at least one label prompt.")
        return np.asarray(self.model.encode(list(labels), show_progress_bar=False), dtype=self.dtype)

    def _compute_similarity(
        self,
        text_embs: np.ndarray,
        label_embs: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        if self.similarity == "cosine":
            sim_matrix = cosine_similarity(text_embs, label_embs)
            if normalize:
                sim_matrix = (sim_matrix + 1.0) / 2.0
            return sim_matrix

        sim_matrix = np.dot(text_embs, label_embs.T)
        if normalize:
            sim_min = sim_matrix.min(axis=0, keepdims=True)
            sim_max = sim_matrix.max(axis=0, keepdims=True)
            sim_matrix = (sim_matrix - sim_min) / (sim_max - sim_min + 1e-8)
        return sim_matrix

    def score_df(
        self,
        df: pd.DataFrame,
        text_embedding_cols: Sequence[str],
        labels: Sequence[str],
        label_names: Optional[Sequence[str]] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        if df.empty:
            raise ValueError("DataFrame is empty.")

        missing_cols = [col for col in text_embedding_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing embedding columns: {missing_cols}")

        label_list = list(labels)
        if label_names is None:
            label_names = [f"score_{i + 1}" for i in range(len(label_list))]
        elif len(label_names) != len(label_list):
            raise ValueError("label_names must have the same length as labels.")

        text_embs = df[list(text_embedding_cols)].to_numpy(dtype=self.dtype)
        label_embs = self._embed_labels(label_list)
        sim_matrix = self._compute_similarity(text_embs, label_embs, normalize=normalize)

        scores_df = pd.DataFrame(sim_matrix, columns=list(label_names), index=df.index)
        return df.copy().join(scores_df)


class ZeroShotNLI:
    """Zero-shot classifier based on Natural Language Inference."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        hypothesis_template: str = "This example is {}.",
        *,
        device: Optional[str] = None,
        batch_size: int = 16,
        verbose: bool = True,
        progress_kwargs: Optional[dict[str, Any]] = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging

        logging.set_verbosity_error()

        self._torch = torch
        self.model_name = model_name
        self.hypothesis_template = hypothesis_template
        self.batch_size = batch_size
        self.verbose = verbose
        self.progress_kwargs = {
            "dynamic_ncols": True,
            "leave": False,
            "smoothing": 0.05,
            "colour": "#2f6db3",
            **(progress_kwargs or {}),
        }
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=False,
        )
        self.tokenizer.clean_up_tokenization_spaces = False
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.entailment_id = self._get_label_index("entailment", default=2)
        self.contradiction_id = self._get_label_index("contradiction", default=0)

    def _resolve_device(self, device: Optional[str]):
        if device is not None:
            return self._torch.device(device)
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        if hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            return self._torch.device("mps")
        return self._torch.device("cpu")

    def _get_label_index(self, target_label: str, default: int) -> int:
        normalized = {str(label).lower(): idx for label, idx in self.model.config.label2id.items()}
        for label, idx in normalized.items():
            if target_label in label:
                return idx
        return default

    def _predict_batch(
        self,
        texts: Sequence[str],
        candidate_labels: Sequence[str],
        multi_label: bool = False,
    ) -> np.ndarray:
        text_list = [str(text) for text in texts]
        label_list = [str(label) for label in candidate_labels]
        hypotheses = [self.hypothesis_template.format(label) for label in label_list]

        repeated_texts = [text for text in text_list for _ in hypotheses]
        repeated_hypotheses = hypotheses * len(text_list)

        inputs = self.tokenizer(
            repeated_texts,
            repeated_hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with self._torch.no_grad():
            logits = self.model(**inputs).logits

        logits = logits.view(len(text_list), len(label_list), -1)

        if multi_label or len(label_list) == 1:
            entail_contr_logits = logits[:, :, [self.contradiction_id, self.entailment_id]]
            probs = self._torch.softmax(entail_contr_logits, dim=2)[:, :, 1]
        else:
            entail_logits = logits[:, :, self.entailment_id]
            probs = self._torch.softmax(entail_logits, dim=1)

        return probs.detach().cpu().numpy()

    def classify(self, text: str, candidate_labels: Sequence[str], multi_label: bool = False) -> dict[str, Any]:
        label_list = list(candidate_labels)
        if not label_list:
            raise ValueError("candidate_labels must contain at least one label.")

        scores = self._predict_batch([text], label_list, multi_label=multi_label)[0]
        return {
            "sequence": text,
            "labels": label_list,
            "scores": scores.tolist(),
        }

    def score_df(
        self,
        df: pd.DataFrame,
        text_col: str,
        labels: Sequence[str],
        label_names: Optional[Sequence[str]] = None,
        multi_label: bool = False,
    ) -> pd.DataFrame:
        if df.empty:
            raise ValueError("DataFrame is empty.")
        if text_col not in df.columns:
            raise ValueError(f"Column {text_col!r} not found in DataFrame.")

        label_list = list(labels)
        if not label_list:
            raise ValueError("labels must contain at least one label.")

        if label_names is None:
            label_names = [f"score_{i + 1}" for i in range(len(label_list))]
        elif len(label_names) != len(label_list):
            raise ValueError("label_names must have the same length as labels.")

        texts = df[text_col].fillna("").astype(str).tolist()
        score_batches = []
        batch_starts = range(0, len(texts), self.batch_size)
        tqdm, _ = get_tqdm()
        iterator = tqdm(
            batch_starts,
            total=(len(texts) + self.batch_size - 1) // self.batch_size,
            desc="Scoring with ZeroShotNLI",
            disable=not self.verbose,
            **self.progress_kwargs,
        )

        for start in iterator:
            batch_texts = texts[start : start + self.batch_size]
            score_batches.append(self._predict_batch(batch_texts, label_list, multi_label=multi_label))

        score_matrix = np.vstack(score_batches) if score_batches else np.empty((0, len(label_list)))
        scores_df = pd.DataFrame(score_matrix, columns=list(label_names), index=df.index)
        return df.copy().join(scores_df)
