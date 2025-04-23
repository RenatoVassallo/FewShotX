import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class Embeddings:
    """
    A simple wrapper for embedding text using a SentenceTransformer model,
    designed for few-shot and NLP teaching applications.

    Parameters
    ----------
    model_name : str
        Name of the pretrained SentenceTransformer model (e.g. 'all-MiniLM-L6-v2').
    dtype : str
        Data type to use for returned embeddings (e.g., 'float32').
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', dtype: str = 'float32'):
        self.model = SentenceTransformer(model_name)
        self.max_length = self.model.get_max_seq_length()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.dtype = dtype

    def embed_text(self, texts: list[str]) -> tuple[np.ndarray, list[bool]]:
        """
        Embed a list of texts using the loaded transformer model.

        Parameters
        ----------
        texts : list of str
            Text data to embed.

        Returns
        -------
        tuple
            - Embedding matrix of shape (n_texts, embedding_dim)
            - List of booleans indicating likely truncation
        """
        # Basic heuristic: longer than model max_seq_length → likely to truncate
        # (max_seq_length is in tokens, this is in characters — not perfect, but practical)
        truncated_flags = [
            isinstance(text, str) and len(text) > self.max_length
            for text in texts
        ]

        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings, dtype=self.dtype), truncated_flags

    def embed_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Embed a DataFrame's text column and return the same DataFrame with embedding columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a column of text to embed.
        text_col : str
            Name of the text column.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with embedding columns and a 'probably_truncated' flag.
        """
        if df.empty or text_col not in df.columns:
            raise ValueError("DataFrame is empty or text column not found.")

        texts = df[text_col].fillna("").astype(str).tolist()
        embeddings, truncated_flags = self.embed_text(texts)
        
        if any(truncated_flags):
            print("Warning: Some texts may have been truncated due to model's max length.")

        emb_df = pd.DataFrame(
            embeddings,
            columns=[f"emb_{i}" for i in range(self.embedding_dim)],
            index=df.index
        )

        emb_df["probably_truncated"] = truncated_flags
        return pd.concat([df, emb_df], axis=1)