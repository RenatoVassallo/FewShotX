import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import logging
logging.set_verbosity_error()  # suppresses warnings

class ZeroShotLearner:
    """
    ZeroShot classifier that scores similarity between embedded texts
    and label prompts using cosine or dot product similarity.

    Parameters
    ----------
    model : SentenceTransformer
        The pre-trained SentenceTransformer model used to embed labels.
    dtype : str
        The data type for numerical computations (e.g., 'float32').
    similarity : str
        Similarity metric to use: 'cosine' or 'dot'.
    """

    def __init__(self, model: SentenceTransformer, dtype: str = 'float32', similarity: str = 'cosine'):
        assert similarity in ['cosine', 'dot'], "similarity must be 'cosine' or 'dot'"
        self.model = model
        self.embedding_dim = model.get_sentence_embedding_dimension()
        self.dtype = dtype
        self.similarity = similarity

    def _embed_labels(self, labels: list[str]) -> np.ndarray:
        """
        Embed label prompts using the model.

        Parameters
        ----------
        labels : list of str
            Label or class prompts (e.g., "positive", "sports", etc.)

        Returns
        -------
        np.ndarray
            Embedding vectors of labels.
        """
        return np.array(self.model.encode(labels), dtype=self.dtype)

    def _compute_similarity(self, text_embs: np.ndarray, label_embs: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Compute similarity matrix between texts and labels.

        Parameters
        ----------
        text_embs : np.ndarray
            Embeddings for text entries.
        label_embs : np.ndarray
            Embeddings for label prompts.
        normalize : bool
            If True, normalize similarity scores to [0, 1].

        Returns
        -------
        np.ndarray
            Similarity matrix (shape: num_texts x num_labels).
        """
        if self.similarity == 'cosine':
            sim_matrix = cosine_similarity(text_embs, label_embs)
            if normalize:
                sim_matrix = (sim_matrix + 1) / 2  # scale to [0, 1]

        elif self.similarity == 'dot':
            sim_matrix = np.dot(text_embs, label_embs.T)
            if normalize:
                # Apply min-max scaling per column
                sim_min = sim_matrix.min(axis=0, keepdims=True)
                sim_max = sim_matrix.max(axis=0, keepdims=True)
                sim_matrix = (sim_matrix - sim_min) / (sim_max - sim_min + 1e-8)

        return sim_matrix
    
    def score_df(self, df: pd.DataFrame, text_embedding_cols: list[str], labels: list[str], 
                 label_names: list[str] = None, normalize: bool = True) -> pd.DataFrame:
        """
        Score similarity between text embeddings and labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing embedded texts.
        text_embedding_cols : list of str
            Column names corresponding to text embeddings.
        labels : list of str
            Label prompts for zero-shot classification.
        label_names : list of str, optional
            Custom names for score columns (defaults to score_1, score_2, etc).
        normalize : bool
            If True, normalize similarity scores to [0, 1].

        Returns
        -------
        pd.DataFrame
            DataFrame with added score columns (one per label).
        """
        if label_names is None:
            label_names = [f"score_{i+1}" for i in range(len(labels))]

        text_embs = df[text_embedding_cols].values.astype(self.dtype)
        label_embs = self._embed_labels(labels)
        sim_matrix = self._compute_similarity(text_embs, label_embs, normalize)

        for i, name in enumerate(label_names):
            df[name] = sim_matrix[:, i]

        return df
    

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class ZeroShotNLI:
    """
    A simple Zero-Shot classifier using Natural Language Inference (NLI).
    
    It uses a model like BART or RoBERTa trained on MNLI to determine how likely
    a piece of text (premise) entails a hypothesis (constructed from each label).
    
    Parameters
    ----------
    model_name : str
        The name of the pre-trained model to use (e.g., 'facebook/bart-large-mnli', 'roberta-large-mnli').
    hypothesis_template : str
        Template for the hypothesis. The label will be inserted into this template.
        Default is "This example is {}.".
    """

    def __init__(self, model_name='facebook/bart-large-mnli', hypothesis_template="This example is {}."):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.hypothesis_template = hypothesis_template
        self.entailment_id = self._get_entailment_index()

    def _get_entailment_index(self):
        # For most MNLI models: [contradiction, neutral, entailment]
        # Returns 2 if entailment is the last
        return self.model.config.label2id.get("entailment", 2)

    def classify(self, text, candidate_labels, multi_label=False):
        """
        Perform zero-shot classification using a Natural Language Inference (NLI) model.

        This method reformulates each candidate label as a hypothesis and computes the likelihood 
        that the input text (premise) entails each hypothesis. It supports both single-label 
        (multi-class) and multi-label classification modes.

        Parameters
        ----------
        text : str
            The input text to classify (used as the premise in the NLI formulation).
        candidate_labels : List[str]
            A list of labels, each of which will be turned into a natural language hypothesis.
        multi_label : bool, optional (default=False)
            If False (multi-class), the method assumes only one label can be true and applies 
            softmax across labels so that scores sum to 1.
            If True (multi-label), each label is evaluated independently and scores do not sum to 1. 
            This allows multiple labels to be simultaneously "true".

        Returns
        -------
        dict
            {
                "sequence": input text,
                "labels": list of labels (sorted by descending score),
                "scores": list of probabilities corresponding to each label
            }
        """
        hypotheses = [self.hypothesis_template.format(label) for label in candidate_labels]

        # Tokenize each (premise, hypothesis) pair
        inputs = self.tokenizer([text] * len(hypotheses), hypotheses, return_tensors='pt', padding=True)

        # Run model
        with torch.no_grad():
            logits = self.model(**inputs).logits

        if multi_label or len(candidate_labels) == 1:
            # Use softmax over [contradiction, entailment] for each label
            entail_contr_logits = logits[:, [0, self.entailment_id]]
            probs = torch.softmax(entail_contr_logits, dim=1)
            entail_probs = probs[:, 1]
        else:
            # Use softmax over all entailment logits
            entail_logits = logits[:, self.entailment_id]
            entail_probs = torch.softmax(entail_logits, dim=0)

        return {
            "sequence": text,
            "labels": candidate_labels,
            "scores": entail_probs.tolist()
        }
        
    
    def score_df(self, df: pd.DataFrame, text_col: str, labels: list[str], 
                 label_names: list[str] = None, multi_label: bool = False) -> pd.DataFrame:
        """
        Apply zero-shot classification to each row in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a column of text (premises) to classify.
        text_col : str
            The name of the column containing the text.
        labels : list of str
            List of label strings to classify against.
        label_names : list of str, optional
            Custom names for score columns (defaults to score_1, score_2, etc).
        multi_label : bool
            Whether to treat this as a multi-label task.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added columns: one for each label’s probability.
        """
        if label_names is None:
            label_names = [f"score_{i+1}" for i in range(len(labels))]

        scores_list = []

        for text in tqdm(df[text_col], desc="Scoring with ZeroShotNLI"):
            result = self.classify(text, labels, multi_label=multi_label)
            scores_list.append(result["scores"])

        scores_df = pd.DataFrame(scores_list, columns=label_names)
        return pd.concat([df.reset_index(drop=True), scores_df], axis=1)