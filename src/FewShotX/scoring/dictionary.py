import pandas as pd
from tqdm import tqdm
import spacy

class DictionaryScorer:
    """
    Dictionary-based text classifier using spaCy for tokenization and stop word filtering.

    Parameters
    ----------
    dictionaries : dict
        A dictionary mapping category names to sets/lists of keywords (lowercased).
    model_name : str
        spaCy language model to load (default: 'en_core_web_sm').
    """

    def __init__(self, dictionaries: dict, model_name: str = 'en_core_web_sm'):
        self.dictionaries = {k: set(map(str.lower, v)) for k, v in dictionaries.items()}
        self.nlp = spacy.load(model_name, disable=["ner", "parser"])  # just need tokenization
        self.nlp.max_length = 2000000  # if you're processing large texts

    def _preprocess(self, text: str) -> list:
        """
        Tokenize and clean text (lowercase, remove stopwords and non-alpha).

        Returns
        -------
        list of str: cleaned tokens
        """
        if not isinstance(text, str):
            return []

        doc = self.nlp(text.lower())
        return [token.text for token in doc if token.is_alpha and not token.is_stop]

    def classify(self, tokens: list) -> dict:
        """
        Count how many tokens match each dictionary.
        """
        return {
            label: sum(token in keyword_set for token in tokens)
            for label, keyword_set in self.dictionaries.items()
        }

    def score_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Apply dictionary scoring to each row of a DataFrame.

        Returns
        -------
        pd.DataFrame with dictionary match counts and preprocessed tokens.
        """
        all_scores = []
        all_tokens = []

        for text in tqdm(df[text_col], desc="Dictionary scoring with spaCy"):
            tokens = self._preprocess(text)
            all_tokens.append(tokens)
            score = self.classify(tokens)
            all_scores.append(score)

        scores_df = pd.DataFrame(all_scores)
        scores_df[f"preprocessed_{text_col}"] = all_tokens

        return pd.concat([df.reset_index(drop=True), scores_df], axis=1)