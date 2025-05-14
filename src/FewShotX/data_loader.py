import pandas as pd
import importlib.resources as pkg_resources
from . import datasets

def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load example dataset from the FewShotX package.
    
    Example:
    df = load_dataset("econland_corpus.parquet")
    """
    try:
        # Open the parquet file in binary mode
        with pkg_resources.files(datasets).joinpath(filename).open("rb") as f:
            return pd.read_parquet(f)
    except FileNotFoundError:
        raise ValueError(f"Dataset {filename} not found in FewShotX/datasets")