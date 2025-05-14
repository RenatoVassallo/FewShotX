import pandas as pd
import importlib.resources as pkg_resources

def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load a dataset from the 'datasets' folder within the FewShotX package.
    """
    try:
        # Construct the dataset path dynamically
        dataset_path = pkg_resources.files(__package__ + ".datasets").joinpath(filename)
        with dataset_path.open("rb") as f:
            return pd.read_parquet(f)
    except FileNotFoundError:
        raise ValueError(f"Dataset {filename} not found in FewShotX/datasets")