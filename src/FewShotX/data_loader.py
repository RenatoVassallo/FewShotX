import pandas as pd
import importlib.resources as pkg_resources

def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load a dataset from the 'datasets' folder within the FewShotX package.

    Supports both .csv and .parquet file formats.
    """
    try:
        # Construct the dataset path dynamically
        dataset_path = pkg_resources.files(__package__ + ".datasets").joinpath(filename)

        # Determine file format and load accordingly
        if filename.endswith(".parquet"):
            with dataset_path.open("rb") as f:
                return pd.read_parquet(f)
        elif filename.endswith(".csv"):
            with dataset_path.open("r") as f:
                return pd.read_csv(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    except FileNotFoundError:
        raise ValueError(f"Dataset {filename} not found in FewShotX/datasets")