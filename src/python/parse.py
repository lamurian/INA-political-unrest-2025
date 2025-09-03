## MODULES

import os
import json
import pandas as pd
from pydantic import (BaseModel, RootModel)
from typing import List


# Serialize any object
def serialize(obj):
    if isinstance(obj, BaseModel):
        return serialize(obj.dict())
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(x) for x in obj]
    else:
        return obj

# Write JSON
def write_json(path, obj):
    obj = serialize(obj)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"JSON is dumped to {path}")
    except (TypeError, ValueError) as e:
        txt_path = os.path.splitext(path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(str(obj))
        print(f"Text file written at {path}")

# Read JSON
def read_json(path):
    with open(path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    return data

# Write dataset
def write_data(obj, path, **kwargs):
    """
    Write data to file. Handles pandas DataFrames (as CSV) or other objects (as JSON or TXT).

    Parameters
    ----------
    obj : object
        Data to save. If a pandas DataFrame, it will be saved as CSV.
        Otherwise, it will be saved as JSON, with TXT fallback if JSON fails.
    path : str
        File path to save the object.
    kwargs : dict
        Extra keyword arguments passed to pandas `to_csv` when writing DataFrames.
    """
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False, **kwargs)
        print(f"Saved DataFrame as CSV: {path}")
    else:
        write_json(path, obj)

# Load dataset
def read_data(path, **kwargs):
    """
    Read a news dataset from a CSV or JSON file and filter out rows with 
    missing `url` and `content`.

    Parameters
    ----------
    path : str
        File path to the dataset. Supported formats: .csv, .json
    **kwargs : dict
        Additional keyword arguments passed to the pandas reader functions.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with non-null `url` or `content`.
    """
    # Check extension
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".csv":
        tbl = pd.read_csv(path, **kwargs)
    elif ext == ".json":
        tbl = read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")

    try:
        tbl_clean = tbl[~tbl["url"].isnull() | ~tbl["content"].isnull()]
    except (KeyError, TypeError) as e:
        tbl_clean = tbl

    return tbl_clean

# Load or create processed dataset
def load_or_create(FUN, path, params = None):
    """
    Load a processed file if it exists; otherwise, create it using a function and save it.

    Parameters
    ----------
    FUN : callable
        Function that takes `tbl` as input and returns a processed DataFrame.
    path : str
        File path to check/save the processed DataFrame.
    params : dict of str -> dict, optional
        Dictionary of keyword arguments for different functions.
        Example:
            {
                "read_data": {"parse_dates": ["pubDateTime"]},
                "FUN": {"some_arg": True}
            }

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    if params is None:
        params = {}

    if os.path.exists(path):
        # Load existing file
        print(f"Loading existing file: {path}")
        df = read_data(path, **params.get("read_data", {}))
    else:
        # Create file
        print(f"File not found. Running {FUN.__name__}() to create it...")
        df = FUN(**params.get(FUN.__name__, {}))
        write_data(df, path, index=False)
        print(f"Saved processed data to {path}")
    
    return df
