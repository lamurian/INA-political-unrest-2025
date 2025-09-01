## MODULES

import os
import pandas as pd


## FUNCTION

# Load dataset
def read_news(path, **kwargs):
    tbl = pd.read_csv(path, **kwargs)
    tbl_clean = tbl[~tbl["url"].isnull() | ~tbl["content"].isnull()]
    return tbl_clean

# Load or create processed dataset
def load_or_create(FUN, tbl, fpath, **kwargs):
    """
    Load a processed file if it exists; otherwise, create it using a function and save it.

    Parameters
    ----------
    FUN : callable
        Function that takes `tbl` as input and returns a processed DataFrame.
    tbl : pd.DataFrame
        Input raw DataFrame.
    fpath : str
        File path to check/save the processed DataFrame.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    if os.path.exists(fpath):
        # Load existing file
        print(f"Loading existing file: {fpath}")
        df = read_news(fpath, **kwargs)
    else:
        # Create file
        print(f"File not found. Running {FUN.__name__}() to create it...")
        df = FUN(tbl)
        df.to_csv(fpath, index=False)
        print(f"Saved processed data to {fpath}")
    
    return df
