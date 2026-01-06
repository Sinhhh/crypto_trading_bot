# data/loader.py

import pandas as pd
from pathlib import Path


def load_ohlcv(file_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.

    Expected columns: 'datetime', 'open', 'high', 'low', 'close', 'volume'
    Index will be set to datetime if present.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {file_path}")

    df = pd.read_csv(path)

    # Ensure required columns
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Parse datetime column if exists
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    # Convert OHLCV to float
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # Sort by datetime just in case
    df = df.sort_index()

    return df
