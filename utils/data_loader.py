"""
Data loading utilities.
Handles CSV-based OHLCV data.
"""

import pandas as pd


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_csv_data(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV.
    Ensures correct column naming and sorting.
    """
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Basic validation
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Ensure chronological order
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df.reset_index(drop=True, inplace=True)
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any OHLCV DataFrame (from API or other source)
    Ensures:
    - lowercase columns
    - required columns exist
    - chronological order
    - reset index
    """
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Basic validation
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Sort by timestamp if exists
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df.reset_index(drop=True, inplace=True)
    return df
