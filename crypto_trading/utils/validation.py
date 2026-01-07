from __future__ import annotations

import pandas as pd


REQUIRED_OHLCV_COLS = {"open", "high", "low", "close", "volume"}


def require_ohlcv_columns(df: pd.DataFrame, *, context: str = "OHLCV") -> None:
    missing = REQUIRED_OHLCV_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{context} missing columns: {sorted(missing)}")
