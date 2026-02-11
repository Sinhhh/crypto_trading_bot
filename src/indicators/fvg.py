"""
Fair Value Gap detection (3-candle logic)
"""

import pandas as pd


def fair_value_gap(df: pd.DataFrame):
    """
    Detect Fair Value Gaps.
    Returns list of dicts:
    {
        'type': 'BULL' or 'BEAR',
        'low': float,
        'high': float,
        'index': int
    }
    """
    fvg_list = []

    for i in range(2, len(df)):
        # Bullish FVG
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            fvg_list.append(
                {
                    "type": "BULL",
                    "low": df["high"].iloc[i - 2],
                    "high": df["low"].iloc[i],
                    "index": i,
                }
            )

        # Bearish FVG
        if df["high"].iloc[i] < df["low"].iloc[i - 2]:
            fvg_list.append(
                {
                    "type": "BEAR",
                    "low": df["high"].iloc[i],
                    "high": df["low"].iloc[i - 2],
                    "index": i,
                }
            )

    return fvg_list
