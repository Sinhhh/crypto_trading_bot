"""
Order Block detection:
- Bullish and Bearish order blocks
"""

import pandas as pd


def identify_order_blocks(df: pd.DataFrame):
    """
    Identify basic order blocks.
    Returns list of tuples:
    (index, type, high, low)
    """
    order_blocks = []

    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # Bullish Order Block
        if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
            order_blocks.append((i - 1, "BULL", prev["high"], prev["low"]))

        # Bearish Order Block
        if prev["close"] > prev["open"] and curr["close"] < curr["open"]:
            order_blocks.append((i - 1, "BEAR", prev["high"], prev["low"]))

    return order_blocks
