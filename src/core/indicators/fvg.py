"""Fair Value Gap (FVG) detection.

This module implements deterministic, price-action-based Fair Value Gap detection
using a 3-candle pattern plus a displacement check.

Intended use in the framework:
- 1H: identify institutional footprint (FVG zones) for setup validation.

Notes:
- No indicators such as RSI/EMA are used.
"""

import pandas as pd

from core.indicators.order_blocks import is_directional_displacement
from utils.math import in_equilibrium


def identify_fvg_clean(
    df: pd.DataFrame,
    min_body_ratio: float = 0.6,
    min_gap_ratio: float = 0.0003,
    equilibrium_filter: bool = True,
):
    """Identify ICT-style Fair Value Gaps using a 3-candle pattern.

    A bullish FVG is detected when the third candle's low is above the first
    candle's high, and the middle candle is a bullish displacement candle.
    A bearish FVG is the inverse.

    Args:
        df: OHLCV data in chronological order. Requires columns: `open`, `high`,
            `low`, `close`.
        min_body_ratio: Minimum displacement candle body-to-range ratio.
        min_gap_ratio: Minimum gap size normalized by the middle candle close.
        equilibrium_filter: If True, labels FVGs as WEAK when the displacement
            candle occurs near equilibrium (see `utils.helpers.in_equilibrium`).

    Returns:
        A list of FVG dicts with keys:
        - `index` (int): index of the displacement candle (middle candle).
        - `type` (str): `BULL` or `BEAR`.
        - `low` (float): lower bound of the gap.
        - `high` (float): upper bound of the gap.
        - `strength` (str): `STRONG` or `WEAK`.
    """
    fvgs = []

    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]  # displacement candle
        c3 = df.iloc[i]

        # -----------------
        # Bullish FVG
        # -----------------
        if is_directional_displacement(c2, "BULL", min_body_ratio):
            if c3["low"] > c1["high"]:
                gap = c3["low"] - c1["high"]
                if gap / c2["close"] >= min_gap_ratio:
                    weak = equilibrium_filter and in_equilibrium(df, i - 1)
                    fvgs.append(
                        {
                            "index": i - 1,
                            "type": "BULL",
                            "low": float(c1["high"]),
                            "high": float(c3["low"]),
                            "strength": "WEAK" if weak else "STRONG",
                        }
                    )

        # -----------------
        # Bearish FVG
        # -----------------
        if is_directional_displacement(c2, "BEAR", min_body_ratio):
            if c3["high"] < c1["low"]:
                gap = c1["low"] - c3["high"]
                if gap / c2["close"] >= min_gap_ratio:
                    weak = equilibrium_filter and in_equilibrium(df, i - 1)
                    fvgs.append(
                        {
                            "index": i - 1,
                            "type": "BEAR",
                            "low": float(c3["high"]),
                            "high": float(c1["low"]),
                            "strength": "WEAK" if weak else "STRONG",
                        }
                    )

    return fvgs
