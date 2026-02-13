"""Liquidity detection utilities.

This module provides deterministic liquidity utilities used by the framework.

Intended use in the framework:
- 1H: detect equal-highs/equal-lows liquidity pools.
- 15M: validate a sweep + reclaim (liquidity grab) aligned with bias.
"""

import pandas as pd

from utils.math import (
    _cluster_levels,
    _most_recent_swing_high,
    _most_recent_swing_low,
)


def detect_liquidity_grab_v2(
    df: pd.DataFrame,
    bias: str,
    lookback: int = 20,
    sweep_window: int = 5,
    max_window: int | None = 120,
):
    """Detect a sweep + reclaim liquidity grab pattern.

    For BUY bias:
    - Find a sweep candle in the recent window whose low breaks a prior swing
      low / prior min low (liquidity sweep), then require current close to be
      back above the swept reference (reclaim).

    For SELL bias:
    - Symmetric logic for highs.

    Args:
        df: OHLCV DataFrame in chronological order.
        bias: Trade direction context. Expected values: `BUY` or `SELL`.
        lookback: Number of prior candles to define the reference liquidity.
        sweep_window: How many candles (excluding the current candle) are
            searched for a sweep.
        max_window: If set, only the most recent `max_window` candles are used.

    Returns:
        Tuple of:
        - grabbed (bool): True when a sweep + reclaim is detected.
        - sweep_index (int | None): Position (0-based) in the (possibly sliced)
          df where the sweep candle occurred.
        - ref_level (float | None): Liquidity reference price that was swept.
    """

    if max_window is not None and max_window > 0 and len(df) > max_window:
        df = df.tail(max_window).copy()

    if len(df) < 5:
        return False, None, None

    lookback = max(3, lookback)
    sweep_window = max(1, sweep_window)

    if len(df) < (lookback + sweep_window + 1):
        lookback = max(3, min(lookback, len(df) - 2))
        sweep_window = max(1, min(sweep_window, len(df) - 2))

    current = df.iloc[-1]

    start_pos = max(1, len(df) - (sweep_window + 1))
    end_pos = len(df) - 1  # exclude current candle

    for sweep_pos in range(start_pos, end_pos):
        sweep_candle = df.iloc[sweep_pos]
        prior = df.iloc[max(0, sweep_pos - lookback) : sweep_pos]
        if prior.empty:
            continue

        if bias == "BUY":
            prior_lows = prior["low"].values
            swing_low = _most_recent_swing_low(prior_lows)
            ref = float(swing_low if swing_low is not None else prior["low"].min())

            swept = float(sweep_candle["low"]) < ref
            reclaimed = float(current["close"]) > ref

            if swept and reclaimed:
                return True, sweep_pos, ref

        if bias == "SELL":
            prior_highs = prior["high"].values
            swing_high = _most_recent_swing_high(prior_highs)
            ref = float(swing_high if swing_high is not None else prior["high"].max())

            swept = float(sweep_candle["high"]) > ref
            reclaimed = float(current["close"]) < ref

            if swept and reclaimed:
                return True, sweep_pos, ref

    return False, None, None


def detect_liquidity_zones(
    df: pd.DataFrame,
    lookback: int = 60,
    min_touches: int = 2,
    tolerance: float = 0.0015,
) -> dict:
    """Detect simple equal-highs/equal-lows liquidity pools.

    Liquidity pools are approximated by clustering recent highs and lows within
    a relative tolerance.

    Args:
        df: OHLCV DataFrame in chronological order.
        lookback: Number of candles used to find clusters.
        min_touches: Minimum number of touches (cluster members) to consider a
            cluster a liquidity zone.
        tolerance: Relative tolerance for clustering (e.g. 0.0015 = 0.15%).

    Returns:
        Dict with keys:
        - `equal_highs`: list of clusters near equal highs.
        - `equal_lows`: list of clusters near equal lows.
        Each cluster is a dict: `{level, indices, min, max}`.
    """
    if df is None or df.empty:
        return {"equal_highs": [], "equal_lows": []}

    if lookback < 5:
        lookback = min(5, len(df))

    df_recent = df.tail(lookback)
    highs = df_recent["high"].astype(float).values
    lows = df_recent["low"].astype(float).values
    idxs = df_recent.index.values

    high_clusters = _cluster_levels(highs, idxs, tolerance)
    low_clusters = _cluster_levels(lows, idxs, tolerance)

    equal_highs = [c for c in high_clusters if len(c["indices"]) >= min_touches]
    equal_lows = [c for c in low_clusters if len(c["indices"]) >= min_touches]

    return {"equal_highs": equal_highs, "equal_lows": equal_lows}
