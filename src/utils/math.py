"""Math/quant helper functions used across the trading system.

This module centralizes small numeric utilities (ATR, clustering, equilibrium)
that are reused by indicators and strategies.
"""

import pandas as pd


def _atr(df: pd.DataFrame, period: int) -> float | None:
    """Compute a simple ATR (Average True Range) value.

    Args:
        df: OHLCV DataFrame in chronological order.
        period: Rolling window length.

    Returns:
        The most recent ATR value, or None if unavailable.
    """
    if df is None or len(df) < (period + 1):
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    last = atr.iloc[-1]
    return float(last) if pd.notna(last) and last > 0 else None


def _cluster_levels(values, indices, tolerance: float):
    """Group numeric values into relative-tolerance clusters.

    Args:
        values: Sequence of numeric values.
        indices: Sequence of corresponding indices.
        tolerance: Relative tolerance (e.g. 0.0015 = 0.15%).

    Returns:
        List of cluster dicts: `{level, indices, min, max}`.
    """
    clusters: list[dict] = []
    for idx, val in zip(indices, values):
        v = float(val)
        placed = False
        for c in clusters:
            threshold = max(abs(c["level"]), 1e-6) * tolerance
            if abs(v - c["level"]) <= threshold:
                c["indices"].append(int(idx))
                c["min"] = min(c["min"], v)
                c["max"] = max(c["max"], v)
                c["level"] = (c["min"] + c["max"]) / 2.0
                placed = True
                break
        if not placed:
            clusters.append({"level": v, "indices": [int(idx)], "min": v, "max": v})
    return clusters


def _most_recent_swing_low(lows) -> float | None:
    """Return the most recent local swing low from a 1D array-like."""
    if len(lows) < 3:
        return None
    for i in range(len(lows) - 2, 0, -1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            return float(lows[i])
    return None


def _most_recent_swing_high(highs) -> float | None:
    """Return the most recent local swing high from a 1D array-like."""
    if len(highs) < 3:
        return None
    for i in range(len(highs) - 2, 0, -1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            return float(highs[i])
    return None


def in_equilibrium(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
    tolerance: float = 0.15,
) -> bool:
    """Return True when the candle close is near the range midpoint.

    Equilibrium is computed as the midpoint of the high/low over a lookback
    window, and the close is considered "in equilibrium" if it is within
    `tolerance` proportion of the total range.

    Args:
        df: OHLCV DataFrame.
        idx: Index of the candle to evaluate.
        lookback: Lookback window length.
        tolerance: Allowed normalized distance from equilibrium.

    Returns:
        True if close is near equilibrium, else False.
    """
    start = max(0, idx - lookback)
    segment = df.iloc[start : idx + 1]

    hi = segment["high"].max()
    lo = segment["low"].min()
    eq = (hi + lo) / 2

    price = df.iloc[idx]["close"]
    range_ = hi - lo

    if range_ == 0:
        return True

    return abs(price - eq) / range_ <= tolerance
