"""Order Block (OB) detection.

This module identifies simple bullish/bearish order blocks using deterministic
price action rules.

Intended use in the framework:
- 1H: locate potential institutional order blocks as setup zones.
"""

import pandas as pd


def identify_order_blocks_clean(df: pd.DataFrame):
    """Identify candidate bullish and bearish order blocks.

    A candle is treated as an OB candidate when it is the opposite color of the
    subsequent directional impulse:
    - Bullish OB: bearish candle that precedes a bullish impulse.
    - Bearish OB: bullish candle that precedes a bearish impulse.

    Args:
        df: OHLCV DataFrame in chronological order.

    Returns:
        List of zones, each a dict with keys: `index`, `type`, `low`, `high`.
        - `type` is `BULL` or `BEAR`.
        - Bounds are based on candle body (ICT-style) for bullish OB, and body
          + wick for bearish OB in this implementation.
    """
    obs = []

    for i in range(1, len(df) - 3):
        c = df.iloc[i]

        # Bullish OB
        if c["close"] < c["open"]:
            if caused_directional_impulse(df, i, "BULL"):
                obs.append(
                    {
                        "index": i,
                        "type": "BULL",
                        "low": float(c["low"]),
                        "high": float(c["open"]),  # body only (ICT style)
                    }
                )

        # Bearish OB
        if c["close"] > c["open"]:
            if caused_directional_impulse(df, i, "BEAR"):
                obs.append(
                    {
                        "index": i,
                        "type": "BEAR",
                        "low": float(c["open"]),
                        "high": float(c["high"]),
                    }
                )

    return obs


def filter_fresh_order_blocks(df: pd.DataFrame, ob_list: list):
    """Filter order blocks to those not yet invalidated by price.

    A bullish OB is invalidated if any subsequent close goes below its low.
    A bearish OB is invalidated if any subsequent close goes above its high.

    Args:
        df: OHLCV DataFrame.
        ob_list: List of OB dicts as returned by `identify_order_blocks_clean`.

    Returns:
        List of OB dicts that remain "fresh" (not violated).
    """
    fresh = []

    for ob in ob_list:
        violated = False
        for i in range(ob["index"] + 1, len(df)):
            close = float(df.iloc[i]["close"])

            if ob["type"] == "BULL" and close < ob["low"]:
                violated = True
                break
            if ob["type"] == "BEAR" and close > ob["high"]:
                violated = True
                break

        if not violated:
            fresh.append(ob)

    return fresh


def is_displacement(candle, min_body_ratio=0.6):
    """Check whether a candle is a displacement candle.

    Args:
        candle: A row-like object with `open`, `high`, `low`, `close`.
        min_body_ratio: Minimum body/range ratio.

    Returns:
        True if candle body/range >= `min_body_ratio` and range > 0.
    """
    body = abs(candle["close"] - candle["open"])
    range_ = candle["high"] - candle["low"]
    if range_ <= 0:
        return False
    return (body / range_) >= min_body_ratio


def is_directional_displacement(
    candle,
    direction: str,
    min_body_ratio: float = 0.6,
) -> bool:
    """Check whether a candle is a directional displacement candle.

    Args:
        candle: A row-like object with `open`, `high`, `low`, `close`.
        direction: `BULL` or `BEAR`.
        min_body_ratio: Minimum body/range ratio.

    Returns:
        True if the candle is a displacement candle and closes in the
        requested direction.
    """
    if not is_displacement(candle, min_body_ratio):
        return False

    if direction == "BULL":
        return candle["close"] > candle["open"]
    if direction == "BEAR":
        return candle["close"] < candle["open"]

    return False


def caused_directional_impulse(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    lookahead: int = 3,
    min_body_ratio: float = 0.6,
) -> bool:
    """Test whether candle `idx` leads to a directional impulse.

    A directional impulse is defined as a displacement candle within a short
    lookahead window that breaks above the base candle high (bullish) or below
    the base candle low (bearish).

    Args:
        df: OHLCV DataFrame.
        idx: Index of the base candle in `df`.
        direction: `BULL` or `BEAR`.
        lookahead: Max number of candles to look forward.
        min_body_ratio: Minimum body/range ratio for the impulse candle.

    Returns:
        True if an impulse is found, otherwise False.
    """
    base = df.iloc[idx]

    for j in range(idx + 1, min(idx + 1 + lookahead, len(df))):
        c = df.iloc[j]

        if not is_displacement(c, min_body_ratio):
            continue

        if direction == "BULL" and c["close"] > base["high"]:
            return True
        if direction == "BEAR" and c["close"] < base["low"]:
            return True

    return False
