"""
Liquidity detection
"""

import pandas as pd


def _most_recent_swing_low(lows) -> float | None:
    """Return the most recent local swing low value from a 1D array-like."""
    if len(lows) < 3:
        return None
    for i in range(len(lows) - 2, 0, -1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            return float(lows[i])
    return None


def _most_recent_swing_high(highs) -> float | None:
    """Return the most recent local swing high value from a 1D array-like."""
    if len(highs) < 3:
        return None
    for i in range(len(highs) - 2, 0, -1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            return float(highs[i])
    return None


def detect_liquidity_grab(
    df: pd.DataFrame,
    bias: str,
    lookback: int = 20,
    sweep_window: int = 5,
    max_window: int | None = 120,
) -> bool:
    """
    Detect a simple liquidity grab (stop-hunt sweep) followed by a reclaim.
    - BUY bias: sweep prior lows, then reclaim above the swept reference.
    - SELL bias: sweep prior highs, then reclaim below the swept reference.
    """
    if max_window is not None and max_window > 0 and len(df) > max_window:
        # Keep the most recent candles only; older data is irrelevant for an intraday sweep+reclaim.
        df = df.tail(max_window)

    if len(df) < 5:
        return False

    if lookback < 3:
        lookback = 3

    if sweep_window < 1:
        sweep_window = 1

    if len(df) < (lookback + sweep_window + 1):
        # still attempt with what we have
        lookback = max(3, min(lookback, len(df) - 2))
        sweep_window = max(1, min(sweep_window, len(df) - 2))

    current = df.iloc[-1]

    # For each candidate sweep candle in the last `sweep_window` candles before `current`,
    # compare to the prior `lookback` candles.
    start_pos = max(1, len(df) - (sweep_window + 1))
    end_pos = len(df) - 1  # exclude current candle
    for sweep_pos in range(start_pos, end_pos):
        sweep_candle = df.iloc[sweep_pos]
        prior_start = max(0, sweep_pos - lookback)
        prior = df.iloc[prior_start:sweep_pos]
        if prior.empty:
            continue

        if bias == "BUY":
            prior_lows = prior["low"].values
            swing_low = _most_recent_swing_low(prior_lows)
            ref = swing_low if swing_low is not None else float(prior["low"].min())
            swept = float(sweep_candle["low"]) < ref
            reclaimed = float(current["close"]) > ref
            if swept and reclaimed:
                return True

        if bias == "SELL":
            prior_highs = prior["high"].values
            swing_high = _most_recent_swing_high(prior_highs)
            ref = swing_high if swing_high is not None else float(prior["high"].max())
            swept = float(sweep_candle["high"]) > ref
            reclaimed = float(current["close"]) < ref
            if swept and reclaimed:
                return True

    return False
