"""Shared helper functions used across indicators and strategies.

This module intentionally keeps small, deterministic utilities that are reused
by:
- indicators (FVG, liquidity)
- strategies (entry/zone handling)

All helpers are rule-based and operate on price data (OHLCV) only.
"""

import pandas as pd

from utils.math import _atr


# -------------------------
# Risk/target helpers
# -------------------------
RR_MULT = 2.0
USE_ATR_STOP = True
ATR_PERIOD = 20
ATR_MULT = 1.0


def _recent_window(df: pd.DataFrame, max_window: int = 120) -> pd.DataFrame:
    """Return a tail window of the DataFrame (used to limit computations).

    Args:
        df: OHLCV DataFrame.
        max_window: Max number of rows to keep.

    Returns:
        The original DataFrame if short enough, otherwise `df.tail(max_window)`.
    """
    return df.tail(max_window) if df is not None and len(df) > max_window else df


def _compute_entry_levels(df_15m: pd.DataFrame, bias: str):
    """Compute entry, stop, and target levels from the most recent 15M candle.

    This is a simple deterministic sizing helper:
    - entry is the most recent close
    - stop is the most recent low (BUY) or high (SELL)
    - target is derived from stop distance and `RR_MULT`

    If `USE_ATR_STOP` is enabled and enough data is present, the stop is widened
    based on ATR.

    Args:
        df_15m: 15M OHLCV DataFrame.
        bias: `BUY` or `SELL`.

    Returns:
        Tuple `(entry, stop, target)` or `(None, None, None)` if unavailable.
    """
    if df_15m is None or len(df_15m) < 1:
        return None, None, None
    recent = df_15m.iloc[-1]
    entry = float(recent["close"])
    stop = float(recent["low"] if bias == "BUY" else recent["high"])
    df_recent = _recent_window(df_15m)
    if USE_ATR_STOP:
        atr_value = _atr(df_recent, ATR_PERIOD)
        if atr_value is not None:
            dist = ATR_MULT * atr_value
            stop = min(stop, entry - dist) if bias == "BUY" else max(stop, entry + dist)
    risk = abs(entry - stop)
    target = entry + RR_MULT * risk if bias == "BUY" else entry - RR_MULT * risk
    return entry, stop, target


# -------------------------
# Zone helpers
# -------------------------
def _candle_overlaps_zone(candle, zone_low: float, zone_high: float) -> bool:
    """Return True when the candle range overlaps a zone range."""
    return candle["low"] <= zone_high and candle["high"] >= zone_low


def parse_zone(zone):
    """Normalize a zone definition into `(low, high, type)`.

    Args:
        zone: Either a dict with low/high/type-like keys, or a tuple/list format.

    Returns:
        Tuple `(low, high, typ)` where typ is typically `BULL` or `BEAR`.

    Raises:
        ValueError: If the zone cannot be parsed.
    """
    if isinstance(zone, dict):
        typ = zone.get("type") or zone.get("direction")
        low = zone.get("low") or zone.get("start") or zone.get("min")
        high = zone.get("high") or zone.get("end") or zone.get("max")
        if low is None or high is None:
            raise ValueError(zone)
        return float(min(low, high)), float(max(low, high)), typ
    if isinstance(zone, (list, tuple)):
        if len(zone) >= 4 and isinstance(zone[1], str) and zone[1] in ("BULL", "BEAR"):
            return float(min(zone[3], zone[2])), float(max(zone[3], zone[2])), zone[1]
        if len(zone) >= 3 and isinstance(zone[2], str) and zone[2] in ("BULL", "BEAR"):
            return float(min(zone[0], zone[1])), float(max(zone[0], zone[1])), zone[2]
    raise ValueError(zone)


def slice_lookback(df, current_ts, lookback):
    return df[df["timestamp"] <= current_ts].tail(lookback)


def calculate_atr(df, period=14):
    tr = df["high"] - df["low"]
    atr = tr.rolling(period).mean()
    return atr.iloc[-1] if not atr.empty else 0.0
