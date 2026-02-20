"""
Candle utilities for 15M entry confirmation.
Pure price action – no indicators.
"""

import pandas as pd

from utils.helpers import _candle_overlaps_zone


def is_bullish_engulfing(prev, current):
    return (
        current["close"] > current["open"]
        and prev["close"] < prev["open"]
        and current["open"] < prev["close"]
        and current["close"] > prev["open"]
    )


def is_bearish_engulfing(prev, current):
    return (
        current["close"] < current["open"]
        and prev["close"] > prev["open"]
        and current["open"] > prev["close"]
        and current["close"] < prev["open"]
    )


def is_hammer(candle):
    body = abs(candle["close"] - candle["open"])
    lower_wick = (
        candle["open"] - candle["low"]
        if candle["close"] > candle["open"]
        else candle["close"] - candle["low"]
    )
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return lower_wick >= 2 * body and upper_wick <= body


def is_bearish_pinbar(candle):
    body = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return upper_wick >= 2 * body and lower_wick <= body


def is_inside_bar(prev, current):
    return current["high"] <= prev["high"] and current["low"] >= prev["low"]


def is_first_tap_zone(
    df: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    current_idx: int,
    lookback: int = 50,
    wick_tolerance: float = 0.15,
) -> bool:
    """
    Returns True if the current candle is the FIRST time price taps this zone.
    """
    zone_low, zone_high = float(min(zone_low, zone_high)), float(max(zone_low, zone_high))
    zone_height = max(zone_high - zone_low, 0.0)
    start = max(0, current_idx - lookback)
    median_range = None
    if df is not None and len(df) > 0:
        window = df.iloc[start:current_idx]
        if not window.empty:
            median_range = float((window["high"] - window["low"]).median())
    for i in range(start, current_idx):
        candle = df.iloc[i]
        if not _candle_overlaps_zone(candle, zone_low, zone_high):
            continue

        body_low = float(min(candle["open"], candle["close"]))
        body_high = float(max(candle["open"], candle["close"]))
        body_overlaps = body_low <= zone_high and body_high >= zone_low
        if body_overlaps:
            return False

        if zone_height <= 0:
            return False

        overlap = min(float(candle["high"]), zone_high) - max(
            float(candle["low"]), zone_low
        )
        if overlap <= 0:
            continue

        wick_ratio = overlap / zone_height
        if median_range is not None:
            vol_ratio = median_range / max(zone_height, 1e-9)
            adaptive_tol = min(0.30, max(0.10, 0.12 + 0.1 * vol_ratio))
        else:
            adaptive_tol = wick_tolerance

        if wick_ratio <= adaptive_tol:
            continue

        return False
    return True


def is_sweep_htf_liquidity(
    sweep_price: float,
    bias: str,
    htf_liquidity: list,
    tolerance: float = 0.0003,  # tùy instrument
) -> bool:
    """
    Validate that a 15M liquidity sweep actually swept HTF (1H) liquidity.
    """
    for liq in htf_liquidity:
        if bias == "BUY" and liq["type"] == "SELL":
            # price swept below HTF sell-side liquidity
            if sweep_price <= liq["price"] + tolerance:
                return True

        if bias == "SELL" and liq["type"] == "BUY":
            # price swept above HTF buy-side liquidity
            if sweep_price >= liq["price"] - tolerance:
                return True

    return False


def is_displacement(candle, min_body_ratio=0.6):
    body = abs(candle["close"] - candle["open"])
    range_ = candle["high"] - candle["low"]
    if range_ <= 0:
        return False
    return (body / range_) >= min_body_ratio


def is_directional_displacement(
    candle, direction: str, min_body_ratio: float = 0.6
) -> bool:
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
