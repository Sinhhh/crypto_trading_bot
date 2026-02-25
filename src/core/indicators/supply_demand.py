"""Higher timeframe supply & demand zone detection.

This module identifies simple supply and demand zones from swing highs/lows.

Intended use in the framework:
- 4H: context only (zones + directional bias guardrails).
"""

import pandas as pd


def detect_supply_demand(df: pd.DataFrame) -> tuple:
    """Detect basic supply and demand zones from local swing points.

    Zones are extracted from simple swing highs and swing lows:
    - Supply zone: candle body low to candle high at a swing high.
    - Demand zone: candle low to candle body high at a swing low.

    For simplicity, only the two most recent zones of each type are kept.

    Args:
        df: OHLCV DataFrame sorted in ascending time order.

    Returns:
        Tuple `(supply_zones, demand_zones)` where each is a list of dicts:
        `{index: int, low: float, high: float}`.
    """
    supply_zones: list[dict] = []
    demand_zones: list[dict] = []

    if len(df) < 5:
        return supply_zones, demand_zones  # not enough data

    highs = df["high"].values
    lows = df["low"].values

    # Scan swing highs/lows
    for i in range(1, len(df) - 1):
        candle = df.iloc[i]

        # Swing high: higher than prev and next high
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            body_low = float(min(candle["open"], candle["close"]))
            zone = {
                "index": int(i),
                "low": float(body_low),
                "high": float(candle["high"]),
            }
            supply_zones.append(zone)

        # Swing low: lower than prev and next low
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            body_high = float(max(candle["open"], candle["close"]))
            zone = {
                "index": int(i),
                "low": float(candle["low"]),
                "high": float(body_high),
            }
            demand_zones.append(zone)

    # Keep last 2 zones by time (most recent last)
    if len(supply_zones) > 2:
        supply_zones = supply_zones[-2:]
    if len(demand_zones) > 2:
        demand_zones = demand_zones[-2:]

    return supply_zones, demand_zones
