import pandas as pd


def detect_supply_demand(df: pd.DataFrame) -> tuple:
    """
    Detect higher-timeframe Supply & Demand zones from swing highs/lows.

    Simple SMC-inspired rules:
    - Demand zone: a swing low area where price may react upward.
    - Supply zone: a swing high area where price may react downward.
    - Keep only the 2 most recent zones for simplicity.

    Args:
        df: OHLC DataFrame sorted in ascending time order.

    Returns:
        supply_zones: list[dict]  # each: {'index': int, 'low': float, 'high': float}
        demand_zones: list[dict]  # each: {'index': int, 'low': float, 'high': float}
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


def price_in_equilibrium(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2:
        return False

    high = float(df["high"].max())
    low = float(df["low"].min())
    mid = (high + low) / 2
    last_close = float(df.iloc[-1]["close"])

    return abs(last_close - mid) / (high - low + 1e-9) < 0.1
