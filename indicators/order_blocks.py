"""
Order Block detection:
- Bullish and Bearish order blocks
"""

import pandas as pd


def is_displacement(candle, min_body_ratio=0.6):
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


def identify_order_blocks_clean(df: pd.DataFrame):
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
