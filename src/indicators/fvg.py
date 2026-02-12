"""
Fair Value Gap detection (3-candle logic)
"""

import pandas as pd


def fair_value_gap_ict(
    df: pd.DataFrame,
    min_body_ratio: float = 0.5,
    min_gap_ratio: float = 0.1,
):
    """
    ICT-style Fair Value Gap detection with displacement validation.

    Args:
        min_body_ratio: body / range của displacement candle (>= 0.5 là mạnh)
        min_gap_ratio: gap tối thiểu so với range của displacement candle

    Returns:
        list[dict]: {
            'type': 'BULL' | 'BEAR',
            'low': float,
            'high': float,
            'index': int,
            'strength': float
        }
    """
    fvg_list = []

    if df is None or len(df) < 3:
        return fvg_list

    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]  # candle trước
        c2 = df.iloc[i - 1]  # displacement candle
        c3 = df.iloc[i]      # candle sau

        # Displacement candle stats
        body = abs(c2["close"] - c2["open"])
        range_ = c2["high"] - c2["low"]

        if range_ <= 0:
            continue

        body_ratio = body / range_

        # ❌ Không có displacement → bỏ
        if body_ratio < min_body_ratio:
            continue

        # -----------------------
        # 🟢 Bullish FVG
        # -----------------------
        if c3["low"] > c1["high"]:
            gap = c3["low"] - c1["high"]

            # Gap quá nhỏ → nhiễu
            if gap / range_ < min_gap_ratio:
                continue

            fvg_list.append(
                {
                    "type": "BULL",
                    "low": float(c1["high"]),
                    "high": float(c3["low"]),
                    "index": i,
                    "strength": round(body_ratio, 2),
                }
            )

        # -----------------------
        # 🔴 Bearish FVG
        # -----------------------
        if c3["high"] < c1["low"]:
            gap = c1["low"] - c3["high"]

            if gap / range_ < min_gap_ratio:
                continue

            fvg_list.append(
                {
                    "type": "BEAR",
                    "low": float(c3["high"]),
                    "high": float(c1["low"]),
                    "index": i,
                    "strength": round(body_ratio, 2),
                }
            )

    return fvg_list

