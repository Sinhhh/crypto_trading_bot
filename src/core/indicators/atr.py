# indicators/atr.py
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR)
    Trả về Series cùng index với df
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr
