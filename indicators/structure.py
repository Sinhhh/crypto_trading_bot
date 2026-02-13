"""Market structure utilities.

This module extracts deterministic market structure information (trend, BOS,
CHOCH, swing points).

Intended use in the framework:
- 4H: determine directional bias (UP/DOWN/SIDEWAY).
- 1H: validate setup via BOS/CHOCH aligned with 4H bias.
"""

import pandas as pd

from indicators.order_blocks import is_displacement


def detect_market_structure(df: pd.DataFrame) -> str:
    """Detect market structure as UP/DOWN/SIDEWAY.

    Structure is determined by comparing the last two confirmed swing highs and
    swing lows:
    - UP: Higher High and Higher Low
    - DOWN: Lower High and Lower Low
    - SIDEWAY: otherwise / insufficient swings

    Args:
        df: OHLCV DataFrame in chronological order.

    Returns:
        One of: `UP`, `DOWN`, `SIDEWAY`.
    """
    swing_highs, swing_lows = _find_swings_robust(
        df, left=2, right=2, min_range_ratio=0.5
    )

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "SIDEWAY"

    # Last two swings by time (already in index order)
    _, h1 = swing_highs[-2]
    _, h2 = swing_highs[-1]
    _, l1 = swing_lows[-2]
    _, l2 = swing_lows[-1]

    higher_high = h2 > h1
    higher_low = l2 > l1
    lower_high = h2 < h1
    lower_low = l2 < l1

    if higher_high and higher_low:
        return "UP"
    if lower_high and lower_low:
        return "DOWN"
    return "SIDEWAY"


def detect_bos_choch_v2(df: pd.DataFrame, trend: str):
    """Detect BOS/CHOCH relative to the most recent structure levels.

    BOS (break of structure) and CHOCH (change of character) are determined by
    whether the most recent close breaks the last swing high/low, and whether
    that break is aligned with the current `trend`.

    A displacement candle is required for a break to be considered valid.

    Args:
        df: OHLCV DataFrame.
        trend: The higher-level trend context from `detect_market_structure`.

    Returns:
        Tuple of:
        - bos (bool): True when break aligns with trend.
        - choch (bool): True when break opposes trend.
        - break_direction (str | None): `BULL`, `BEAR`, or None.
    """
    if len(df) < 5:
        return False, False, None

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_high, last_low = get_last_structure_levels(df)

    if last_high is None or last_low is None:
        return False, False, None

    bos = False
    choch = False
    direction = None

    displacement = is_displacement(last, min_body_ratio=0.6)

    # Break above structure high
    # BOS chỉ hợp lệ nếu nến phá cấu trúc là displacement candle
    if last_close > last_high and displacement:
        direction = "BULL"
        if trend == "UP":
            bos = True
        elif trend == "DOWN":
            choch = True

    # Break below structure low
    elif last_close < last_low and displacement:
        direction = "BEAR"
        if trend == "DOWN":
            bos = True
        elif trend == "UP":
            choch = True

    return bos, choch, direction


def detect_bos_index(df_1h: pd.DataFrame, structure: str) -> int | None:
    """Find the most recent index where a BOS-like event occurred.

    This helper scans backward and identifies a simplistic "new high" (UP) or
    "new low" (DOWN) against a short recent window.

    Args:
        df_1h: 1H OHLCV DataFrame.
        structure: Expected market structure: `UP` or `DOWN`.

    Returns:
        The most recent index (0-based) where the condition is met, or None.
    """
    if df_1h is None or len(df_1h) < 3:
        return None

    # duyệt từ gần nhất về xa
    for idx in range(len(df_1h) - 1, 2, -1):
        curr = df_1h.iloc[idx]

        if structure == "UP":
            # BOS UP: nến hiện tại tạo high mới so với vài nến trước
            recent_high = df_1h["high"].iloc[idx - 3 : idx].max()
            if curr["high"] > recent_high:
                return idx
        elif structure == "DOWN":
            # BOS DOWN: nến hiện tại tạo low mới so với vài nến trước
            recent_low = df_1h["low"].iloc[idx - 3 : idx].min()
            if curr["low"] < recent_low:
                return idx

    return None


def _find_swings(df: pd.DataFrame, left: int = 1, right: int = 1):
    """Find swing highs/lows using local extrema.

    A swing high is a high that is greater than the highs of `left` candles to
    the left and `right` candles to the right. Swing low is analogous.

    Args:
        df: OHLCV DataFrame.
        left: Number of candles to the left.
        right: Number of candles to the right.

    Returns:
        Tuple `(swing_highs, swing_lows)` where each is a list of `(index, price)`.
    """
    if len(df) < (left + right + 3):
        return [], []

    highs = df["high"].values
    lows = df["low"].values

    swing_highs = []
    swing_lows = []

    for i in range(left, len(df) - right):
        left_high = highs[i - left : i].max()
        right_high = highs[i + 1 : i + right + 1].max()
        if highs[i] > left_high and highs[i] > right_high:
            swing_highs.append((i, float(highs[i])))

        left_low = lows[i - left : i].min()
        right_low = lows[i + 1 : i + right + 1].min()
        if lows[i] < left_low and lows[i] < right_low:
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


def _find_swings_robust(
    df: pd.DataFrame, left: int = 2, right: int = 2, min_range_ratio: float = 0.5
):
    """Find swing highs/lows with an ATR-based significance filter.

    This variant filters swings by requiring the swing move magnitude to be at
    least `min_range_ratio * ATR` at that point.

    Args:
        df: OHLCV DataFrame.
        left: Number of candles to the left.
        right: Number of candles to the right.
        min_range_ratio: Minimum move relative to ATR for swing significance.

    Returns:
        Tuple `(swing_highs, swing_lows)`.
    """
    swing_highs = []
    swing_lows = []

    if len(df) < (left + right + 1):
        return swing_highs, swing_lows

    # Tính ATR để làm ngưỡng biến động
    df = df.copy()
    df["prev_close"] = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(left + right + 1, min_periods=1).mean()

    highs = df["high"].values
    lows = df["low"].values
    atr_values = atr.values

    for i in range(left, len(df) - right):
        # Swing high
        left_high = highs[i - left : i].max() if left > 0 else highs[i]
        right_high = highs[i + 1 : i + right + 1].max() if right > 0 else highs[i]
        if highs[i] > left_high and highs[i] > right_high:
            if (highs[i] - min(left_high, right_high)) >= min_range_ratio * atr_values[
                i
            ]:
                swing_highs.append((i, float(highs[i])))

        # Swing low
        left_low = lows[i - left : i].min() if left > 0 else lows[i]
        right_low = lows[i + 1 : i + right + 1].min() if right > 0 else lows[i]
        if lows[i] < left_low and lows[i] < right_low:
            if (max(left_low, right_low) - lows[i]) >= min_range_ratio * atr_values[i]:
                swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


def get_last_structure_levels(df: pd.DataFrame):
    """Get the most recent confirmed swing high and swing low.

    Args:
        df: OHLCV DataFrame.

    Returns:
        Tuple `(last_high, last_low)`; each may be None if unavailable.
    """
    swing_highs, swing_lows = _find_swings(df, left=1, right=1)

    last_high = swing_highs[-1][1] if swing_highs else None
    last_low = swing_lows[-1][1] if swing_lows else None

    return last_high, last_low
