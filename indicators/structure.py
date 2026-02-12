import pandas as pd


def _find_swings(df: pd.DataFrame, left: int = 1, right: int = 1):
    """Find swing highs/lows using simple local extrema.

    Returns:
        swing_highs: list[tuple[index, price]]
        swing_lows: list[tuple[index, price]]
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
    """
    Tìm swing highs/lows robust hơn.

    Args:
        df: OHLC DataFrame
        left, right: số nến xem xét bên trái/phải
        min_range_ratio: tỷ lệ tối thiểu so với ATR để swing được coi là quan trọng

    Returns:
        swing_highs: list of tuples (index, price)
        swing_lows: list of tuples (index, price)
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


def detect_market_structure(df: pd.DataFrame) -> str:
    """
    Returns: 'UP', 'DOWN', 'SIDEWAY'
    Deterministic HH/HL vs LH/LL using the last two swing highs and lows.
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


def get_last_structure_levels(df: pd.DataFrame):
    """
    Returns the most recent confirmed swing high and swing low.
    """
    swing_highs, swing_lows = _find_swings(df, left=1, right=1)

    last_high = swing_highs[-1][1] if swing_highs else None
    last_low = swing_lows[-1][1] if swing_lows else None

    return last_high, last_low


def detect_bos_choch_clean(df: pd.DataFrame, trend: str):
    """
    Detect BOS and CHOCH using real market structure.

    Args:
        trend: "UP", "DOWN", or "SIDEWAY" (from detect_market_structure)

    Returns:
        bos: bool
        choch: bool
        break_direction: "BULL" | "BEAR" | None
    """
    if len(df) < 5:
        return False, False, None

    last_close = float(df.iloc[-1]["close"])
    last_high, last_low = get_last_structure_levels(df)

    if last_high is None or last_low is None:
        return False, False, None

    bos = False
    choch = False
    direction = None

    # Break above structure high
    if last_close > last_high:
        direction = "BULL"
        if trend == "UP":
            bos = True
        elif trend == "DOWN":
            choch = True

    # Break below structure low
    elif last_close < last_low:
        direction = "BEAR"
        if trend == "DOWN":
            bos = True
        elif trend == "UP":
            choch = True

    return bos, choch, direction
