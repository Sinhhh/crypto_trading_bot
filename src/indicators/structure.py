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


def detect_market_structure(df: pd.DataFrame) -> str:
    """
    Returns: 'UP', 'DOWN', 'SIDEWAY'
    Deterministic HH/HL vs LH/LL using the last two swing highs and lows.
    """
    swing_highs, swing_lows = _find_swings(df, left=1, right=1)

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


def detect_bos_choch(df: pd.DataFrame):
    """
    Detect Break of Structure (BOS) and Change of Character (CHOCH) on the 1H timeframe.

    Simplified SMC rules:
    - BOS: price closes beyond the previous swing high/low.
    - CHOCH: a potential character change when the break occurs after opposite momentum.

    Args:
        df: OHLC DataFrame (requires at least 3 candles).

    Returns:
        bos: bool
        choch: bool
    """
    if len(df) < 3:
        return False, False

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    prev_prev_candle = df.iloc[-3]

    # Define a simple prior swing high/low reference
    swing_high = max(prev_prev_candle["high"], prev_candle["high"])
    swing_low = min(prev_prev_candle["low"], prev_candle["low"])

    # BOS/CHOCH evaluation
    bos = False
    choch = False

    # If price closes above prior swing high
    if last_candle["close"] > swing_high:
        bos = True
        # CHOCH heuristic: previous candle momentum was down
        if prev_candle["close"] < prev_prev_candle["close"]:
            choch = True

    # If price closes below prior swing low
    elif last_candle["close"] < swing_low:
        bos = True
        # CHOCH heuristic: previous candle momentum was up
        if prev_candle["close"] > prev_prev_candle["close"]:
            choch = True

    return bos, choch
