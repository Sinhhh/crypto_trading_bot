import pandas as pd

from src.indicators.structure import detect_market_structure, detect_bos_choch

from src.indicators.order_blocks import identify_order_blocks
from src.indicators.fvg import fair_value_gap
from src.indicators.liquidity import detect_liquidity_grab, detect_liquidity_zones
from src.indicators.supply_demand import detect_supply_demand

from src.utils.candle_utils import (
    is_bullish_engulfing,
    is_bearish_engulfing,
    is_hammer,
    is_bearish_pinbar,
    is_inside_bar,
)


# ---------------------------
# Risk/management parameters
# ---------------------------
RR_MULT = 2.0
USE_ATR_STOP = True
ATR_PERIOD = 20
ATR_MULT = 1.0

# Liquidity zone parameters (1H)
LIQ_LOOKBACK_1H = 60
LIQ_MIN_TOUCHES = 1
LIQ_TOLERANCE = 0.0030
LIQ_PROXIMITY = 0.0040


def _recent_window(df: pd.DataFrame, max_window: int = 120) -> pd.DataFrame:
    if df is None:
        return df
    return df.tail(max_window) if len(df) > max_window else df


def _compute_entry_levels(
    df_15m: pd.DataFrame, bias: str
) -> tuple[float | None, float | None, float | None]:
    """Compute candidate (entry, stop, target) from the latest 15M candle.

    - Entry: latest close
    - Stop: structure stop (low/high) optionally extended by ATR min distance
    - Target: RR-based using the final stop
    """
    if df_15m is None or len(df_15m) < 1:
        return None, None, None

    recent = df_15m.iloc[-1]
    entry = float(recent["close"])

    if bias == "BUY":
        stop = float(recent["low"])
    elif bias == "SELL":
        stop = float(recent["high"])
    else:
        return entry, None, None

    df_recent = _recent_window(df_15m, max_window=120)
    if USE_ATR_STOP:
        atr_value = _atr(df_recent, period=ATR_PERIOD)
        if atr_value is not None:
            atr_dist = ATR_MULT * atr_value
            if bias == "BUY":
                stop = min(stop, entry - atr_dist)
            else:
                stop = max(stop, entry + atr_dist)

    risk = abs(entry - stop)
    if bias == "BUY":
        target = entry + RR_MULT * risk
    else:
        target = entry - RR_MULT * risk

    return entry, stop, target


def _atr(df: pd.DataFrame, period: int) -> float | None:
    """Compute ATR (simple moving average of True Range) on the provided OHLCV window."""
    if df is None or len(df) < (period + 1):
        return None

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period, min_periods=period).mean()
    last = atr.iloc[-1]
    if pd.isna(last) or float(last) <= 0:
        return None
    return float(last)


def _candle_overlaps_zone(candle, zone_low: float, zone_high: float) -> bool:
    low = float(candle["low"])
    high = float(candle["high"])
    return (low <= zone_high) and (high >= zone_low)


def parse_zone(zone):
    """Normalize an OB/FVG zone to (low, high, typ).

    Supported formats:
    - Order block tuple: (index, 'BULL'|'BEAR', high, low)
    - Generic tuple/list: (low, high, 'BULL'|'BEAR', ...)
    - FVG dict: {'type': 'BULL'|'BEAR', 'low': x, 'high': y, ...}
    """
    if isinstance(zone, dict):
        typ = zone.get("type") or zone.get("direction")
        low = (
            zone.get("low")
            if zone.get("low") is not None
            else zone.get("start") or zone.get("min")
        )
        high = (
            zone.get("high")
            if zone.get("high") is not None
            else zone.get("end") or zone.get("max")
        )
        if low is None or high is None:
            raise ValueError(f"Invalid zone dict: {zone}")
        low, high = float(min(low, high)), float(max(low, high))
        return low, high, typ

    if not isinstance(zone, (list, tuple)):
        raise TypeError(f"Unsupported zone type: {type(zone)}")

    # Order block format: (index, typ, high, low)
    if len(zone) >= 4 and isinstance(zone[1], str) and zone[1] in ("BULL", "BEAR"):
        typ = zone[1]
        high = float(zone[2])
        low = float(zone[3])
        low, high = float(min(low, high)), float(max(low, high))
        return low, high, typ

    # Generic format: (low, high, typ, ...)
    if len(zone) >= 3 and isinstance(zone[2], str) and zone[2] in ("BULL", "BEAR"):
        low = float(zone[0])
        high = float(zone[1])
        typ = zone[2]
        low, high = float(min(low, high)), float(max(low, high))
        return low, high, typ

    raise ValueError(f"Unsupported zone tuple/list format: {zone}")


# ---------------------------
# 4H BIAS
# ---------------------------
def get_4h_bias(df_4h: pd.DataFrame) -> str:
    structure = detect_market_structure(df_4h)
    supply_zones, demand_zones = detect_supply_demand(df_4h)

    last_close = None
    if df_4h is not None and len(df_4h) > 0:
        last_close = float(df_4h.iloc[-1]["close"])

    def _in_zone(price: float, zone: dict) -> bool:
        return float(zone["low"]) <= price <= float(zone["high"])

    # 4H responsibility: define direction (bias) and key zones (context).
    # Bias must be BUY/SELL only when structure is clear.
    if structure == "UP":
        if last_close is not None:
            for z in supply_zones:
                if _in_zone(last_close, z):
                    return "HOLD"
        return "BUY"
    if structure == "DOWN":
        if last_close is not None:
            for z in demand_zones:
                if _in_zone(last_close, z):
                    return "HOLD"
        return "SELL"
    return "HOLD"


# ---------------------------
# 1H SETUP
# ---------------------------
def get_1h_setup(df_1h: pd.DataFrame, bias: str) -> dict:
    result = {
        "setup_valid": False,
        "structure": None,
        "BOS": False,
        "CHOCH": False,
        "in_OB_FVG": False,
        "liquidity_zone_ok": False,
    }

    if bias == "HOLD" or len(df_1h) < 3:
        return result

    structure = detect_market_structure(df_1h)
    bos, choch = detect_bos_choch(df_1h)

    result["structure"] = structure
    result["BOS"] = bos
    result["CHOCH"] = choch

    # Order Blocks & FVG
    ob_list = identify_order_blocks(df_1h)
    fvg_list = fair_value_gap(df_1h)

    last_candle = df_1h.iloc[-1]
    in_zone = False

    for zone in ob_list + fvg_list:
        low, high, typ = parse_zone(zone)
        if (
            typ == "BULL"
            and bias == "BUY"
            and _candle_overlaps_zone(last_candle, low, high)
        ):
            in_zone = True
            break
        if (
            typ == "BEAR"
            and bias == "SELL"
            and _candle_overlaps_zone(last_candle, low, high)
        ):
            in_zone = True
            break

    result["in_OB_FVG"] = in_zone

    # Liquidity zones (equal highs/lows) on 1H
    liq = detect_liquidity_zones(
        df_1h,
        lookback=LIQ_LOOKBACK_1H,
        min_touches=LIQ_MIN_TOUCHES,
        tolerance=LIQ_TOLERANCE,
    )
    last_close = float(last_candle["close"])

    def _near_level(level: float) -> bool:
        return abs(last_close - float(level)) / max(last_close, 1e-9) <= LIQ_PROXIMITY

    if bias == "BUY":
        result["liquidity_zone_ok"] = any(
            _near_level(z["level"]) for z in liq.get("equal_lows", [])
        )
    elif bias == "SELL":
        result["liquidity_zone_ok"] = any(
            _near_level(z["level"]) for z in liq.get("equal_highs", [])
        )

    # Setup logic (1H): align with 4H bias + structure, require a footprint (BOS or CHOCH) and good location
    footprint_ok = bool(bos or choch)

    if (
        bias == "BUY"
        and structure == "UP"
        and footprint_ok
        and in_zone
        and result["liquidity_zone_ok"]
    ):
        result["setup_valid"] = True
    elif (
        bias == "SELL"
        and structure == "DOWN"
        and footprint_ok
        and in_zone
        and result["liquidity_zone_ok"]
    ):
        result["setup_valid"] = True

    return result


# ---------------------------
# 15M ENTRY
# ---------------------------
def get_15m_entry(
    df_15m: pd.DataFrame, bias: str, setup_valid: bool, ob_fvg_list: list
) -> dict:
    result = {
        "entry_signal": False,
        "entry_price": None,
        "stop": None,
        "target": None,
        "reason": "",
    }

    # Always compute candidate levels when possible (for logging/diagnostics).
    # This does NOT override gating; `entry_signal` remains False unless all rules pass.
    entry_any, stop_any, target_any = _compute_entry_levels(df_15m, bias)
    result["entry_price"] = entry_any
    result["stop"] = stop_any
    result["target"] = target_any

    if not setup_valid:
        result["reason"] = "SETUP_INVALID"
        return result

    if len(df_15m) < 3:
        result["reason"] = "SHORT_15M_DATA"
        return result

    recent = df_15m.iloc[-1]
    prev = df_15m.iloc[-2]
    prev2 = df_15m.iloc[-3]

    # Liquidity grab check (sweep then reclaim leading into entry)
    # Use a recent window to avoid stale history impacting detection in live mode.
    df_15m_recent = _recent_window(df_15m, max_window=120)
    if not detect_liquidity_grab(df_15m_recent, bias=bias):
        result["reason"] = "LIQUIDITY_FAIL"
        return result

    # Candle in OB/FVG
    in_zone = False
    for zone in ob_fvg_list:
        low, high, typ = parse_zone(zone)
        if typ == "BULL" and bias == "BUY" and _candle_overlaps_zone(recent, low, high):
            in_zone = True
            break
        if (
            typ == "BEAR"
            and bias == "SELL"
            and _candle_overlaps_zone(recent, low, high)
        ):
            in_zone = True
            break
    if not in_zone:
        result["reason"] = "CANDLE_NOT_IN_OB_FVG"
        return result

    # Inside-bar handling (15M timing improvement):
    # - If the current candle is an inside bar, wait for a breakout candle.
    # - If the previous candle was an inside bar, allow a breakout of the mother bar as confirmation.
    if is_inside_bar(prev, recent):
        result["reason"] = "WAIT_INSIDE_BAR_BREAKOUT"
        return result

    inside_breakout_ok = False
    if is_inside_bar(prev2, prev):
        if bias == "BUY":
            inside_breakout_ok = bool(
                recent["close"] > prev2["high"] and recent["close"] > recent["open"]
            )
        elif bias == "SELL":
            inside_breakout_ok = bool(
                recent["close"] < prev2["low"] and recent["close"] < recent["open"]
            )

    # Candle confirmation patterns (15M confirmation only; no trend definition here)
    bullish_reclaim_ok = bool(
        recent["close"] > prev["high"] and recent["close"] > recent["open"]
    )
    bearish_reclaim_ok = bool(
        recent["close"] < prev["low"] and recent["close"] < recent["open"]
    )

    if bias == "BUY" and not (
        is_bullish_engulfing(prev, recent)
        or is_hammer(recent)
        or inside_breakout_ok
        or bullish_reclaim_ok
    ):
        result["reason"] = "NO_BULLISH_CONFIRM"
        return result
    if bias == "SELL" and not (
        is_bearish_engulfing(prev, recent)
        or is_bearish_pinbar(recent)
        or inside_breakout_ok
        or bearish_reclaim_ok
    ):
        result["reason"] = "NO_BEARISH_CONFIRM"
        return result

    # Stop / target (reuse the same computation for consistency)
    entry, stop, target = _compute_entry_levels(df_15m, bias)
    if entry is None or stop is None or target is None:
        result["reason"] = "LEVELS_UNAVAILABLE"
        return result

    result.update(
        {
            "entry_signal": True,
            "entry_price": entry,
            "stop": stop,
            "target": target,
            "reason": "TRADE_OK",
        }
    )

    return result


# ---------------------------
# FULL SIGNAL GENERATOR
# ---------------------------
def generate_signal(df_4h, df_1h, df_15m) -> dict:
    bias = get_4h_bias(df_4h)
    setup_1h = get_1h_setup(df_1h, bias)

    # Combine OB/FVG from 1H for 15M entry
    ob_1h = identify_order_blocks(df_1h)
    fvg_1h = fair_value_gap(df_1h)
    ob_fvg_15m = ob_1h + fvg_1h

    entry_15m = get_15m_entry(df_15m, bias, setup_1h["setup_valid"], ob_fvg_15m)

    # Provide both nested and flat keys to keep consumers simple.
    return {
        "bias_4h": bias,
        "setup_1h": setup_1h,
        "entry_15m": entry_15m,
        "bias": bias,
        "setup_valid": bool(setup_1h.get("setup_valid")),
        "entry_signal": bool(entry_15m.get("entry_signal")),
    }
