import pandas as pd

from indicators.structure import detect_bos_choch_clean, detect_market_structure
from indicators.order_blocks import (
    filter_fresh_order_blocks,
    identify_order_blocks_clean,
)
from indicators.fvg import fair_value_gap_ict, identify_fvg_clean
from indicators.liquidity import detect_liquidity_grab, detect_liquidity_zones
from indicators.supply_demand import detect_supply_demand

from utils.candle_utils import (
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
    return df.tail(max_window) if df is not None and len(df) > max_window else df


def _compute_entry_levels(df_15m: pd.DataFrame, bias: str):
    if df_15m is None or len(df_15m) < 1:
        return None, None, None
    recent = df_15m.iloc[-1]
    entry = float(recent["close"])
    stop = float(recent["low"] if bias == "BUY" else recent["high"])
    df_recent = _recent_window(df_15m)
    if USE_ATR_STOP:
        atr_value = _atr(df_recent, ATR_PERIOD)
        if atr_value is not None:
            dist = ATR_MULT * atr_value
            stop = min(stop, entry - dist) if bias == "BUY" else max(stop, entry + dist)
    risk = abs(entry - stop)
    target = entry + RR_MULT * risk if bias == "BUY" else entry - RR_MULT * risk
    return entry, stop, target


def _atr(df: pd.DataFrame, period: int) -> float | None:
    if df is None or len(df) < (period + 1):
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    last = atr.iloc[-1]
    return float(last) if pd.notna(last) and last > 0 else None


def _candle_overlaps_zone(candle, zone_low: float, zone_high: float) -> bool:
    return candle["low"] <= zone_high and candle["high"] >= zone_low


def parse_zone(zone):
    if isinstance(zone, dict):
        typ = zone.get("type") or zone.get("direction")
        low = zone.get("low") or zone.get("start") or zone.get("min")
        high = zone.get("high") or zone.get("end") or zone.get("max")
        if low is None or high is None:
            raise ValueError(zone)
        return float(min(low, high)), float(max(low, high)), typ
    if isinstance(zone, (list, tuple)):
        if len(zone) >= 4 and isinstance(zone[1], str) and zone[1] in ("BULL", "BEAR"):
            return float(min(zone[3], zone[2])), float(max(zone[3], zone[2])), zone[1]
        if len(zone) >= 3 and isinstance(zone[2], str) and zone[2] in ("BULL", "BEAR"):
            return float(min(zone[0], zone[1])), float(max(zone[0], zone[1])), zone[2]
    raise ValueError(zone)


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
    """
    Compute 1H setup:
    - Align with 4H bias
    - Require BOS/CHOCH
    - Filter FVG/OB by direction and equilibrium
    - Return only zones valid for 15M execution
    """
    result = {
        "setup_valid": False,
        "structure": None,
        "BOS": False,
        "CHOCH": False,
        "break_dir": None,
        "zones": [],
        "liquidity_zone_ok": False,
    }

    if bias == "HOLD" or df_1h is None or len(df_1h) < 50:
        return result

    # -------------------------
    # Market structure + break
    # -------------------------
    structure = detect_market_structure(df_1h)
    bos, choch, break_dir = detect_bos_choch_clean(df_1h, structure)

    result.update(
        {"structure": structure, "BOS": bos, "CHOCH": choch, "break_dir": break_dir}
    )

    # Direction alignment check
    if (bias == "BUY" and (structure != "UP" or break_dir != "BULL")) or (
        bias == "SELL" and (structure != "DOWN" or break_dir != "BEAR")
    ):
        return result

    # -------------------------
    # Equilibrium / premium/discount
    # -------------------------
    lookback = 100
    range_high = df_1h["high"].rolling(lookback).max().iloc[-1]
    range_low = df_1h["low"].rolling(lookback).min().iloc[-1]
    equilibrium = (range_high + range_low) / 2

    def in_discount(zone):
        mid = (zone["high"] + zone["low"]) / 2
        return mid < equilibrium

    def in_premium(zone):
        mid = (zone["high"] + zone["low"]) / 2
        return mid > equilibrium

    # -------------------------
    # Order Blocks
    # -------------------------
    obs = filter_fresh_order_blocks(df_1h, identify_order_blocks_clean(df_1h))
    obs = [z for z in obs if z["type"] == ("BULL" if bias == "BUY" else "BEAR")]

    # -------------------------
    # Fair Value Gaps
    # -------------------------
    fvgs = identify_fvg_clean(df_1h)
    fvgs = [f for f in fvgs if f["type"] == ("BULL" if bias == "BUY" else "BEAR")]

    # BOS strength filter
    if bos:
        fvgs = [f for f in fvgs if f.get("strength") == "STRONG"]

    # Equilibrium filter
    if bias == "BUY":
        fvgs = [f for f in fvgs if in_discount(f)]
        obs = [z for z in obs if in_discount(z)]
    else:
        fvgs = [f for f in fvgs if in_premium(f)]
        obs = [z for z in obs if in_premium(z)]

    zones = obs + fvgs
    if not zones:
        return result

    result["zones"] = zones

    # -------------------------
    # Liquidity context
    # -------------------------
    liq = detect_liquidity_zones(
        df_1h,
        lookback=LIQ_LOOKBACK_1H,
        min_touches=LIQ_MIN_TOUCHES,
        tolerance=LIQ_TOLERANCE,
    )

    last_close = float(df_1h.iloc[-1]["close"])

    def _near(level: float) -> bool:
        return abs(last_close - float(level)) / max(last_close, 1e-9) <= LIQ_PROXIMITY

    if bias == "BUY":
        result["liquidity_zone_ok"] = any(
            _near(z["level"]) for z in liq.get("equal_lows", [])
        )
    else:
        result["liquidity_zone_ok"] = any(
            _near(z["level"]) for z in liq.get("equal_highs", [])
        )

    # -------------------------
    # Final validation
    # -------------------------
    if result["liquidity_zone_ok"]:
        result["setup_valid"] = True

    return result


# ---------------------------
# 15M ENTRY
# ---------------------------
# ---------------------------
# 15M ENTRY (cleaned)
# ---------------------------
def get_15m_entry(
    df_15m: pd.DataFrame, bias: str, setup_valid: bool, ob_fvg_list: list
) -> dict:
    """
    Compute 15M entry signals based on 1H zones (OB/FVG):
    - Requires 1H setup to be valid
    - Candle must touch a valid OB/FVG
    - Validate liquidity grab
    - Respect inside-bar breakout rules
    - Confirm bullish/bearish candle patterns
    """
    result = {
        "entry_signal": False,
        "entry_price": None,
        "stop": None,
        "target": None,
        "reason": "",
    }

    if not setup_valid:
        result["reason"] = "SETUP_INVALID"
        return result

    if len(df_15m) < 3:
        result["reason"] = "SHORT_15M_DATA"
        return result

    recent = df_15m.iloc[-1]
    prev = df_15m.iloc[-2]
    prev2 = df_15m.iloc[-3]

    # -------------------------
    # Liquidity grab check
    # -------------------------
    df_recent = _recent_window(df_15m, max_window=120)
    if not detect_liquidity_grab(df_recent, bias=bias):
        result["reason"] = "LIQUIDITY_FAIL"
        return result

    # -------------------------
    # Candle overlaps 1H OB/FVG
    # -------------------------
    in_zone = False
    for zone in ob_fvg_list:
        low, high, typ = parse_zone(zone)
        if bias == "BUY" and typ == "BULL" and _candle_overlaps_zone(recent, low, high):
            in_zone = True
            break
        if (
            bias == "SELL"
            and typ == "BEAR"
            and _candle_overlaps_zone(recent, low, high)
        ):
            in_zone = True
            break

    if not in_zone:
        result["reason"] = "CANDLE_NOT_IN_OB_FVG"
        return result

    # -------------------------
    # Inside-bar breakout
    # -------------------------
    if is_inside_bar(prev, recent):
        result["reason"] = "WAIT_INSIDE_BAR_BREAKOUT"
        return result

    inside_breakout_ok = False
    if is_inside_bar(prev2, prev):
        if bias == "BUY":
            inside_breakout_ok = (
                recent["close"] > prev2["high"] and recent["close"] > recent["open"]
            )
        elif bias == "SELL":
            inside_breakout_ok = (
                recent["close"] < prev2["low"] and recent["close"] < recent["open"]
            )

    # -------------------------
    # 15M confirmation patterns
    # -------------------------
    bullish_ok = (
        is_bullish_engulfing(prev, recent)
        or is_hammer(recent)
        or inside_breakout_ok
        or recent["close"] > prev["high"]
    )
    bearish_ok = (
        is_bearish_engulfing(prev, recent)
        or is_bearish_pinbar(recent)
        or inside_breakout_ok
        or recent["close"] < prev["low"]
    )

    if (bias == "BUY" and not bullish_ok) or (bias == "SELL" and not bearish_ok):
        result["reason"] = "NO_CONFIRM_CANDLE"
        return result

    # -------------------------
    # Stop / target computation
    # -------------------------
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
# FULL SIGNAL GENERATOR (cleaned)
# ---------------------------
def generate_signal(df_4h, df_1h, df_15m) -> dict:
    bias = get_4h_bias(df_4h)
    setup_1h = get_1h_setup(df_1h, bias)

    zones_1h = setup_1h.get("zones", [])
    entry_15m = get_15m_entry(df_15m, bias, setup_1h["setup_valid"], zones_1h)

    return {
        "bias_4h": bias,
        "setup_1h": setup_1h,
        "entry_15m": entry_15m,
        "bias": bias,
        "setup_valid": bool(setup_1h.get("setup_valid")),
        "entry_signal": bool(entry_15m.get("entry_signal")),
    }
