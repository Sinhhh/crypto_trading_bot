import pandas as pd

from core.indicators.structure import (
    detect_market_structure,
    detect_bos_choch_v2,
    detect_bos_index,
    _find_swings_robust,
    get_last_structure_levels,
)

from core.indicators.order_blocks import (
    identify_order_blocks_clean,
    filter_fresh_order_blocks,
)
from core.indicators.fvg import identify_fvg_clean
from core.indicators.liquidity import (
    detect_liquidity_grab_v2,
    detect_liquidity_zones,
)
from core.indicators.supply_demand import detect_supply_demand

from utils.candle_utils import (
    is_bullish_engulfing,
    is_bearish_engulfing,
    is_first_tap_zone,
    is_hammer,
    is_bearish_pinbar,
    is_inside_bar,
    is_sweep_htf_liquidity,
)
from utils.helpers import (
    _compute_entry_levels,
    _recent_window,
    _candle_overlaps_zone,
    parse_zone,
)


# Liquidity zone parameters (1H)
LIQ_LOOKBACK_1H = 60
LIQ_MIN_TOUCHES = 2
LIQ_TOLERANCE = 0.0030
LIQ_PROXIMITY = 0.0040

# Adaptive filters (price-action only)
MIN_ZONE_PROX_PCT = 0.002
MAX_ZONE_PROX_PCT = 0.008
BOS_BREAK_PCT = 0.05


# ---------------------------
# 4H BIAS
# ---------------------------
def get_4h_bias(df_4h: pd.DataFrame) -> str:
    if df_4h is None or len(df_4h) < 40:
        return "HOLD"

    structure = detect_market_structure(df_4h)
    supply_zones, demand_zones = detect_supply_demand(df_4h)

    # Range compression filter (price-action only)
    recent = df_4h.tail(20)
    prev = df_4h.iloc[-40:-20]
    recent_range = float(recent["high"].max() - recent["low"].min())
    prev_range = float(prev["high"].max() - prev["low"].min())
    if prev_range > 0 and (recent_range / prev_range) < 0.6:
        return "HOLD"

    # Structure strength: require meaningful HH/HL or LH/LL moves
    swing_highs, swing_lows = _find_swings_robust(
        df_4h, left=2, right=2, min_range_ratio=0.5
    )
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "HOLD"

    _, h1 = swing_highs[-2]
    _, h2 = swing_highs[-1]
    _, l1 = swing_lows[-2]
    _, l2 = swing_lows[-1]
    if recent_range > 0:
        if abs(h2 - h1) < 0.3 * recent_range and abs(l2 - l1) < 0.3 * recent_range:
            return "HOLD"

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
    bos, choch, break_dir = detect_bos_choch_v2(df_1h, structure)

    # Range compression / low-volatility filter (price-action only)
    if len(df_1h) >= 40:
        recent = df_1h.tail(20)
        prev = df_1h.iloc[-40:-20]
        recent_range = float(recent["high"].max() - recent["low"].min())
        prev_range = float(prev["high"].max() - prev["low"].min())
        if prev_range > 0 and (recent_range / prev_range) < 0.6:
            return result

    last_close = float(df_1h.iloc[-1]["close"])
    recent_ranges = (df_1h["high"] - df_1h["low"]).tail(20)
    if recent_ranges.median() / max(last_close, 1e-9) < 0.001:
        return result

    # BOS quality: require a meaningful break beyond the last structure level
    last_high, last_low = get_last_structure_levels(df_1h)
    recent_range = float(recent_ranges.max()) if not recent_ranges.empty else 0.0
    if break_dir == "BULL" and last_high is not None and recent_range > 0:
        if last_close < (last_high + BOS_BREAK_PCT * recent_range):
            return result
    if break_dir == "BEAR" and last_low is not None and recent_range > 0:
        if last_close > (last_low - BOS_BREAK_PCT * recent_range):
            return result

    FRESH_BOS_WINDOW = 8  # 8 cây 1H gần nhất (~8 giờ)
    if bos:
        last_bos_idx = detect_bos_index(df_1h, structure)
        if last_bos_idx is None or last_bos_idx < len(df_1h) - FRESH_BOS_WINDOW:
            bos = False
            choch = False
            break_dir = None

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
    # Liquidity zone
    # -------------------------
    liq = detect_liquidity_zones(
        df_1h,
        lookback=LIQ_LOOKBACK_1H,
        min_touches=LIQ_MIN_TOUCHES,
        tolerance=LIQ_TOLERANCE,
    )

    def _liq_cluster_ok(cluster: dict) -> bool:
        span = float(cluster.get("max", 0)) - float(cluster.get("min", 0))
        level = float(cluster.get("level", 0))
        return level > 0 and span / level <= (LIQ_TOLERANCE * 2)

    # HTF liquidity levels (for LTF sweep validation)
    htf_liquidity = []
    for z in liq.get("equal_lows", []):
        if not _liq_cluster_ok(z):
            continue
        htf_liquidity.append(
            {
                "type": "BUY",
                "price": float(z["level"]),
            }
        )

    for z in liq.get("equal_highs", []):
        if not _liq_cluster_ok(z):
            continue
        htf_liquidity.append(
            {
                "type": "SELL",
                "price": float(z["level"]),
            }
        )

    last_close = float(df_1h.iloc[-1]["close"])
    base_range_pct = (recent_range / max(last_close, 1e-9)) if recent_range > 0 else 0
    prox_pct = min(MAX_ZONE_PROX_PCT, max(MIN_ZONE_PROX_PCT, base_range_pct))
    liq_prox = min(MAX_ZONE_PROX_PCT, max(LIQ_PROXIMITY, base_range_pct))

    def _near(level: float) -> bool:
        return abs(last_close - float(level)) / max(last_close, 1e-9) <= liq_prox

    def _zone_near_price(zone: dict) -> bool:
        mid = (float(zone["low"]) + float(zone["high"])) / 2.0
        return abs(last_close - mid) / max(last_close, 1e-9) <= prox_pct

    if not any(_zone_near_price(z) for z in zones):
        return result

    if bias == "BUY":
        result["liquidity_zone_ok"] = any(
            _liq_cluster_ok(z) and _near(z["level"]) for z in liq.get("equal_lows", [])
        )
    else:
        result["liquidity_zone_ok"] = any(
            _liq_cluster_ok(z) and _near(z["level"]) for z in liq.get("equal_highs", [])
        )

    # -------------------------
    # Final validation
    # -------------------------
    if result["liquidity_zone_ok"]:
        result["setup_valid"] = True

    result["htf_liquidity"] = htf_liquidity

    return result


# ---------------------------
# 15M ENTRY
# ---------------------------
# ---------------------------
# 15M ENTRY (cleaned)
# ---------------------------
def get_15m_entry(
    df_15m: pd.DataFrame,
    bias: str,
    setup_valid: bool,
    ob_fvg_list: list,
    htf_liquidity: list,
) -> dict:
    """
    Compute 15M entry signals based on SMC v2.1:
    - HTF setup validation
    - Liquidity grab (sweep + reclaim)
    - Entry must occur AFTER sweep (freshness enforced)
    - Price must react at HTF OB/FVG
    - Inside-bar compression handling
    - 15M confirmation / displacement
    """
    result = {
        "entry_signal": False,
        "entry_price": None,
        "stop": None,
        "target": None,
        "reason": "",
    }
    # HTF validation
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
    # Liquidity grab check (Liquidity sweep)
    # -------------------------
    df_recent = _recent_window(df_15m, max_window=120)
    grabbed, sweep_idx, ref = detect_liquidity_grab_v2(
        df_recent,
        bias=bias,
        lookback=20,
        sweep_window=5,
        max_window=120,
    )

    if not grabbed:
        result["reason"] = "NO_LIQUIDITY_SWEEP"
        return result

    entry_idx = len(df_recent) - 1
    if entry_idx <= sweep_idx:
        result["reason"] = "ENTRY_BEFORE_SWEEP"
        return result

    # Validate sweep is HTF liquidity (BUG #1 FIX)
    if not is_sweep_htf_liquidity(
        sweep_price=ref,
        bias=bias,
        htf_liquidity=htf_liquidity,  # lấy từ 1H setup
    ):
        result["reason"] = "SWEEP_NOT_HTF_LIQUIDITY"
        return result

    # Sweep freshness (anti-late entry)
    MAX_SWEEP_DISTANCE = 6  # 6 x 15m = 90 minutes
    if entry_idx - sweep_idx > MAX_SWEEP_DISTANCE:
        result["reason"] = "SWEEP_TOO_OLD"
        return result

    # Reclaim validation
    if bias == "BUY" and recent["close"] <= ref:
        result["reason"] = "NO_RECLAIM_CONFIRM"
        return result

    if bias == "SELL" and recent["close"] >= ref:
        result["reason"] = "NO_RECLAIM_CONFIRM"
        return result

    # -------------------------
    # Price returns to HTF OB / FVG (FIRST TAP ONLY)
    # -------------------------
    in_zone = False
    zone_first_tap = False
    current_idx = len(df_15m) - 1

    for zone in ob_fvg_list:
        low, high, typ = parse_zone(zone)

        if bias == "BUY" and typ != "BULL":
            continue
        if bias == "SELL" and typ != "BEAR":
            continue

        if _candle_overlaps_zone(recent, low, high):
            in_zone = True

            if is_first_tap_zone(
                df_15m,
                zone_low=low,
                zone_high=high,
                current_idx=current_idx,
                lookback=50,
            ):
                zone_first_tap = True
                break  # ✅ first valid zone found
            else:
                # touched but NOT first tap → reject this zone
                continue

    if not in_zone:
        result["reason"] = "CANDLE_NOT_IN_OB_FVG"
        return result

    if not zone_first_tap:
        result["reason"] = "NOT_FIRST_TAP_ZONE"
        return result

    # OB reaction (zone must HOLD price)
    if bias == "BUY" and recent["close"] < recent["open"]:
        result["reason"] = "NO_OB_REJECTION"
        return result

    if bias == "SELL" and recent["close"] > recent["open"]:
        result["reason"] = "NO_OB_REJECTION"
        return result

    # -------------------------
    # Inside-bar logic (compression)
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
    # 5M confirmation / displacement
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
    entry_15m = get_15m_entry(
        df_15m,
        bias,
        setup_1h["setup_valid"],
        zones_1h,
        setup_1h.get("htf_liquidity", []),
    )

    return {
        "bias_4h": bias,
        "setup_1h": setup_1h,
        "entry_15m": entry_15m,
        "bias": bias,
        "setup_valid": bool(setup_1h.get("setup_valid")),
        "entry_signal": bool(entry_15m.get("entry_signal")),
    }
