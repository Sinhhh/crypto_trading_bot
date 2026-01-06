"""
Paper trading / backtest simulator (single-position, long-only).

What this script does:
- Pulls recent OHLCV for 1h and 4h from MEXC via CCXT.
- Runs a low-frequency strategy stack with regime switching.
- Simulates order execution with optional fees and slippage.
- Logs every entry/exit with timestamps, exact prices, PnL (USDT), win/loss flag,
  and the entry reason.
- Auto-exports a CSV of all completed trades.

Important limitations (by design):
- Candle-close execution only (no intrabar path modeling).
- No order book, partial fills, funding, or borrowing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import ccxt
import pandas as pd

from crypto_trading.indicators.volatility import atr
from crypto_trading.indicators.momentum import rsi, adx
from crypto_trading.regimes.regime_detector import detect_regime
from crypto_trading.risk.risk_manager import calculate_position_size
from crypto_trading.indicators.volatility import bollinger_bands
from crypto_trading.strategies.breakout import breakout_signal
from crypto_trading.strategies.ema_sma_crossover import ema_sma_cross_signal
from crypto_trading.strategies.mean_reversion import mean_reversion_signal
from crypto_trading.strategies.trend_participation import trend_participation_signal


# --- Simulator Settings ---
START_BALANCE = 100.0  # USDT
SYMBOL = "BTC/USDT"

# Trading personality: "SAFE", "GROWTH", or "AUTO"
PERSONALITY_MODE = "AUTO"

# Timeframe for simulation/regime/strategies. Choose "1h" or "4h".
TIMEFRAME = "4h"

# Risk management (non-negotiable): max 1–2% per trade.
MAX_RISK_PER_TRADE = 0.02

# Friction model (set to 0.0 to disable)
FEE_RATE = 0.001  # 0.1% per side
SLIPPAGE_RATE = 0.0002  # 0.02% price slippage per fill

# Output
OUTPUT_DIR = os.path.join("reports")
LOG_PATH = os.path.join("logs", "paper_trading.log")

# Execution behavior
ALLOW_REENTRY_SAME_CANDLE = False


def setup_logger() -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger = logging.getLogger("paper_trading")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        # Keep output strictly as CSV lines (no timestamps/levels).
        fmt = logging.Formatter("%(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(LOG_PATH)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _is_valid_number(x: Any) -> bool:
    try:
        return x is not None and pd.notna(x)
    except Exception:
        return False


@dataclass
class Position:
    entry_ts: pd.Timestamp
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float | None
    entry_fee: float
    entry_notional: float
    entry_reason: str
    run_ts: str = ""
    personality: str = ""
    trailing_atr_mult: float | None = None


@dataclass(frozen=True)
class PersonalitySettings:
    name: str
    target_notional_usdt: float
    sl_pct: float
    tp_pct: float | None
    trailing_atr_mult: float | None
    ml_threshold: float
    risk_per_trade: float


def get_personality_settings(name: str) -> PersonalitySettings:
    n = str(name).strip().upper()
    if n == "SAFE":
        return PersonalitySettings(
            name="SAFE",
            target_notional_usdt=20.0,
            sl_pct=0.03,
            tp_pct=0.01,
            trailing_atr_mult=None,
            ml_threshold=0.80,
            risk_per_trade=0.01,
        )
    if n == "GROWTH":
        return PersonalitySettings(
            name="GROWTH",
            target_notional_usdt=30.0,
            sl_pct=0.02,
            tp_pct=None,
            trailing_atr_mult=1.5,
            ml_threshold=0.60,
            risk_per_trade=0.02,
        )

    # AUTO is handled elsewhere; default to SAFE settings.
    return get_personality_settings("SAFE")


def estimate_confidence(
    *,
    df: pd.DataFrame,
    source_strategy: str,
    regime: str,
) -> float:
    """Heuristic confidence score in [0, 1].

    This acts as the "ML threshold" gate without requiring a real ML model.
    """
    if df is None or len(df) < 30 or "close" not in df.columns:
        return 0.0

    s = str(source_strategy)
    try:
        if s == "mean_reversion":
            close = df["close"].astype(float)
            r = float(rsi(close, 14).iloc[-1])
            if pd.isna(r):
                return 0.0
            # Extremeness away from neutral 50.
            return max(0.0, min(1.0, abs(r - 50.0) / 25.0))

        if s in {"trend_participation", "breakout", "ema_sma_crossover"}:
            a = float(adx(df, 14).iloc[-1])
            if pd.isna(a):
                return 0.0
            return max(0.0, min(1.0, a / 50.0))
    except Exception:
        return 0.0

    # Fallback: confidence from regime clarity.
    if regime in {"TREND_UP", "TREND_DOWN", "RANGE"}:
        return 0.6
    return 0.4


def auto_select_personality(
    *,
    balance: float,
    equity: float,
    equity_peak: float,
    daily_pnl: float,
    regime: str,
) -> str:
    """Choose SAFE/GROWTH based on balance, pnl, drawdown, and regime."""
    if equity_peak <= 0:
        return "SAFE"

    dd_pct = (float(equity) - float(equity_peak)) / float(equity_peak)

    # Protection first.
    if float(balance) < 150.0:
        return "SAFE"
    if daily_pnl < 0.0:
        return "SAFE"
    if dd_pct <= -0.05:
        return "SAFE"

    # Prefer SAFE in non-directional regimes.
    if regime in {"RANGE", "TRANSITION"}:
        return "SAFE"

    # Otherwise, allow growth when the market is behaving directionally.
    if regime in {"TREND_UP", "TREND_DOWN", "CROSS_UP", "CROSS_DOWN"}:
        return "GROWTH"

    return "SAFE"


def choose_signal(df_tf: pd.DataFrame, df_regime: pd.DataFrame):
    """Select a strategy signal based on regime timeframe.

    Returns (regime, signal, source_strategy, reason).
    """
    regime = (
        detect_regime(df_regime)
        if df_regime is not None and len(df_regime) >= 50
        else "TRANSITION"
    )

    if regime in ["TREND_UP", "TREND_DOWN"]:
        signal = trend_participation_signal(df_tf, regime)
        source_strategy = "trend_participation"
        reason = "trend_participation"
        if signal == "HOLD":
            signal = breakout_signal(df_tf, period=3)
            source_strategy = "breakout"
            reason = "breakout"
    elif regime == "RANGE":
        signal = mean_reversion_signal(df_tf, regime)
        source_strategy = "mean_reversion"
        reason = "mean_reversion"
    elif regime in ["CROSS_UP", "CROSS_DOWN"]:
        signal = ema_sma_cross_signal(df_tf)
        source_strategy = "ema_sma_crossover"
        reason = "ema_sma_crossover"
    else:
        signal = breakout_signal(df_tf, period=3)
        source_strategy = "breakout"
        reason = "breakout"

    return regime, signal, source_strategy, reason


def volume_confirmation(df: pd.DataFrame, multiplier: float = 1.5) -> bool:
    """Return True when latest volume exceeds multiplier × 20-period average."""
    if df is None or len(df) < 20 or "volume" not in df.columns:
        return False
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    latest_vol = df["volume"].iloc[-1]
    if pd.isna(avg_vol) or pd.isna(latest_vol):
        return False
    return float(latest_vol) > float(avg_vol) * float(multiplier)


def stochastic_filter(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> float:
    """Return the latest Stochastic %D (0-100). Uses 50 as neutral fallback."""
    if df is None or len(df) < k_period or not {"high", "low", "close"}.issubset(df.columns):
        return 50.0

    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = high_max - low_min
    if pd.isna(denom.iloc[-1]) or float(denom.iloc[-1]) == 0.0:
        return 50.0

    percent_k = 100.0 * (df["close"] - low_min) / denom
    percent_d = percent_k.rolling(d_period).mean()
    latest = percent_d.iloc[-1]
    if pd.isna(latest):
        return 50.0
    return float(latest)


def opening_range_breakout(df: pd.DataFrame, k: float = 0.5) -> bool:
    """Return True if latest close exceeds today's opening range + k*ATR.

    Uses the first candle of the latest day in `df` as the opening range.
    """
    if df is None or len(df) < 2 or "timestamp" not in df.columns:
        return False

    latest_day = df["timestamp"].iloc[-1].date()
    df_day = df[df["timestamp"].dt.date == latest_day]
    if len(df_day) < 2:
        return False

    first_candle = df_day.iloc[0]
    latest_close = df["close"].iloc[-1]
    atr_val = atr(df).iloc[-1]
    if pd.isna(atr_val) or float(atr_val) <= 0.0:
        return False

    return float(latest_close) > float(first_candle["high"]) + float(k) * float(atr_val)


def apply_entry_filters(
    *,
    signal: str,
    df_1h: pd.DataFrame,
    source_strategy: str,
) -> str:
    """Mutate the signal in-place by applying entry filters.

    Mirrors the gating used in `bot.py`:
    - Volume confirmation for any enter_long.
    - Stochastic filter only for mean reversion.
    - ORB filter only for breakout.
    """
    if signal != "BUY":
        return signal

    if not volume_confirmation(df_1h):
        return "HOLD"

    if source_strategy == "mean_reversion":
        stoch = stochastic_filter(df_1h)
        if stoch > 80.0:
            return "HOLD"

    if source_strategy == "breakout":
        if not opening_range_breakout(df_1h):
            return "HOLD"

    return signal


def compute_levels_for_personality(
    *,
    df_tf: pd.DataFrame,
    source_strategy: str,
    settings: PersonalitySettings,
) -> tuple[float | None, float | None, float | None]:
    """Compute entry, stop, and TP using the personality rules.

    SAFE: fixed % SL/TP.
    GROWTH: fixed % SL, no TP.
    """
    if df_tf is None or len(df_tf) < 30 or not {"high", "low", "close"}.issubset(df_tf.columns):
        return None, None, None

    entry = float(df_tf["close"].iloc[-1])
    if entry <= 0:
        return None, None, None

    stop = entry * (1.0 - float(settings.sl_pct))
    tp = None if settings.tp_pct is None else entry * (1.0 + float(settings.tp_pct))

    return entry, stop, tp


def update_trailing_stop(
    *,
    pos: Position,
    df_tf: pd.DataFrame,
) -> Position:
    if pos is None or pos.trailing_atr_mult is None:
        return pos
    if df_tf is None or len(df_tf) < 30:
        return pos

    try:
        a = float(atr(df_tf, 14).iloc[-1])
        if pd.isna(a) or a <= 0:
            return pos
        last_close = float(df_tf["close"].iloc[-1])
        candidate = last_close - float(pos.trailing_atr_mult) * a
        if candidate > pos.stop_loss:
            pos.stop_loss = float(candidate)
    except Exception:
        return pos

    return pos


def compute_entry_levels(
    *,
    df_1h: pd.DataFrame,
    source_strategy: str,
) -> tuple[float | None, float | None, float | None]:
    """Compute entry, stop, and take-profit levels for long-only simulation."""
    if df_1h is None or len(df_1h) < 30 or not {"high", "low", "close"}.issubset(df_1h.columns):
        return None, None, None

    entry = float(df_1h["close"].iloc[-1])
    atr_val = atr(df_1h, 14).iloc[-1]
    if pd.isna(atr_val) or float(atr_val) <= 0.0:
        return None, None, None
    a = float(atr_val)

    if source_strategy == "trend_participation":
        stop = entry - 1.5 * a
        tp = entry + 3.0 * a
        return entry, stop, tp

    if source_strategy == "breakout":
        stop = entry - 1.0 * a
        tp = entry + 2.0 * a
        return entry, stop, tp

    if source_strategy == "ema_sma_crossover":
        stop = entry - 1.0 * a
        tp = entry + 2.0 * a
        return entry, stop, tp

    if source_strategy == "mean_reversion":
        close = df_1h["close"].astype(float)
        mid, _upper, _lower = bollinger_bands(close, 20, 2.0)
        bb_mid = mid.iloc[-1]
        if pd.isna(bb_mid):
            return None, None, None
        tp = float(bb_mid)
        stop = entry - 1.0 * a
        return entry, stop, tp

    return None, None, None


def try_close_position(
    *,
    logger: logging.Logger,
    pos: Position | None,
    candle_ts: pd.Timestamp,
    last_close: float,
    balance: float,
    trades: list[dict],
) -> tuple[Position | None, float, bool]:
    """Check stop-loss / take-profit at candle close and close if hit."""
    if pos is None:
        return None, balance, False

    # Trailing stop update (GROWTH) happens at candle close.
    # We update before checking stop-loss to trail as soon as possible.
    # (Candle-close execution only; no intrabar path modeling.)
    # NOTE: trailing uses the same df as the signal timeframe.
    # This function does not have the df context; callers update before invoking.

    exit_reason = None
    trigger_price = None
    if last_close <= pos.stop_loss:
        exit_reason = "stop_loss"
        trigger_price = pos.stop_loss
    elif pos.take_profit is not None and last_close >= pos.take_profit:
        exit_reason = "take_profit"
        trigger_price = pos.take_profit

    if exit_reason is None:
        return pos, balance, False

    # Slippage applies on exit; for sells we get slightly worse price.
    exit_price = float(trigger_price) * (1 - SLIPPAGE_RATE)
    exit_notional = pos.size * exit_price
    exit_fee = exit_notional * FEE_RATE
    balance_after = balance + (exit_notional - exit_fee)

    pnl_usdt = (exit_notional - exit_fee) - (pos.entry_notional + pos.entry_fee)
    win = pnl_usdt > 0

    trade = {
        "run_ts": pos.run_ts,
        "symbol": SYMBOL,
        "entry_ts": pos.entry_ts.isoformat(),
        "exit_ts": candle_ts.isoformat(),
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "pnl": pnl_usdt,
        "size": pos.size,
        "entry_fee": pos.entry_fee,
        "exit_fee": exit_fee,
        "pnl_usdt": pnl_usdt,
        "win": bool(win),
        "exit_reason": exit_reason,
        "entry_reason": pos.entry_reason,
    }
    trades.append(trade)
    # CSV-style exit log line (requested format)
    logger.info(
        "%s,%s,%s,%.2f,%.2f",
        candle_ts.strftime("%Y-%m-%d %H:%M"),
        SYMBOL,
        "SELL",
        float(pnl_usdt),
        float(balance_after),
    )

    return None, balance_after, True


def try_open_position(
    *,
    logger: logging.Logger,
    pos: Position | None,
    candle_ts: pd.Timestamp,
    signal: str,
    df_tf: pd.DataFrame,
    source_strategy: str,
    entry_reason: str,
    run_ts: str,
    balance: float,
    settings: PersonalitySettings,
    regime: str,
) -> tuple[Position | None, float, bool]:
    """Open a long position if the signal is actionable and affordable."""
    if pos is not None:
        return pos, balance, False

    if signal != "BUY":
        return pos, balance, False

    confidence = estimate_confidence(df=df_tf, source_strategy=source_strategy, regime=regime)
    if confidence < float(settings.ml_threshold):
        return pos, balance, False

    entry, stop, take_profit = compute_levels_for_personality(
        df_tf=df_tf,
        source_strategy=source_strategy,
        settings=settings,
    )
    if not (_is_valid_number(entry) and _is_valid_number(stop)):
        logger.debug("SKIP entry: missing/NaN entry/SL")
        return pos, balance, False

    entry = float(entry)
    stop = float(stop)
    tp: float | None
    if take_profit is None or (isinstance(take_profit, float) and pd.isna(take_profit)):
        tp = None
    else:
        tp = float(take_profit)
    if stop >= entry:
        logger.debug("SKIP entry: invalid long (stop >= entry)")
        return pos, balance, False

    if tp is not None and tp <= entry:
        logger.debug("SKIP entry: invalid long (tp <= entry)")
        return pos, balance, False

    # Slippage applies on entry; for buys we pay slightly worse price.
    entry_price = entry * (1 + SLIPPAGE_RATE)

    # Risk-based sizing (1–2%), then cap by target notional and affordability.
    risk_pct = min(float(settings.risk_per_trade), float(MAX_RISK_PER_TRADE))
    size_risk = calculate_position_size(
        balance_usdt=balance,
        risk_per_trade=risk_pct,
        entry_price=entry_price,
        stop_loss=stop,
    )
    size_target = float(settings.target_notional_usdt) / float(entry_price)
    size = min(float(size_risk), float(size_target))

    if entry_price <= 0:
        return pos, balance, False

    max_affordable = balance / (entry_price * (1 + FEE_RATE))
    size = min(size, max_affordable)
    if size <= 0:
        return pos, balance, False

    entry_notional = size * entry_price
    entry_fee = entry_notional * FEE_RATE
    if entry_notional + entry_fee > balance:
        logger.debug("SKIP entry: insufficient USDT after fees")
        return pos, balance, False

    balance_after = balance - (entry_notional + entry_fee)
    reason = str(entry_reason)

    new_pos = Position(
        entry_ts=candle_ts,
        entry_price=entry_price,
        size=size,
        stop_loss=stop,
        take_profit=tp,
        entry_fee=entry_fee,
        entry_notional=entry_notional,
        entry_reason=reason,
        run_ts=run_ts,
        personality=settings.name,
        trailing_atr_mult=settings.trailing_atr_mult,
    )

    logger.debug(
        "ENTRY %s ts=%s entry=%.2f size=%.6f SL=%.2f TP=%.2f fee=%.4f reason=%s",
        SYMBOL,
        candle_ts.isoformat(),
        entry_price,
        size,
        stop,
        tp if tp is not None else float("nan"),
        entry_fee,
        reason,
    )

    return new_pos, balance_after, True


def export_trades_csv(trades: list[dict], starting_balance: float) -> str | None:
    """Export completed trades to CSV and return the output path.

    Output schema (exact order):
    - time
    - symbol
    - side
    - pnl_usdt
    - balance_after
    """
    if not trades:
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"paper_trades_{ts}.csv")

    df = pd.DataFrame(trades).copy()
    if "pnl_usdt" not in df.columns and "pnl" in df.columns:
        df["pnl_usdt"] = df["pnl"]

    if "symbol" not in df.columns:
        df["symbol"] = SYMBOL

    # Exit time is the printed time for a close.
    if "exit_ts" in df.columns:
        exit_ts = pd.to_datetime(df["exit_ts"], errors="coerce")
    else:
        exit_ts = pd.to_datetime(pd.Series([pd.NA] * len(df)), errors="coerce")

    df["time"] = exit_ts.dt.strftime("%Y-%m-%d %H:%M")
    df["side"] = "SELL"  # long-only simulator: exits are sells

    pnl = pd.to_numeric(df.get("pnl_usdt", 0.0), errors="coerce").fillna(0.0).astype(float)
    df["pnl_usdt"] = pnl.round(2)

    # Balance after each close is starting balance + cumulative realized PnL.
    df["balance_after"] = (float(starting_balance) + pnl.cumsum()).round(2)

    df = df[["time", "symbol", "side", "pnl_usdt", "balance_after"]]
    df.to_csv(out_path, index=False)
    return out_path


def build_trades_df(trades: list[dict], starting_balance: float) -> pd.DataFrame:
    """Combine all trade metrics into a single DataFrame for analysis/plotting."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades).copy()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=False)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=False)
    df["holding_seconds"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds()
    df["holding_hours"] = df["holding_seconds"] / 3600.0
    if "pnl" not in df.columns and "pnl_usdt" in df.columns:
        df["pnl"] = df["pnl_usdt"]
    df["cum_pnl"] = df["pnl"].astype(float).cumsum()
    df["equity_usdt"] = starting_balance + df["cum_pnl"]
    df["equity_peak_usdt"] = df["equity_usdt"].cummax()
    df["drawdown_usdt"] = df["equity_usdt"] - df["equity_peak_usdt"]
    return df


def print_summary(
    logger: logging.Logger,
    trades_df: pd.DataFrame,
    starting_balance: float,
    final_balance: float,
) -> None:
    """Print win-rate, PnL, and holding-time summary + best/worst trade."""
    closed = int(len(trades_df))
    wins = int((trades_df["win"] == True).sum()) if closed else 0
    losses = int((trades_df["win"] == False).sum()) if closed else 0
    win_rate = (wins / closed) * 100.0 if closed else 0.0
    total_pnl = float(trades_df["pnl"].astype(float).sum()) if closed else 0.0
    avg_pnl = (total_pnl / closed) if closed else 0.0
    avg_hold_h = float(trades_df["holding_hours"].mean()) if closed else 0.0

    logger.debug(
        "SUMMARY closed_trades=%d wins=%d losses=%d win_rate=%.2f%%",
        closed,
        wins,
        losses,
        win_rate,
    )
    logger.debug(
        "SUMMARY start_balance=%.2f final_balance=%.2f net_change=%.2f",
        starting_balance,
        final_balance,
        final_balance - starting_balance,
    )
    logger.debug(
        "SUMMARY total_pnl=%.4f avg_pnl=%.4f avg_hold_hours=%.2f",
        total_pnl,
        avg_pnl,
        avg_hold_h,
    )

    if closed:
        best = trades_df.loc[trades_df["pnl"].astype(float).idxmax()]
        worst = trades_df.loc[trades_df["pnl"].astype(float).idxmin()]
        logger.debug(
            "BEST trade pnl=%.4f entry=%s exit=%s reason=%s",
            float(best["pnl"]),
            str(best["entry_ts"]),
            str(best["exit_ts"]),
            str(best.get("entry_reason", "")),
        )
        logger.debug(
            "WORST trade pnl=%.4f entry=%s exit=%s reason=%s",
            float(worst["pnl"]),
            str(worst["entry_ts"]),
            str(worst["exit_ts"]),
            str(worst.get("entry_reason", "")),
        )


def main() -> int:
    logger = setup_logger()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_ts = datetime.now(timezone.utc).isoformat()
    exchange = ccxt.mexc({"enableRateLimit": True})

    df_tf = pd.DataFrame(
        exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=500),
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df_tf["timestamp"] = pd.to_datetime(df_tf["timestamp"], unit="ms")

    balance = float(START_BALANCE)
    pos: Position | None = None
    completed_trades: list[dict] = []
    equity_peak = float(balance)

    # CSV-style header for trade exit logs
    logger.info("time,symbol,side,pnl_usdt,balance_after")

    # Simulate over last 100 timeframe candles
    for i in range(100, len(df_tf)):
        df_ctx = df_tf.iloc[:i]
        candle_ts = df_ctx["timestamp"].iloc[-1]
        last_close = float(df_ctx["close"].iloc[-1])

        # Use the same timeframe for regime to match the personalities.
        regime, signal, source_strategy, entry_reason = choose_signal(df_ctx, df_ctx)

        # Equity & drawdown tracking (rough MTM equity).
        equity = float(balance)
        if pos is not None:
            equity += float(pos.size) * float(last_close)
        equity_peak = max(float(equity_peak), float(equity))

        # Daily PnL from realized exits.
        day = candle_ts.date()
        daily_pnl = 0.0
        for t in reversed(completed_trades[-50:]):
            try:
                if pd.to_datetime(t.get("exit_ts")) .date() != day:
                    continue
                daily_pnl += float(t.get("pnl_usdt", t.get("pnl", 0.0)) or 0.0)
            except Exception:
                continue

        personality = PERSONALITY_MODE.strip().upper()
        if personality == "AUTO":
            personality = auto_select_personality(
                balance=float(balance),
                equity=float(equity),
                equity_peak=float(equity_peak),
                daily_pnl=float(daily_pnl),
                regime=str(regime),
            )

        settings = get_personality_settings(personality)

        # Close first (if SL/TP hit on this candle close)
        # Trailing stop update before close check (GROWTH).
        if pos is not None:
            pos = update_trailing_stop(pos=pos, df_tf=df_ctx)

        pos, balance, closed = try_close_position(
            logger=logger,
            pos=pos,
            candle_ts=candle_ts,
            last_close=last_close,
            balance=balance,
            trades=completed_trades,
        )

        # Optional prevention of re-entry in the same candle
        if closed and not ALLOW_REENTRY_SAME_CANDLE:
            continue

        # Apply the same entry gating as the live bot.
        signal = apply_entry_filters(
            signal=signal,
            df_1h=df_ctx,
            source_strategy=source_strategy,
        )

        # Open
        pos, balance, _opened = try_open_position(
            logger=logger,
            pos=pos,
            candle_ts=candle_ts,
            signal=signal,
            df_tf=df_ctx,
            source_strategy=source_strategy,
            entry_reason=entry_reason,
            run_ts=run_ts,
            balance=balance,
            settings=settings,
            regime=str(regime),
        )

        # If you want extra visibility per bar, uncomment:
        # logger.info("BAR ts=%s regime=%s balance=%.2f", candle_ts.isoformat(), regime, balance)

    # Mark-to-market: close any open position at the final available price.
    if pos is not None:
        candle_ts = df_tf["timestamp"].iloc[-1]
        last_close = float(df_tf["close"].iloc[-1])
        exit_price = last_close * (1 - SLIPPAGE_RATE)
        exit_notional = pos.size * exit_price
        exit_fee = exit_notional * FEE_RATE
        balance_after = balance + (exit_notional - exit_fee)
        balance = balance_after
        pnl_usdt = (exit_notional - exit_fee) - (pos.entry_notional + pos.entry_fee)
        win = pnl_usdt > 0

        completed_trades.append(
            {
                "run_ts": pos.run_ts,
                "symbol": SYMBOL,
                "entry_ts": pos.entry_ts.isoformat(),
                "exit_ts": candle_ts.isoformat(),
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "pnl": pnl_usdt,
                "size": pos.size,
                "entry_fee": pos.entry_fee,
                "exit_fee": exit_fee,
                "pnl_usdt": pnl_usdt,
                "win": bool(win),
                "exit_reason": "liquidate_end",
                "entry_reason": pos.entry_reason,
            }
        )

        logger.info(
            "%s,%s,%s,%.2f,%.2f",
            candle_ts.strftime("%Y-%m-%d %H:%M"),
            SYMBOL,
            "SELL",
            float(pnl_usdt),
            float(balance_after),
        )

    trades_df = build_trades_df(completed_trades, START_BALANCE)

    csv_path = export_trades_csv(completed_trades, START_BALANCE)
    if csv_path:
        logger.debug("CSV exported: %s", csv_path)
    else:
        logger.debug("CSV not exported (no completed trades).")

    print_summary(logger, trades_df, START_BALANCE, balance)
    logger.debug("Final simulated balance: %.2f USDT", balance)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Allows piping to tools like `head` without a noisy traceback.
        raise SystemExit(0)
