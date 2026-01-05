"""
Paper trading / backtest simulator (single-position, long-only).

What this script does:
- Pulls recent OHLCV for 1h and 4h from MEXC via CCXT.
- Runs a low-frequency strategy stack with regime switching.
- Simulates order execution with optional fees and slippage.
- Logs every entry/exit with timestamps, exact prices, PnL (USDT), win/loss flag,
  and the entry reason.
- Auto-exports a CSV of all completed trades and prints a win-rate summary.

Important limitations (by design):
- Candle-close execution only (no intrabar path modeling).
- No order book, partial fills, funding, or borrowing.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import ccxt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from indicators.bb_atr import atr
from regimes.regime_detector import detect_regime
from risk.risk_manager import calculate_position_size
from strategies.breakout import breakout_strategy
from strategies.mean_reversion import mean_reversion_strategy
from strategies.trend_participation import (
    detect_trend_permission,
    trend_participation_strategy,
)


# --- Simulator Settings ---
START_BALANCE = 100.0  # USDT
RISK_PER_TRADE = 0.01  # 1% per trade
SYMBOL = "BTC/USDT"

# Friction model (set to 0.0 to disable)
FEE_RATE = 0.001  # 0.1% per side
SLIPPAGE_RATE = 0.0002  # 0.02% price slippage per fill

# Output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "paper_trading.log")

# Execution behavior
ALLOW_REENTRY_SAME_CANDLE = False


def setup_logger() -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger = logging.getLogger("paper_trading")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
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
    take_profit: float
    entry_fee: float
    entry_notional: float
    entry_reason: str
    run_ts: str = ""


def choose_signal(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    """Select a strategy signal based on 4h regime.

    Returns (regime, signal, source_strategy).
    """
    regime = (
        detect_regime(df_4h) if df_4h is not None and len(df_4h) >= 50 else "TRANSITION"
    )

    if regime in ["TREND_UP", "TREND_DOWN"]:
        trend_dir = detect_trend_permission(df_4h)
        signal = trend_participation_strategy(df_1h, trend_dir)
        source_strategy = "trend_participation"
        if signal.action == "hold":
            signal = breakout_strategy(df_1h, period=3)
            source_strategy = "breakout"
    elif regime == "RANGE":
        signal = mean_reversion_strategy(df_1h)
        source_strategy = "mean_reversion"
    else:
        signal = breakout_strategy(df_1h, period=3)
        source_strategy = "breakout"

    return regime, signal, source_strategy


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
    signal,
    df_1h: pd.DataFrame,
    source_strategy: str,
) -> None:
    """Mutate the signal in-place by applying entry filters.

    Mirrors the gating used in `bot.py`:
    - Volume confirmation for any enter_long.
    - Stochastic filter only for mean reversion.
    - ORB filter only for breakout.
    """
    if getattr(signal, "action", "hold") != "enter_long":
        return

    if not volume_confirmation(df_1h):
        signal.action = "hold"
        signal.reason = f"{getattr(signal, 'reason', '')} | SKIP: low volume".strip()
        return

    if source_strategy == "mean_reversion":
        stoch = stochastic_filter(df_1h)
        if stoch > 80.0:
            signal.action = "hold"
            signal.reason = (
                f"{getattr(signal, 'reason', '')} | SKIP: stochastic overbought ({stoch:.1f})".strip()
            )
            return

    if source_strategy == "breakout":
        if not opening_range_breakout(df_1h):
            signal.action = "hold"
            signal.reason = f"{getattr(signal, 'reason', '')} | SKIP: no ORB".strip()
            return


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

    exit_reason = None
    trigger_price = None
    if last_close <= pos.stop_loss:
        exit_reason = "stop_loss"
        trigger_price = pos.stop_loss
    elif last_close >= pos.take_profit:
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
    logger.info(
        "EXIT %s ts=%s entry=%.2f exit=%.2f size=%.6f pnl=%.4f win=%s reason=%s",
        SYMBOL,
        candle_ts.isoformat(),
        pos.entry_price,
        exit_price,
        pos.size,
        pnl_usdt,
        win,
        exit_reason,
    )

    return None, balance_after, True


def try_open_position(
    *,
    logger: logging.Logger,
    pos: Position | None,
    candle_ts: pd.Timestamp,
    signal,
    run_ts: str,
    balance: float,
) -> tuple[Position | None, float, bool]:
    """Open a long position if the signal is actionable and affordable."""
    if pos is not None:
        return pos, balance, False

    if getattr(signal, "action", "hold") != "enter_long":
        return pos, balance, False

    if not (
        _is_valid_number(signal.entry)
        and _is_valid_number(signal.stop_loss)
        and _is_valid_number(signal.take_profit)
    ):
        logger.info("SKIP entry: missing/NaN entry/SL/TP")
        return pos, balance, False

    entry = float(signal.entry)
    stop = float(signal.stop_loss)
    take_profit = float(signal.take_profit)
    if stop >= entry:
        logger.info("SKIP entry: invalid long (stop >= entry)")
        return pos, balance, False

    # Slippage applies on entry; for buys we pay slightly worse price.
    entry_price = entry * (1 + SLIPPAGE_RATE)

    # Risk-based sizing, then cap by affordability.
    size = calculate_position_size(
        balance_usdt=balance,
        risk_per_trade=RISK_PER_TRADE,
        entry_price=entry_price,
        stop_loss=stop,
    )

    if entry_price <= 0:
        return pos, balance, False

    max_affordable = balance / (entry_price * (1 + FEE_RATE))
    size = min(size, max_affordable)
    if size <= 0:
        return pos, balance, False

    entry_notional = size * entry_price
    entry_fee = entry_notional * FEE_RATE
    if entry_notional + entry_fee > balance:
        logger.info("SKIP entry: insufficient USDT after fees")
        return pos, balance, False

    balance_after = balance - (entry_notional + entry_fee)
    reason = str(getattr(signal, "reason", ""))

    new_pos = Position(
        entry_ts=candle_ts,
        entry_price=entry_price,
        size=size,
        stop_loss=stop,
        take_profit=take_profit,
        entry_fee=entry_fee,
        entry_notional=entry_notional,
        entry_reason=reason,
        run_ts=run_ts,
    )

    logger.info(
        "ENTRY %s ts=%s entry=%.2f size=%.6f SL=%.2f TP=%.2f fee=%.4f reason=%s",
        SYMBOL,
        candle_ts.isoformat(),
        entry_price,
        size,
        stop,
        take_profit,
        entry_fee,
        reason,
    )

    return new_pos, balance_after, True


def export_trades_csv(trades: list[dict]) -> str | None:
    """Export completed trades to CSV and return the output path."""
    if not trades:
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"paper_trades_{ts}.csv")

    # Export only the requested schema (in this exact order).
    df = pd.DataFrame(trades)
    if "pnl" not in df.columns and "pnl_usdt" in df.columns:
        df["pnl"] = df["pnl_usdt"]

    export_cols = [
        "run_ts",
        "symbol",
        "entry_ts",
        "exit_ts",
        "entry_price",
        "exit_price",
        "pnl",
        "win",
    ]
    for col in export_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[export_cols]
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


def plot_performance(
    logger: logging.Logger, df: pd.DataFrame, run_id: str
) -> list[str]:
    """Generate cumulative PnL and equity curve charts; returns created paths."""
    if df.empty:
        return []

    try:
        matplotlib.use("Agg")
    except Exception as e:
        logger.info("Plotting skipped (matplotlib not available): %s", e)
        return []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    created: list[str] = []

    # Use exit timestamp for performance curves (realized PnL).
    x = df["exit_ts"]

    # --- Cumulative PnL ---
    pnl_path = os.path.join(OUTPUT_DIR, f"cum_pnl_{run_id}.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, df["cum_pnl"], linewidth=2)
    ax.set_title("Cumulative PnL (USDT)")
    ax.set_xlabel("Time")
    ax.set_ylabel("PnL (USDT)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pnl_path, dpi=150)
    plt.close(fig)
    created.append(pnl_path)

    # --- Equity curve + annotate best/worst trades ---
    equity_path = os.path.join(OUTPUT_DIR, f"equity_curve_{run_id}.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, df["equity_usdt"], linewidth=2)
    ax.set_title("Equity Curve (USDT)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (USDT)")
    ax.grid(True, alpha=0.3)

    best_idx = int(df["pnl"].astype(float).idxmax())
    worst_idx = int(df["pnl"].astype(float).idxmin())

    bx = df.loc[best_idx, "exit_ts"]
    by = float(df.loc[best_idx, "equity_usdt"])
    wx = df.loc[worst_idx, "exit_ts"]
    wy = float(df.loc[worst_idx, "equity_usdt"])

    ax.scatter([bx], [by], color="green", s=40, zorder=3)
    ax.scatter([wx], [wy], color="red", s=40, zorder=3)
    ax.annotate("Best", (bx, by), textcoords="offset points", xytext=(6, 6))
    ax.annotate("Worst", (wx, wy), textcoords="offset points", xytext=(6, -10))

    fig.tight_layout()
    fig.savefig(equity_path, dpi=150)
    plt.close(fig)
    created.append(equity_path)

    return created


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

    logger.info(
        "SUMMARY closed_trades=%d wins=%d losses=%d win_rate=%.2f%%",
        closed,
        wins,
        losses,
        win_rate,
    )
    logger.info(
        "SUMMARY start_balance=%.2f final_balance=%.2f net_change=%.2f",
        starting_balance,
        final_balance,
        final_balance - starting_balance,
    )
    logger.info(
        "SUMMARY total_pnl=%.4f avg_pnl=%.4f avg_hold_hours=%.2f",
        total_pnl,
        avg_pnl,
        avg_hold_h,
    )

    if closed:
        best = trades_df.loc[trades_df["pnl"].astype(float).idxmax()]
        worst = trades_df.loc[trades_df["pnl"].astype(float).idxmin()]
        logger.info(
            "BEST trade pnl=%.4f entry=%s exit=%s reason=%s",
            float(best["pnl"]),
            str(best["entry_ts"]),
            str(best["exit_ts"]),
            str(best.get("entry_reason", "")),
        )
        logger.info(
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

    df_1h = pd.DataFrame(
        exchange.fetch_ohlcv(SYMBOL, "1h", limit=500),
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], unit="ms")

    df_4h = pd.DataFrame(
        exchange.fetch_ohlcv(SYMBOL, "4h", limit=500),
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], unit="ms")

    balance = float(START_BALANCE)
    pos: Position | None = None
    completed_trades: list[dict] = []

    # Simulate over last 100 1h candles
    for i in range(100, len(df_1h)):
        df_1h_ctx = df_1h.iloc[:i]
        candle_ts = df_1h_ctx["timestamp"].iloc[-1]
        last_close = float(df_1h_ctx["close"].iloc[-1])

        # Align 4h context to current 1h candle timestamp to avoid future leakage
        df_4h_ctx = df_4h[df_4h["timestamp"] <= candle_ts]

        # Choose strategy signal
        regime, signal, source_strategy = choose_signal(df_1h_ctx, df_4h_ctx)

        # Close first (if SL/TP hit on this candle close)
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
        apply_entry_filters(
            signal=signal,
            df_1h=df_1h_ctx,
            source_strategy=source_strategy,
        )

        # Open
        pos, balance, _opened = try_open_position(
            logger=logger,
            pos=pos,
            candle_ts=candle_ts,
            signal=signal,
            run_ts=run_ts,
            balance=balance,
        )

        # If you want extra visibility per bar, uncomment:
        # logger.info("BAR ts=%s regime=%s balance=%.2f", candle_ts.isoformat(), regime, balance)

    # Mark-to-market: close any open position at the final available price.
    if pos is not None:
        candle_ts = df_1h["timestamp"].iloc[-1]
        last_close = float(df_1h["close"].iloc[-1])
        exit_price = last_close * (1 - SLIPPAGE_RATE)
        exit_notional = pos.size * exit_price
        exit_fee = exit_notional * FEE_RATE
        balance += exit_notional - exit_fee
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
            "EXIT %s ts=%s entry=%.2f exit=%.2f size=%.6f pnl=%.4f win=%s reason=%s",
            SYMBOL,
            candle_ts.isoformat(),
            pos.entry_price,
            exit_price,
            pos.size,
            pnl_usdt,
            win,
            "liquidate_end",
        )

    trades_df = build_trades_df(completed_trades, START_BALANCE)

    csv_path = export_trades_csv(completed_trades)
    if csv_path:
        logger.info("CSV exported: %s", csv_path)
    else:
        logger.info("CSV not exported (no completed trades).")

    chart_paths = plot_performance(logger, trades_df, run_id)
    for p in chart_paths:
        logger.info("Chart saved: %s", p)

    print_summary(logger, trades_df, START_BALANCE, balance)
    logger.info("Final simulated balance: %.2f USDT", balance)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Allows piping to tools like `head` without a noisy traceback.
        raise SystemExit(0)
