"""15m spot simulation template (BTC/ETH) using a 1H regime filter.

What this demonstrates (based on the earlier suggestion):
- Trade 15m entries, but filter direction using 1H regime (avoid longs in bearish regimes).
- Prefer breakout + trend_participation in trending regimes.
- Allow mean_reversion only in RANGE regime.
- Use the existing SpotTradeLifecycle for ATR stop, trailing, and partial profit.

Input data:
- CSVs with columns: datetime, open, high, low, close, volume
- Example paths:
  - data/BTCUSDT_15M.csv
  - data/ETHUSDT_15M.csv

This script does NOT modify your existing codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from crypto_trading.io.loader import load_ohlcv
from crypto_trading.regimes.regime_detector import detect_regime
from crypto_trading.lifecycle.spot_lifecycle import SpotTradeLifecycle, SpotTradeState

from crypto_trading.strategies.breakout import breakout_signal
from crypto_trading.strategies.trend_participation import trend_participation_signal
from crypto_trading.strategies.mean_reversion import mean_reversion_signal
from crypto_trading.strategies.ema_sma_crossover import ema_sma_cross_signal


Signal = Literal["BUY", "SELL", "HOLD"]


def resample_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Convert 15m OHLCV into 1H OHLCV."""
    if df_15m is None or len(df_15m) == 0:
        return df_15m
    if not isinstance(df_15m.index, pd.DatetimeIndex):
        raise ValueError("df_15m must have a DatetimeIndex")

    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_1h = df_15m.resample("1h").agg(ohlc).dropna()
    return df_1h


def volume_confirmation(df_in: pd.DataFrame, *, lookback: int = 20, multiplier: float = 1.3) -> bool:
    if df_in is None or len(df_in) < int(lookback) or "volume" not in df_in.columns:
        return False
    avg_vol = df_in["volume"].rolling(int(lookback)).mean().iloc[-1]
    latest_vol = df_in["volume"].iloc[-1]
    if pd.isna(avg_vol) or pd.isna(latest_vol):
        return False
    return float(latest_vol) > float(avg_vol) * float(multiplier)


def pick_15m_signal(df_15m_slice: pd.DataFrame, df_1h_slice: pd.DataFrame) -> tuple[Signal, str, str]:
    """Return (signal, regime_1h, source).

    - Uses 1H regime for context.
    - Generates entries on 15m.
    """
    if df_15m_slice is None or len(df_15m_slice) < 60:
        return "HOLD", "TRANSITION", "none"

    regime_1h = detect_regime(df_1h_slice) if (df_1h_slice is not None and len(df_1h_slice) >= 60) else "TRANSITION"

    # Spot long-only safety: never open new longs in bearish regimes.
    if regime_1h in {"TREND_DOWN", "CROSS_DOWN"}:
        return "HOLD", regime_1h, "filter_bearish"

    # RANGE: allow mean reversion only here.
    if regime_1h == "RANGE":
        sig = mean_reversion_signal(df_15m_slice, regime=regime_1h)
        return sig, regime_1h, "mean_reversion"

    # TREND/CROSS/TRANSITION: prefer trend participation, then breakout.
    if regime_1h in {"TREND_UP", "CROSS_UP", "TRANSITION"}:
        sig = trend_participation_signal(df_15m_slice, regime="TREND_UP")
        if sig != "HOLD":
            return sig, regime_1h, "trend_participation"

        sig = breakout_signal(df_15m_slice)
        if sig == "BUY":
            return "BUY", regime_1h, "breakout"

        # Optional: crossover as confirmation only (never forced entry)
        cross = ema_sma_cross_signal(df_15m_slice)
        if cross == "SELL":
            return "SELL", regime_1h, "ema_sma_crossover"

        return "HOLD", regime_1h, "none"

    return "HOLD", regime_1h, "none"


@dataclass
class BacktestConfig:
    starting_balance: float = 100.0
    position_usdt: float = 20.0
    max_risk_per_trade: float = 0.02

    # Entry selectivity (important on 15m)
    require_volume_expansion: bool = True
    volume_multiplier: float = 1.3

    # Lifecycle (risk management)
    lifecycle_kwargs: dict | None = None


def run_15m_backtest(symbol: str, ohlcv_15m_csv: str, cfg: BacktestConfig) -> dict:
    df_15m = load_ohlcv(ohlcv_15m_csv)
    df_1h = resample_to_1h(df_15m)

    lifecycle_defaults = {
        "atr_multiplier": 3.0,
        "trail_pct": None,
        "trail_atr_mult": 2.5,
        "partial_profit_pct": 0.03,
        "partial_sell_fraction": 0.5,
        "max_bars_in_trade": 48 * 4,  # 48h worth of 15m bars
        "cooldown_bars": 3,
    }
    if cfg.lifecycle_kwargs:
        lifecycle_defaults.update(cfg.lifecycle_kwargs)

    lifecycle = SpotTradeLifecycle(**lifecycle_defaults)
    state = SpotTradeState()

    balance = float(cfg.starting_balance)
    closed_trades: list[dict] = []

    # Need ~200 1H bars for SMA200 regime detection -> ~800 15m bars.
    min_15m_bars = 900

    for idx in range(min_15m_bars, len(df_15m)):
        t = df_15m.index[idx]

        df_15m_slice = df_15m.iloc[: idx + 1]
        df_1h_slice = df_1h.loc[: t.floor("1h")]

        price = float(df_15m_slice["close"].iloc[-1])
        prev_state = state

        signal, regime_1h, source = pick_15m_signal(df_15m_slice, df_1h_slice)

        # Optional entry filter: volume expansion (good on 15m to avoid chop)
        if signal == "BUY" and cfg.require_volume_expansion:
            if not volume_confirmation(df_15m_slice, multiplier=float(cfg.volume_multiplier)):
                signal = "HOLD"

        state, trade_event = lifecycle.update(
            df_1h=df_15m_slice,  # lifecycle needs high/low/close; timeframe-agnostic
            state=state,
            signal=signal,
            regime=regime_1h,
            price=price,
            bar_index=idx,
        )

        if trade_event is None:
            continue

        # Console output (no HOLD)
        pnl_usdt = 0.0
        side = "HOLD"

        if trade_event["type"] == "ENTRY":
            side = "BUY"
            entry_price = float(trade_event.get("entry_price") or price)

            # Determine qty using: min(notional cap, risk cap)
            if entry_price > 0 and balance > 0:
                target_notional = min(float(cfg.position_usdt), float(balance))
                target_qty = target_notional / entry_price

                stop_price = getattr(state, "atr_stop", None)
                stop_dist = (entry_price - float(stop_price)) if stop_price is not None else 0.0

                risk_usdt = float(balance) * float(cfg.max_risk_per_trade)
                risk_qty = (risk_usdt / stop_dist) if stop_dist > 0 else target_qty

                qty = max(0.0, min(float(target_qty), float(risk_qty)))
                state.qty = qty
                state.entry_notional_usdt = qty * entry_price

        elif trade_event["type"] == "PARTIAL":
            side = "SELL"
            pnl_usdt = float(trade_event.get("pnl_usdt") or 0.0)
            balance += pnl_usdt

        elif trade_event["type"] == "EXIT":
            side = "SELL"
            pnl_usdt = float(trade_event.get("pnl_usdt") or 0.0)
            pnl_total_usdt = float(trade_event.get("pnl_total_usdt") or pnl_usdt)
            balance += pnl_usdt

            closed_trades.append(
                {
                    "entry_time": None,
                    "exit_time": t,
                    "entry_price": prev_state.entry_price,
                    "exit_price": float(trade_event.get("price") or price),
                    "pnl": pnl_total_usdt,
                    "regime": regime_1h,
                    "source": source,
                }
            )

        print(f"{t},{symbol},{side},{pnl_usdt:.2f},{balance:.2f}")

        if balance <= 0:
            break

    total_trades = len(closed_trades)
    wins = sum(1 for tr in closed_trades if tr["pnl"] > 0)
    losses = sum(1 for tr in closed_trades if tr["pnl"] < 0)
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = sum(tr["pnl"] for tr in closed_trades)

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "total_PnL": round(float(total_pnl), 2),
        "ending_balance": round(float(balance), 2),
    }


if __name__ == "__main__":
    cfg = BacktestConfig(
        starting_balance=100.0,
        position_usdt=20.0,
        max_risk_per_trade=0.02,
        require_volume_expansion=True,
        volume_multiplier=1.3,
        lifecycle_kwargs={
            "atr_multiplier": 3.0,
            "trail_pct": None,
            "trail_atr_mult": 2.5,
            "partial_profit_pct": 0.03,
            "partial_sell_fraction": 0.5,
        },
    )

    # Update these to your actual 15m CSV paths
    # btc_csv = "data/BTCUSDT_15M.csv"
    eth_csv = "data/ETHUSDT_15M.csv"

    # print("time,symbol,side,pnl_usdt,balance_after")
    # btc_stats = run_15m_backtest("BTCUSDT", btc_csv, cfg)
    # print("\nBTC summary:", btc_stats)

    print("\n---")

    print("time,symbol,side,pnl_usdt,balance_after")
    eth_stats = run_15m_backtest("ETHUSDT", eth_csv, cfg)
    print("\nETH summary:", eth_stats)
