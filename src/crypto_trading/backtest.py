"""Spot backtest runner.

This module is importable (for reuse) and runnable via a thin wrapper at repo root
([bot.py](bot.py)).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_trading.io.loader import load_ohlcv
from crypto_trading.unified.unified_signal import unified_signal_with_meta
from crypto_trading.lifecycle.spot_lifecycle import SpotTradeState, SpotTradeLifecycle
from crypto_trading.indicators.momentum import rsi, adx


def run_spot_backtest(
    symbol: str,
    timeframe: str,
    ohlcv_file: str,
    starting_balance: float = 100.0,
    *,
    lifecycle_kwargs: dict | None = None,
    gate_kwargs: dict | None = None,
    print_rows: bool = True,
    max_candles: int | None = None,
    lookback_bars: int | None = None,
    bar_step: int = 1,
):
    """
    Spot backtest with lifecycle, PnL logging, and console output.
    Optimized for higher profit using dynamic position sizing, tiered profits, 
    and trend filtering.
    """
    df = load_ohlcv(ohlcv_file)
    if max_candles is not None and max_candles > 0 and len(df) > int(max_candles):
        df = df.tail(int(max_candles))
    if print_rows:
        print(f"Loaded {len(df)} candles for {symbol} {timeframe}.")

    state = SpotTradeState()
    lifecycle_defaults = {
        "atr_multiplier": 2.0,         # tighter stop
        "trail_pct": None,
        "trail_atr_mult": 2.5,         # allow some trailing
        "partial_profit_pct": 0.05,    # take profits later
        "partial_sell_fraction": 0.3,  # tiered profit taking
        "max_bars_in_trade": 48,
        "cooldown_bars": 3,
    }
    if lifecycle_kwargs:
        lifecycle_defaults.update(lifecycle_kwargs)
    lifecycle = SpotTradeLifecycle(**lifecycle_defaults)

    trades = []
    closed_trades = []  # Only ENTRY → EXIT trades
    balance = float(starting_balance)

    # Simple spot sizing controls
    POSITION_USDT = 25.0
    MAX_RISK_PER_TRADE = 0.03  # 3% per strong trade

    # Quality gates
    SAFE_ML_THRESHOLD = 0.80
    GROWTH_ML_THRESHOLD = 0.55  # slightly lower for more trades
    PERSONALITY_MODE = "GROWTH"  # "SAFE" | "GROWTH"
    VOLUME_MULTIPLIER = 1.2

    if gate_kwargs:
        SAFE_ML_THRESHOLD = float(gate_kwargs.get("SAFE_ML_THRESHOLD", SAFE_ML_THRESHOLD))
        GROWTH_ML_THRESHOLD = float(gate_kwargs.get("GROWTH_ML_THRESHOLD", GROWTH_ML_THRESHOLD))
        PERSONALITY_MODE = str(gate_kwargs.get("PERSONALITY_MODE", PERSONALITY_MODE))
        VOLUME_MULTIPLIER = float(gate_kwargs.get("VOLUME_MULTIPLIER", VOLUME_MULTIPLIER))

    # detect_regime() needs >=200 bars (SMA200), so skip until then.
    min_bars_required = 220

    def volume_confirmation(df_in: pd.DataFrame, multiplier: float = 1.2) -> bool:
        if df_in is None or len(df_in) < 20 or "volume" not in df_in.columns:
            return False
        avg_vol = df_in["volume"].rolling(20).mean().iloc[-1]
        latest_vol = df_in["volume"].iloc[-1]
        if pd.isna(avg_vol) or pd.isna(latest_vol):
            return False
        return float(latest_vol) > float(avg_vol) * float(multiplier)

    def estimate_confidence(df_in: pd.DataFrame, *, source: str) -> float:
        if df_in is None or len(df_in) < 30:
            return 0.0
        try:
            if source == "mean_reversion":
                r = float(rsi(df_in["close"].astype(float), 14).iloc[-1])
                if pd.isna(r):
                    return 0.0
                return max(0.0, min(1.0, abs(r - 50.0) / 25.0))
            a = float(adx(df_in, 14).iloc[-1])
            if pd.isna(a):
                return 0.0
            return max(0.0, min(1.0, a / 50.0))
        except Exception:
            return 0.0

    step = max(1, int(bar_step))

    for idx in range(min_bars_required, len(df), step):
        if lookback_bars is not None and lookback_bars > 0:
            start = max(0, idx + 1 - int(lookback_bars))
            df_slice = df.iloc[start : idx + 1]
        else:
            df_slice = df.iloc[: idx + 1]

        price = float(df_slice["close"].iloc[-1])
        bar_index = idx
        prev_state = state

        # Trend filter: avoid buying below SMA50
        sma50 = df_slice["close"].rolling(50).mean().iloc[-1]

        # Unified signal + metadata
        signal, regime, source = unified_signal_with_meta(df_slice)

        # Entry quality gates to increase win-rate / reduce churn
        # Only gate BUY entries; do NOT suppress SELL signals (they are useful for exits).
        if signal == "BUY":
            if pd.isna(sma50) or price < float(sma50):
                signal = "HOLD"
            elif not volume_confirmation(df_slice, multiplier=VOLUME_MULTIPLIER):
                signal = "HOLD"
            else:
                conf = estimate_confidence(df_slice, source=source)
                thr = SAFE_ML_THRESHOLD if PERSONALITY_MODE.upper() == "SAFE" else GROWTH_ML_THRESHOLD
                if conf < float(thr):
                    signal = "HOLD"

        # Update lifecycle
        state, trade_event = lifecycle.update(
            df_1h=df_slice,
            state=state,
            signal=signal,
            regime=regime,
            price=price,
            bar_index=bar_index,
        )

        # Compute PnL and update balance
        pnl_usdt = 0.0
        side = "HOLD"

        if trade_event is not None:
            if trade_event["type"] == "ENTRY":
                side = "BUY"
                pnl_usdt = 0.0

                # Choose qty using both (a) target notional and (b) risk cap to ATR stop.
                entry_price = float(trade_event["entry_price"]) if trade_event.get("entry_price") else price
                if entry_price > 0 and balance > 0:
                    target_notional = min(float(POSITION_USDT), float(balance))
                    target_qty = target_notional / entry_price

                    # Dynamic position sizing by confidence
                    conf = estimate_confidence(df_slice, source=source)
                    qty_multiplier = 0.5 + conf  # 0.5–1.5x position
                    target_notional = min(float(balance), float(target_notional) * float(qty_multiplier))
                    target_qty = target_notional / entry_price

                    stop_price = getattr(state, "atr_stop", None)
                    if stop_price is not None:
                        stop_dist = entry_price - float(stop_price)
                    else:
                        stop_dist = 0.0

                    risk_usdt = float(balance) * float(MAX_RISK_PER_TRADE)
                    risk_qty = (risk_usdt / stop_dist) if stop_dist > 0 else target_qty

                    qty = max(0.0, min(float(target_qty), float(risk_qty)))
                    state.qty = qty
                    state.entry_notional_usdt = qty * entry_price

            elif trade_event["type"] == "PARTIAL":
                side = "SELL"
                pnl_usdt = float(trade_event.get("pnl_usdt") or 0.0)
                balance += float(pnl_usdt)

            elif trade_event["type"] == "EXIT":
                side = "SELL"
                exit_price = float(trade_event.get("price", price))
                pnl_usdt = float(trade_event.get("pnl_usdt") or 0.0)
                pnl_total_usdt = float(trade_event.get("pnl_total_usdt") or pnl_usdt)
                balance += float(pnl_usdt)

                # Save closed trade
                closed_trades.append({
                    "entry_price": prev_state.entry_price,
                    "exit_price": exit_price,
                    "qty": float(trade_event.get("qty")) if trade_event.get("qty") is not None else None,
                    "pnl": pnl_total_usdt,
                    "bar_index_exit": bar_index,
                    "side": "BUY"
                })

                if balance <= 0:
                    if print_rows:
                        print(f"{df_slice.index[-1]},{symbol},{side},{pnl_usdt:.2f},{balance:.2f}")
                        print("Balance depleted. Backtest stopped.")
                    break

            if print_rows:
                print(f"{df_slice.index[-1]},{symbol},{side},{pnl_usdt:.2f},{balance:.2f}")

        else:
            # HOLD (skip console output)
            pass

        trades.append({
            "time": df_slice.index[-1],
            "symbol": symbol,
            "side": side,
            "pnl_usdt": round(pnl_usdt, 2),
            "balance_after": round(balance, 2),
            "regime": regime,
        })

    # --- Compute stats ---
    total_trades = len(closed_trades)
    wins = sum(1 for t in closed_trades if t["pnl"] > 0)
    losses = sum(1 for t in closed_trades if t["pnl"] < 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t["pnl"] for t in closed_trades)

    print(f"\nBacktest completed: {symbol}")
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total PnL: {total_pnl:.2f} USDT")    


def main() -> None:
    symbol = "BTCUSDT"
    timeframe = "1H"

    data_dir = Path("data")
    preferred = [
        data_dir / f"{symbol}_1H.csv",
        data_dir / f"{symbol}_1h.csv",
        data_dir / f"{symbol}_4H.csv",
        data_dir / f"{symbol}_4h.csv",
        data_dir / f"{symbol}_15M.csv",
        data_dir / f"{symbol}_15m.csv",
    ]
    ohlcv_path = next((p for p in preferred if p.exists()), None)
    if ohlcv_path is None:
        candidates = sorted(data_dir.glob(f"{symbol}_*.csv")) if data_dir.exists() else []
        ohlcv_path = candidates[0] if candidates else None

    if ohlcv_path is None:
        expected = data_dir / f"{symbol}_{timeframe}.csv"
        print(f"OHLCV CSV not found: {expected}")
        print(
            "Fetch candles first, e.g.:\n"
            "  python3 scripts/mexc_fetch_ohlcv.py --timeframe 1h --symbols BTCUSDT\n"
            "or point the code to an existing CSV under data/."
        )
        return

    # Derive timeframe label from filename for nicer printing.
    stem = ohlcv_path.stem
    if stem.startswith(f"{symbol}_"):
        timeframe = stem[len(f"{symbol}_") :]

    run_spot_backtest(symbol, timeframe, str(ohlcv_path), starting_balance=100.0)


if __name__ == "__main__":
    main()

