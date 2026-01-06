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

"""Spot backtest runner (proper spot simulation).

- Cash + asset accounting (spot-realistic)
- Signal computed on bar i-1, executed at bar i open (reduces lookahead)
- Intrabar stop/partial handled by lifecycle using current bar high/low
- Zero-fee support (MEXC: 0.000% maker/taker)
- Theoretical max mode (zero-fee + zero-slippage)
- Safety checks (no negative cash/asset, lifecycle-wallet desync detection)
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
    # --- execution params ---
    theoretical_max: bool = False,  # ✅ zero-fee + zero-slippage
    fee_rate: float = 0.0,          # ✅ MEXC spot: 0.000% maker & taker
    slippage_rate: float = 0.0001,  # realistic default; theoretical_max overrides to 0.0
):
    df = load_ohlcv(ohlcv_file).copy()

    if max_candles is not None and max_candles > 0 and len(df) > int(max_candles):
        df = df.tail(int(max_candles))

    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")

    if theoretical_max:
        fee_rate = 0.0
        slippage_rate = 0.0

    mode = "THEORETICAL_MAX" if theoretical_max else "REALISTIC"
    if print_rows:
        print(f"Loaded {len(df)} candles for {symbol} {timeframe}.")
        print(f"Mode: {mode} | fee_rate={fee_rate:.6f} | slippage_rate={slippage_rate:.6f}")

    # --- lifecycle ---
    state = SpotTradeState()
    lifecycle_defaults = {
        "atr_multiplier": 2.0,
        "trail_pct": None,
        "trail_atr_mult": 2.5,
        "partial_profit_pct": 0.05,
        "partial_sell_fraction": 0.3,
        "max_bars_in_trade": 48,
        "cooldown_bars": 3,
    }
    if lifecycle_kwargs:
        lifecycle_defaults.update(lifecycle_kwargs)
    lifecycle = SpotTradeLifecycle(**lifecycle_defaults)

    # --- wallet (spot) ---
    cash_usdt = float(starting_balance)
    asset_qty = 0.0  # base-asset units held (e.g., BTC)
    equity_curve: list[dict] = []
    closed_trades: list[dict] = []

    # --- sizing ---
    POSITION_USDT = 25.0
    MAX_RISK_PER_TRADE = 0.03

    # --- gates ---
    SAFE_ML_THRESHOLD = 0.80
    GROWTH_ML_THRESHOLD = 0.55
    PERSONALITY_MODE = "GROWTH"
    VOLUME_MULTIPLIER = 1.2

    if gate_kwargs:
        SAFE_ML_THRESHOLD = float(gate_kwargs.get("SAFE_ML_THRESHOLD", SAFE_ML_THRESHOLD))
        GROWTH_ML_THRESHOLD = float(gate_kwargs.get("GROWTH_ML_THRESHOLD", GROWTH_ML_THRESHOLD))
        PERSONALITY_MODE = str(gate_kwargs.get("PERSONALITY_MODE", PERSONALITY_MODE))
        VOLUME_MULTIPLIER = float(gate_kwargs.get("VOLUME_MULTIPLIER", VOLUME_MULTIPLIER))

    # detect_regime needs ~200 bars (you mentioned SMA200), keep a buffer
    min_bars_required = 220
    step = max(1, int(bar_step))

    # --- safety ---
    EPS_QTY = 1e-12
    EPS_USDT = 1e-9

    def assert_invariants(*, cash: float, asset: float, context: str) -> None:
        if cash < -EPS_USDT:
            raise RuntimeError(f"[SAFETY] Negative cash ({cash}) at {context}")
        if asset < -EPS_QTY:
            raise RuntimeError(f"[SAFETY] Negative asset ({asset}) at {context}")

    def assert_desync(*, in_position: bool, asset: float, context: str) -> None:
        if in_position and asset <= EPS_QTY:
            raise RuntimeError(f"[SAFETY] Desync: in_position=True but asset_qty={asset} at {context}")
        if (not in_position) and asset > EPS_QTY:
            raise RuntimeError(f"[SAFETY] Desync: in_position=False but asset_qty={asset} at {context}")

    # --- helpers ---
    def fee(amount_usdt: float) -> float:
        return float(amount_usdt) * float(fee_rate)

    def apply_buy_fill(px: float) -> float:
        # adverse slippage for buy
        return float(px) * (1.0 + float(slippage_rate))

    def apply_sell_fill(px: float) -> float:
        # adverse slippage for sell
        return float(px) * (1.0 - float(slippage_rate))

    def volume_confirmation(df_in: pd.DataFrame, multiplier: float = 1.2) -> bool:
        if df_in is None or len(df_in) < 20:
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

    for idx in range(min_bars_required, len(df), step):
        assert_invariants(cash=cash_usdt, asset=asset_qty, context=f"start bar idx={idx}")

        # --- signal slice: up to previous bar (no lookahead) ---
        if idx <= 0:
            continue

        if lookback_bars is not None and lookback_bars > 0:
            start_sig = max(0, idx - int(lookback_bars))
            df_sig = df.iloc[start_sig:idx]  # bars [start_sig .. idx-1]
        else:
            df_sig = df.iloc[:idx]

        # Current bar (execution/management bar)
        row = df.iloc[idx]
        open_i = float(row["open"])
        close_i = float(row["close"])

        # Trend filter computed on df_sig (no lookahead)
        sma50 = df_sig["close"].rolling(50).mean().iloc[-1] if len(df_sig) >= 50 else float("nan")

        signal, regime, source = unified_signal_with_meta(df_sig)

        # Gate BUY only (never suppress exits)
        if signal == "BUY":
            last_close_prev = float(df_sig["close"].iloc[-1])
            if pd.isna(sma50) or last_close_prev < float(sma50):
                signal = "HOLD"
            elif not volume_confirmation(df_sig, multiplier=VOLUME_MULTIPLIER):
                signal = "HOLD"
            else:
                conf = estimate_confidence(df_sig, source=source)
                thr = SAFE_ML_THRESHOLD if PERSONALITY_MODE.upper() == "SAFE" else GROWTH_ML_THRESHOLD
                if conf < float(thr):
                    signal = "HOLD"

        # --- management slice includes current bar so lifecycle can use its high/low ---
        if lookback_bars is not None and lookback_bars > 0:
            start_mgmt = max(0, idx + 1 - int(lookback_bars))
            df_mgmt = df.iloc[start_mgmt: idx + 1]
        else:
            df_mgmt = df.iloc[: idx + 1]

        # Execute signal actions at open_i (ENTRY / signal-exit price),
        # but lifecycle can still trigger stop/partial via df_mgmt last bar high/low.
        state, trade_event = lifecycle.update(
            df_1h=df_mgmt,
            state=state,
            signal=signal,
            regime=regime,
            price=float(open_i),
            bar_index=int(idx),
        )

        if trade_event is not None:
            etype = str(trade_event.get("type", ""))

            if etype == "ENTRY":
                # BUY at open with slippage
                entry_price = apply_buy_fill(float(trade_event.get("entry_price", open_i)))

                # Stop from event is ideal; fallback to state.atr_stop
                stop_price = trade_event.get("atr_stop", None)
                if stop_price is None:
                    stop_price = getattr(state, "atr_stop", None)
                stop_price = float(stop_price) if stop_price is not None else None

                stop_dist = (entry_price - stop_price) if (stop_price is not None) else 0.0

                # Target notional limited by cash
                target_notional = min(float(POSITION_USDT), float(cash_usdt))

                # Dynamic sizing by confidence
                conf = estimate_confidence(df_sig, source=source)
                qty_multiplier = 0.5 + conf  # 0.5–1.5x
                target_notional = min(float(cash_usdt), float(target_notional) * float(qty_multiplier))

                target_qty = (target_notional / entry_price) if entry_price > 0 else 0.0

                # Risk cap to stop distance
                risk_usdt = float(cash_usdt) * float(MAX_RISK_PER_TRADE)
                risk_qty = (risk_usdt / stop_dist) if (stop_dist and stop_dist > 0) else target_qty

                qty = max(0.0, min(float(target_qty), float(risk_qty)))

                if qty <= EPS_QTY or entry_price <= 0:
                    # Cancel entry cleanly (cannot afford / invalid)
                    state = SpotTradeState(cooldown_until=state.cooldown_until)
                else:
                    cost = qty * entry_price
                    total_cost = cost + fee(cost)

                    # SAFETY: scale down if slightly unaffordable
                    if total_cost > cash_usdt + EPS_USDT:
                        denom = entry_price * (1.0 + float(fee_rate))
                        affordable_qty = (cash_usdt / denom) if denom > 0 else 0.0
                        qty = min(qty, max(0.0, affordable_qty))
                        cost = qty * entry_price
                        total_cost = cost + fee(cost)

                    if qty <= EPS_QTY or total_cost > cash_usdt + EPS_USDT:
                        state = SpotTradeState(cooldown_until=state.cooldown_until)
                    else:
                        cash_usdt -= total_cost
                        asset_qty += qty

                        state.qty = qty
                        state.entry_notional_usdt = cost

                        assert_invariants(cash=cash_usdt, asset=asset_qty, context=f"after BUY idx={idx}")
                        assert_desync(in_position=state.in_position, asset=asset_qty, context=f"after BUY idx={idx}")

                        if print_rows:
                            print(
                                f"{df.index[idx]},{symbol},BUY,0.00,"
                                f"cash={cash_usdt:.2f},asset={asset_qty:.8f}"
                            )

            elif etype == "PARTIAL":
                exec_price = apply_sell_fill(float(trade_event.get("price", open_i)))
                qty_sold = float(trade_event.get("qty_sold") or 0.0)
                qty_sold = max(0.0, qty_sold)

                # SAFETY: cannot sell more than held
                qty_sold = min(qty_sold, asset_qty)

                if qty_sold > EPS_QTY and exec_price > 0:
                    proceeds = qty_sold * exec_price
                    cash_usdt += (proceeds - fee(proceeds))
                    asset_qty -= qty_sold

                    assert_invariants(cash=cash_usdt, asset=asset_qty, context=f"after PARTIAL idx={idx}")
                    assert_desync(in_position=state.in_position, asset=asset_qty, context=f"after PARTIAL idx={idx}")

                    pnl_partial = float(trade_event.get("pnl_usdt") or 0.0)
                    if print_rows:
                        print(
                            f"{df.index[idx]},{symbol},SELL_PARTIAL,{pnl_partial:.2f},"
                            f"cash={cash_usdt:.2f},asset={asset_qty:.8f}"
                        )

            elif etype == "EXIT":
                exit_price = apply_sell_fill(float(trade_event.get("price", open_i)))
                qty_to_sell = float(trade_event.get("qty") or 0.0)
                qty_to_sell = max(0.0, qty_to_sell)

                # SAFETY: cannot sell more than held
                qty_to_sell = min(qty_to_sell, asset_qty)

                if qty_to_sell > EPS_QTY and exit_price > 0:
                    proceeds = qty_to_sell * exit_price
                    cash_usdt += (proceeds - fee(proceeds))
                    asset_qty -= qty_to_sell

                    assert_invariants(cash=cash_usdt, asset=asset_qty, context=f"after EXIT idx={idx}")
                    # After EXIT, lifecycle returns a flat state (cooldown-only), so desync check must reflect that:
                    assert_desync(in_position=state.in_position, asset=asset_qty, context=f"after EXIT idx={idx}")

                pnl_leg = float(trade_event.get("pnl_usdt") or 0.0)
                pnl_total = float(trade_event.get("pnl_total_usdt") or pnl_leg)

                closed_trades.append(
                    {
                        "entry_index": int(trade_event.get("entry_index") or -1),
                        "exit_index": int(trade_event.get("bar_index") or idx),
                        "entry_price": float(trade_event.get("entry_price") or 0.0),
                        "exit_price": float(trade_event.get("price") or exit_price),
                        "qty_sold_exit": float(qty_to_sell),
                        "pnl_total_usdt": pnl_total,
                        "partial_taken": bool(trade_event.get("partial_taken")),
                        "regime": str(trade_event.get("regime")),
                    }
                )

                if print_rows:
                    print(
                        f"{df.index[idx]},{symbol},SELL_EXIT,{pnl_leg:.2f},"
                        f"cash={cash_usdt:.2f},asset={asset_qty:.8f}"
                    )

        # --- mark-to-market equity at close ---
        equity = float(cash_usdt) + float(asset_qty) * float(close_i)
        equity_curve.append(
            {
                "time": df.index[idx],
                "cash_usdt": round(cash_usdt, 6),
                "asset_qty": round(asset_qty, 12),
                "close": round(close_i, 6),
                "equity": round(equity, 6),
                "regime": regime,
            }
        )

        if equity <= 0.0:
            if print_rows:
                print("Equity depleted. Backtest stopped.")
            break

    # --- stats (closed trades) ---
    total_trades = len(closed_trades)
    wins = sum(1 for t in closed_trades if t["pnl_total_usdt"] > 0)
    losses = sum(1 for t in closed_trades if t["pnl_total_usdt"] < 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    total_pnl = sum(t["pnl_total_usdt"] for t in closed_trades)

    final_equity = equity_curve[-1]["equity"] if equity_curve else float(starting_balance)

    print(f"\nBacktest completed: {symbol}")
    print(f"Closed trades: {total_trades}")
    print(f"Wins: {wins} Losses: {losses} Win rate: {win_rate:.2f}%")
    print(f"Total (lifecycle) trade PnL sum: {total_pnl:.2f} USDT")
    print(f"Final equity (cash + asset MTM): {final_equity:.2f} USDT")


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

    run_spot_backtest(
        symbol,
        timeframe,
        str(ohlcv_path),
        starting_balance=100.0,
        theoretical_max=False,   # set True for perfect fills (0 fee, 0 slippage)
        fee_rate=0.0,            # MEXC zero fee
        slippage_rate=0.0001,    # ignored if theoretical_max=True
        print_rows=True,
    )


if __name__ == "__main__":
    main()

