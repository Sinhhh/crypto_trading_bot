from __future__ import annotations

from typing import Literal, Optional

import os
import sys
from pathlib import Path

import pandas as pd

if __package__ is None:  # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_trading.io.loader import load_ohlcv
from crypto_trading.lifecycle.spot_lifecycle import SpotTradeState, SpotTradeLifecycle
from crypto_trading.unified.unified_signal import (
    unified_signal_with_meta,
    unified_signal_mtf,
)

from crypto_trading.indicators.volatility import atr
from crypto_trading.indicators.momentum import rsi, adx
from crypto_trading.indicators.moving_averages import ema


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
    # --- execution ---
    theoretical_max: bool = False,
    fee_rate: float = 0.0,  # ✅ MEXC spot 0.000%
    slippage_rate: float = 0.0001,  # ignored if theoretical_max=True
    # --- bot selection ---
    entry_tf: Literal["15m", "1h", "4h"] | None = None,
    context_files: (
        dict[str, str] | None
    ) = None,  # {"1h": "...csv", "4h": "...csv"} for 15m bot
):
    """
    Spot backtest runner (proper spot simulation).

    Supports:
    - 15m bot (entries on 15m + context 1h/4h via unified_signal_mtf)
    - 1h bot (unified_signal_with_meta on 1h data)
    - 4h bot (unified_signal_with_meta on 4h data)

    Execution model:
    - Signal computed on bar i-1 (closed candle)
    - Entry/Signal-exit executed at bar i OPEN (no lookahead)
    - Stops/partials are intrabar via lifecycle using bar i HIGH/LOW

    Notes:
    - fee_rate defaults to 0.0 for MEXC zero-fee spot
    - theoretical_max=True forces fee_rate=0 and slippage_rate=0
    - Safety checks included
    """

    # -------------------------
    # Load entry timeframe data
    # -------------------------
    df_entry = load_ohlcv(ohlcv_file).copy()
    if max_candles is not None and max_candles > 0 and len(df_entry) > int(max_candles):
        df_entry = df_entry.tail(int(max_candles))

    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df_entry.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns in entry data: {missing}")

    # Normalize entry_tf selection
    tf_norm = timeframe.lower().replace("min", "m")
    if entry_tf is None:
        # infer from timeframe label
        if "15" in tf_norm:
            entry_tf = "15m"
        elif "4h" in tf_norm or "240" in tf_norm:
            entry_tf = "4h"
        else:
            entry_tf = "1h"

    # Theoretical max mode
    if theoretical_max:
        fee_rate = 0.0
        slippage_rate = 0.0

    mode_str = "THEORETICAL_MAX" if theoretical_max else "REALISTIC"
    if print_rows:
        print(f"Loaded {len(df_entry)} candles for {symbol} {timeframe}.")
        print(
            f"Mode: {mode_str} | fee_rate={fee_rate:.6f} | slippage_rate={slippage_rate:.6f} | entry_tf={entry_tf}"
        )

    # -------------------------
    # Load context data if 15m bot
    # -------------------------
    df_1h: Optional[pd.DataFrame] = None
    df_4h: Optional[pd.DataFrame] = None
    df_1d: Optional[pd.DataFrame] = None

    if context_files:
        # For entry_tf='15m', 1h/4h context is REQUIRED for unified_signal_mtf.
        if entry_tf == "15m" and (
            ("1h" not in context_files) or ("4h" not in context_files)
        ):
            raise ValueError(
                "For entry_tf='15m', you must provide context_files={'1h': <csv>, '4h': <csv>} "
                "containing CLOSED 1H and 4H candles."
            )

        # For other entry_tfs, context is optional and only used for trade filters.
        if "1h" in context_files:
            df_1h = load_ohlcv(context_files["1h"]).copy()
        if "4h" in context_files:
            df_4h = load_ohlcv(context_files["4h"]).copy()
        if "1d" in context_files:
            df_1d = load_ohlcv(context_files["1d"]).copy()

        for name, dfx in [("1h", df_1h), ("4h", df_4h), ("1d", df_1d)]:
            if dfx is None:
                continue
            missing_ctx = needed - set(dfx.columns)
            if missing_ctx:
                raise ValueError(
                    f"OHLCV missing columns in {name} context data: {missing_ctx}"
                )

    # -------------------------
    # Lifecycle
    # -------------------------
    state = SpotTradeState()
    lifecycle_defaults = {
        "atr_multiplier": 2.2,
        "trail_atr_mult": 3.8,
        "partial_profit_pct": 0.14,
        "partial_sell_fraction": 0.2,
        "max_bars_in_trade": 96,  # 4 days
        "cooldown_bars": 6,
    }
    if lifecycle_kwargs:
        lifecycle_defaults.update(lifecycle_kwargs)
    lifecycle = SpotTradeLifecycle(**lifecycle_defaults)

    # -------------------------
    # Wallet
    # -------------------------
    cash_usdt = float(starting_balance)
    asset_qty = 0.0

    closed_trades: list[dict] = []
    equity_curve: list[dict] = []

    # Sizing
    POSITION_USDT = 25.0
    MAX_RISK_PER_TRADE = 0.03

    # Gates (SAFE/GROWTH)
    SAFE_ML_THRESHOLD = 0.80
    GROWTH_ML_THRESHOLD = 0.55
    PERSONALITY_MODE = "GROWTH"
    VOLUME_MULTIPLIER = 1.2
    SMA_LEN = 50  # timeframe-dependent; override with gate_kwargs if needed

    # Optional win-rate focused filters (opt-in via gate_kwargs)
    ALLOWED_REGIMES: set[str] | None = None
    MIN_ADX_FOR_BUY: float | None = None
    MIN_ATR_RATIO_FOR_BUY: float | None = None
    HTF_TREND_FOR_BUY: bool = False
    HTF_TF: str | None = None
    HTF_EMA_LEN: int = 200
    MIN_HTF_BARS: int = 50

    if gate_kwargs:
        SAFE_ML_THRESHOLD = float(
            gate_kwargs.get("SAFE_ML_THRESHOLD", SAFE_ML_THRESHOLD)
        )
        GROWTH_ML_THRESHOLD = float(
            gate_kwargs.get("GROWTH_ML_THRESHOLD", GROWTH_ML_THRESHOLD)
        )
        PERSONALITY_MODE = str(gate_kwargs.get("PERSONALITY_MODE", PERSONALITY_MODE))
        VOLUME_MULTIPLIER = float(
            gate_kwargs.get("VOLUME_MULTIPLIER", VOLUME_MULTIPLIER)
        )
        SMA_LEN = int(gate_kwargs.get("SMA_LEN", SMA_LEN))

        allowed = gate_kwargs.get("ALLOWED_REGIMES", None)
        if allowed is not None:
            if isinstance(allowed, str):
                # e.g. "TREND_UP,CROSS_UP"
                allowed_list = [x.strip() for x in allowed.split(",") if x.strip()]
            else:
                allowed_list = list(allowed)
            ALLOWED_REGIMES = {
                str(x).strip().upper() for x in allowed_list if str(x).strip()
            }

        min_adx = gate_kwargs.get("MIN_ADX_FOR_BUY", None)
        if min_adx is not None:
            MIN_ADX_FOR_BUY = float(min_adx)

        min_atr_ratio = gate_kwargs.get("MIN_ATR_RATIO_FOR_BUY", None)
        if min_atr_ratio is not None:
            MIN_ATR_RATIO_FOR_BUY = float(min_atr_ratio)

        HTF_TREND_FOR_BUY = bool(gate_kwargs.get("HTF_TREND_FOR_BUY", HTF_TREND_FOR_BUY))
        HTF_TF = gate_kwargs.get("HTF_TF", HTF_TF)
        HTF_EMA_LEN = int(gate_kwargs.get("HTF_EMA_LEN", HTF_EMA_LEN))
        MIN_HTF_BARS = int(gate_kwargs.get("MIN_HTF_BARS", MIN_HTF_BARS))

    # Precompute HTF EMA if we will use HTF trend gating.
    htf_df: Optional[pd.DataFrame] = None
    htf_ema_col: str | None = None
    if HTF_TREND_FOR_BUY:
        # Prefer explicit selection, otherwise: 1d if available, else 4h.
        tf_key = str(HTF_TF).strip().lower() if HTF_TF is not None else None
        if tf_key in {"1d", "d", "1day", "day", "daily"}:
            htf_df = df_1d
        elif tf_key in {"4h", "240"}:
            htf_df = df_4h
        elif tf_key in {"1h", "60"}:
            htf_df = df_1h
        else:
            htf_df = df_1d if df_1d is not None else df_4h

        if htf_df is not None:
            htf_ema_col = f"_htf_ema_{int(HTF_EMA_LEN)}"
            if htf_ema_col not in htf_df.columns:
                htf_df[htf_ema_col] = ema(htf_df["close"].astype(float), int(HTF_EMA_LEN))

    mode = PERSONALITY_MODE.upper()

    # Precompute common indicator series once.
    # This reduces per-bar rolling/indicator costs even when lookback_bars is used.
    # (It also reflects the usual "use all history up to now" behavior.)
    use_precomputed = True

    # detect_regime needs >=200 bars on the timeframe it runs on.
    # For 1h/4h bots, this is okay. For 15m bot, regime comes from df_4h/df_1h inside unified_signal_mtf.
    min_bars_required = int(gate_kwargs.get("MIN_BARS", 220)) if gate_kwargs else 220
    step = max(1, int(bar_step))

    # -------------------------
    # Precompute common series
    # -------------------------
    if "_sma_gate" not in df_entry.columns:
        df_entry["_sma_gate"] = df_entry["close"].rolling(int(SMA_LEN)).mean()
    if "_vol_sma20" not in df_entry.columns:
        df_entry["_vol_sma20"] = df_entry["volume"].rolling(20).mean()
    if "_rsi14" not in df_entry.columns:
        df_entry["_rsi14"] = rsi(df_entry["close"].astype(float), 14)
    if "_adx14" not in df_entry.columns:
        df_entry["_adx14"] = adx(df_entry, 14)
    if "atr_14" not in df_entry.columns:
        df_entry["atr_14"] = atr(df_entry, 14)

    # Cache hot columns as arrays (faster than repeated .iloc row access in the loop).
    idx_values = df_entry.index
    open_arr = df_entry["open"].to_numpy()
    close_arr = df_entry["close"].to_numpy()
    vol_arr = df_entry["volume"].to_numpy()
    sma_gate_arr = df_entry["_sma_gate"].to_numpy()
    vol_sma20_arr = df_entry["_vol_sma20"].to_numpy()

    # -------------------------
    # Safety
    # -------------------------
    EPS_QTY = 1e-12
    EPS_USDT = 1e-9

    def assert_invariants(*, cash: float, asset: float, context: str) -> None:
        if cash < -EPS_USDT:
            raise RuntimeError(f"[SAFETY] Negative cash ({cash}) at {context}")
        if asset < -EPS_QTY:
            raise RuntimeError(f"[SAFETY] Negative asset ({asset}) at {context}")

    def assert_desync(*, in_position: bool, asset: float, context: str) -> None:
        if in_position and asset <= EPS_QTY:
            raise RuntimeError(
                f"[SAFETY] Desync: in_position=True but asset_qty={asset} at {context}"
            )
        if (not in_position) and asset > EPS_QTY:
            raise RuntimeError(
                f"[SAFETY] Desync: in_position=False but asset_qty={asset} at {context}"
            )

    # -------------------------
    # Helpers
    # -------------------------
    def fee(amount_usdt: float) -> float:
        return float(amount_usdt) * float(fee_rate)

    def apply_buy_fill(px: float) -> float:
        return float(px) * (1.0 + float(slippage_rate))

    def apply_sell_fill(px: float) -> float:
        return float(px) * (1.0 - float(slippage_rate))

    def volume_confirmation(df_in: pd.DataFrame, multiplier: float = 1.2) -> bool:
        if df_in is None or len(df_in) < 20:
            return False
        if "_vol_sma20" in df_in.columns:
            avg_vol = df_in["_vol_sma20"].iloc[-1]
        else:
            avg_vol = df_in["volume"].rolling(20).mean().iloc[-1]

        latest_vol = df_in["volume"].iloc[-1]
        if pd.isna(avg_vol) or pd.isna(latest_vol):
            return False
        return float(latest_vol) > float(avg_vol) * float(multiplier)

    def estimate_confidence(df_in: pd.DataFrame, *, source: str) -> float:
        """
        Heuristic confidence:
        - mean_reversion: RSI distance from 50
        - others: ADX strength
        """
        if df_in is None or len(df_in) < 30:
            return 0.0
        try:
            if source == "mean_reversion":
                if "_rsi14" in df_in.columns:
                    r = float(df_in["_rsi14"].iloc[-1])
                else:
                    r = float(rsi(df_in["close"].astype(float), 14).iloc[-1])
                if pd.isna(r):
                    return 0.0
                return max(0.0, min(1.0, abs(r - 50.0) / 25.0))
            if "_adx14" in df_in.columns:
                a = float(df_in["_adx14"].iloc[-1])
            else:
                a = float(adx(df_in, 14).iloc[-1])
            if pd.isna(a):
                return 0.0
            return max(0.0, min(1.0, a / 50.0))
        except Exception:
            return 0.0

    def threshold_for() -> float:
        return float(SAFE_ML_THRESHOLD if mode == "SAFE" else GROWTH_ML_THRESHOLD)

    # -------------------------
    # Main loop
    # -------------------------
    for idx in range(min_bars_required, len(df_entry), step):
        assert_invariants(
            cash=cash_usdt, asset=asset_qty, context=f"start bar idx={idx}"
        )

        # Compute signals on bars up to idx-1 (no lookahead)
        if lookback_bars is not None and lookback_bars > 0:
            # Keep at least min_bars_required so regime/signal functions aren't starved.
            sig_window = max(int(lookback_bars), int(min_bars_required))
            start_sig = max(0, idx - sig_window)
            df_sig_entry = df_entry.iloc[start_sig:idx]
        else:
            df_sig_entry = df_entry.iloc[:idx]

        # current bar is idx (execution/management bar)
        open_i = float(open_arr[idx])
        close_i = float(close_arr[idx])

        # Trend filter on entry timeframe (configurable length)
        sma_val = float(sma_gate_arr[idx - 1])

        # Choose signal source based on bot timeframe
        if entry_tf == "15m":
            assert df_1h is not None and df_4h is not None

            # align context slices by timestamp (use only closed candles <= current decision time)
            now_ts = df_sig_entry.index[-1]  # last closed 15m bar
            df_1h_sig = df_1h.loc[:now_ts]
            df_4h_sig = df_4h.loc[:now_ts]

            signal, regime, source = unified_signal_mtf(
                df_sig_entry, df_1h_sig, df_4h_sig
            )
        elif entry_tf == "1h":
            signal, regime, source = unified_signal_with_meta(df_sig_entry)
        elif entry_tf == "4h":
            signal, regime, source = unified_signal_with_meta(df_sig_entry)
        else:
            raise ValueError(f"Unsupported entry_tf={entry_tf}")

        # Gate BUY entries only
        if signal == "BUY":
            # Optional regime filter (often increases win rate by skipping chop)
            if (
                ALLOWED_REGIMES is not None
                and str(regime).upper() not in ALLOWED_REGIMES
            ):
                signal = "HOLD"

        if signal == "BUY":
            # Optional HTF trend filter: require HTF close above HTF EMA.
            if HTF_TREND_FOR_BUY:
                if htf_df is None or htf_ema_col is None:
                    signal = "HOLD"
                else:
                    try:
                        now_ts = df_sig_entry.index[-1]  # last closed entry bar
                        df_htf_sig = htf_df.loc[:now_ts]
                        if df_htf_sig is None or len(df_htf_sig) < int(MIN_HTF_BARS):
                            signal = "HOLD"
                        else:
                            htf_close = float(df_htf_sig["close"].iloc[-1])
                            htf_ema_val = float(df_htf_sig[htf_ema_col].iloc[-1])
                            if pd.isna(htf_close) or pd.isna(htf_ema_val):
                                signal = "HOLD"
                            elif float(htf_close) < float(htf_ema_val):
                                signal = "HOLD"
                    except Exception:
                        signal = "HOLD"

        if signal == "BUY":
            # Optional ATR volatility filter (high-vol only): atr_14 / close >= threshold.
            if MIN_ATR_RATIO_FOR_BUY is not None:
                try:
                    atr_val = float(df_sig_entry["atr_14"].iloc[-1])
                    close_val = float(df_sig_entry["close"].iloc[-1])
                    if pd.isna(atr_val) or pd.isna(close_val) or close_val <= 0:
                        signal = "HOLD"
                    else:
                        atr_ratio = float(atr_val) / float(close_val)
                        if float(atr_ratio) < float(MIN_ATR_RATIO_FOR_BUY):
                            signal = "HOLD"
                except Exception:
                    signal = "HOLD"

        if signal == "BUY":
            # SMA gate
            if pd.isna(sma_val) or float(df_sig_entry["close"].iloc[-1]) < float(
                sma_val
            ):
                signal = "HOLD"
            else:
                # Optional ADX gate (trend strength)
                if MIN_ADX_FOR_BUY is not None:
                    try:
                        adx_val = (
                            float(df_sig_entry["_adx14"].iloc[-1])
                            if "_adx14" in df_sig_entry.columns
                            else float(adx(df_sig_entry, 14).iloc[-1])
                        )
                        if pd.isna(adx_val) or float(adx_val) < float(MIN_ADX_FOR_BUY):
                            signal = "HOLD"
                    except Exception:
                        signal = "HOLD"

                # Volume gate
                vol_ok = None
                avg_vol = vol_sma20_arr[idx - 1]
                latest_vol = vol_arr[idx - 1]
                if not (pd.isna(avg_vol) or pd.isna(latest_vol)):
                    vol_ok = float(latest_vol) > float(avg_vol) * float(
                        VOLUME_MULTIPLIER
                    )
                if vol_ok is None:
                    vol_ok = volume_confirmation(
                        df_sig_entry, multiplier=VOLUME_MULTIPLIER
                    )

                if not bool(vol_ok):
                    signal = "HOLD"
                else:
                    conf = estimate_confidence(df_sig_entry, source=source)
                    if conf < threshold_for():
                        signal = "HOLD"

        # Optional: dynamic ATR stop computed from CLOSED candles only.
        # This aligns sizing with the actual lifecycle stop and avoids lookahead.
        atr_stop_override = None
        if signal == "BUY" and (not state.in_position):
            try:
                atr_val = float(df_sig_entry["atr_14"].iloc[-1])
                recent_vol = (
                    df_sig_entry["close"].pct_change().rolling(20).std().iloc[-1]
                )
                if pd.isna(recent_vol):
                    atr_mult = float(lifecycle.atr_multiplier)
                elif float(recent_vol) < 0.005:
                    atr_mult = 1.5
                else:
                    atr_mult = 3.0

                if (not pd.isna(atr_val)) and atr_val > 0:
                    # Entry happens at bar idx OPEN; stop must be derived from closed-candle ATR.
                    atr_stop_override = float(open_i) - float(atr_mult) * float(atr_val)
            except Exception:
                atr_stop_override = None

        # df_mgmt includes current bar for intrabar stop/partial handling
        if lookback_bars is not None and lookback_bars > 0:
            start_mgmt = max(0, idx + 1 - int(lookback_bars))
            df_mgmt = df_entry.iloc[start_mgmt : idx + 1]
        else:
            df_mgmt = df_entry.iloc[: idx + 1]

        # lifecycle update: signal-exec at open_i, intrabar uses high/low from df_mgmt last row
        state, trade_event = lifecycle.update(
            df_1h=df_mgmt,
            state=state,
            signal=signal,
            regime=regime,
            price=float(open_i),
            bar_index=int(idx),
            atr_stop_override=atr_stop_override,
        )

        if trade_event is not None:
            etype = str(trade_event.get("type", ""))

            if etype == "ENTRY":
                entry_price = apply_buy_fill(
                    float(trade_event.get("entry_price", open_i))
                )

                # Use lifecycle's atr_stop (now deterministic via atr_stop_override).
                stop_price = trade_event.get("atr_stop", None)
                if stop_price is None:
                    stop_price = getattr(state, "atr_stop", None)
                stop_price = float(stop_price) if stop_price is not None else None
                stop_dist = (
                    max(entry_price - float(stop_price), 1e-6)
                    if stop_price is not None
                    else 0.0
                )

                # Confidence-scaled position sizing
                conf = estimate_confidence(df_sig_entry, source=source)
                qty_multiplier = 0.5 + conf * 2.0  # 0.5x–2.5x scaling
                target_notional = min(float(POSITION_USDT), float(cash_usdt))
                target_notional = min(
                    float(cash_usdt), target_notional * float(qty_multiplier)
                )
                target_qty = target_notional / entry_price

                # Risk-based adjustment using ATR stop
                risk_usdt = float(cash_usdt) * float(MAX_RISK_PER_TRADE)
                risk_qty = (
                    (risk_usdt / stop_dist)
                    if (stop_dist and stop_dist > 0)
                    else target_qty
                )

                qty = max(0.0, min(float(target_qty), float(risk_qty)))

                # Cancel if qty too small or price invalid
                if qty <= EPS_QTY or entry_price <= 0:
                    # cancel entry
                    state = SpotTradeState(cooldown_until=state.cooldown_until)
                else:
                    # Execute trade
                    cost = qty * entry_price
                    total_cost = cost + fee(cost)

                    # scale down if slightly unaffordable
                    if total_cost > cash_usdt + EPS_USDT:
                        affordable_qty = cash_usdt / (entry_price * (1.0 + fee_rate))
                        qty = min(qty, max(0.0, affordable_qty))
                        cost = qty * entry_price
                        total_cost = cost + fee(cost)

                    if qty <= EPS_QTY or total_cost > cash_usdt + EPS_USDT:
                        # cancel entry
                        state = SpotTradeState(cooldown_until=state.cooldown_until)
                    else:
                        # execute once
                        cash_usdt -= total_cost
                        asset_qty += qty

                        state.qty = qty
                        state.entry_notional_usdt = cost

                        assert_invariants(
                            cash=cash_usdt,
                            asset=asset_qty,
                            context=f"after BUY idx={idx}",
                        )
                        assert_desync(
                            in_position=state.in_position,
                            asset=asset_qty,
                            context=f"after BUY idx={idx}",
                        )

                        if print_rows:
                            print(
                                f"{idx_values[idx]},{symbol},BUY,0.00,cash={cash_usdt:.2f},asset={asset_qty:.8f}"
                            )

            elif etype == "PARTIAL":
                exec_price = apply_sell_fill(float(trade_event.get("price", open_i)))
                qty_sold = float(trade_event.get("qty_sold") or 0.0)
                qty_sold = max(0.0, qty_sold)
                qty_sold = min(qty_sold, asset_qty)

                if qty_sold > EPS_QTY and exec_price > 0:
                    proceeds = qty_sold * exec_price
                    cash_usdt += proceeds - fee(proceeds)
                    asset_qty -= qty_sold

                    assert_invariants(
                        cash=cash_usdt,
                        asset=asset_qty,
                        context=f"after PARTIAL idx={idx}",
                    )
                    assert_desync(
                        in_position=state.in_position,
                        asset=asset_qty,
                        context=f"after PARTIAL idx={idx}",
                    )

                    pnl_partial = float(trade_event.get("pnl_usdt") or 0.0)
                    if print_rows:
                        print(
                            f"{idx_values[idx]},{symbol},SELL_PARTIAL,{pnl_partial:.2f},cash={cash_usdt:.2f},asset={asset_qty:.8f}"
                        )

            elif etype == "EXIT":
                exit_price = apply_sell_fill(float(trade_event.get("price", open_i)))
                qty_to_sell = float(trade_event.get("qty") or 0.0)
                qty_to_sell = max(0.0, qty_to_sell)
                qty_to_sell = min(qty_to_sell, asset_qty)

                if qty_to_sell > EPS_QTY and exit_price > 0:
                    proceeds = qty_to_sell * exit_price
                    cash_usdt += proceeds - fee(proceeds)
                    asset_qty -= qty_to_sell

                    assert_invariants(
                        cash=cash_usdt, asset=asset_qty, context=f"after EXIT idx={idx}"
                    )
                    assert_desync(
                        in_position=state.in_position,
                        asset=asset_qty,
                        context=f"after EXIT idx={idx}",
                    )

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
                        "source": str(source),
                    }
                )

                if print_rows:
                    print(
                        f"{idx_values[idx]},{symbol},SELL_EXIT,{pnl_leg:.2f},cash={cash_usdt:.2f},asset={asset_qty:.8f}"
                    )

        # equity mark-to-market at close
        equity = float(cash_usdt) + float(asset_qty) * float(close_i)
        equity_curve.append(
            {
                "time": idx_values[idx],
                "cash_usdt": round(cash_usdt, 6),
                "asset_qty": round(asset_qty, 12),
                "close": round(close_i, 6),
                "equity": round(equity, 6),
                "regime": regime,
                "signal": signal,
                "source": source,
            }
        )

        if equity <= 0.0:
            if print_rows:
                print("Equity depleted. Backtest stopped.")
            break

    # -------------------------
    # Stats
    # -------------------------
    total_trades = len(closed_trades)
    wins = sum(1 for t in closed_trades if t["pnl_total_usdt"] > 0)
    losses = sum(1 for t in closed_trades if t["pnl_total_usdt"] < 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    total_pnl = sum(float(t["pnl_total_usdt"]) for t in closed_trades)
    final_equity = (
        equity_curve[-1]["equity"] if equity_curve else float(starting_balance)
    )

    print(f"\nBacktest completed: {symbol}")
    print(f"Closed trades: {total_trades}")
    print(f"Wins: {wins} Losses: {losses} Win rate: {win_rate:.2f}%")
    print(f"Total (lifecycle) trade PnL sum: {total_pnl:.2f} USDT")
    print(f"Final equity (cash + asset MTM): {final_equity:.2f} USDT")


def main() -> None:
    # Allow piping output without crashing (e.g. `... | head`).
    try:  # pragma: no cover
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass

    symbol = "ETHUSDT"

    data_dir = Path("data/raw")
    preferred = [
        data_dir / f"{symbol}_15M.csv",
        data_dir / f"{symbol}_1H.csv",
        data_dir / f"{symbol}_4H.csv",
    ]
    ohlcv_path = next((p for p in preferred if p.exists()), None)
    if ohlcv_path is None:
        candidates = (
            sorted(data_dir.glob(f"{symbol}_*.csv")) if data_dir.exists() else []
        )
        ohlcv_path = candidates[0] if candidates else None

    if ohlcv_path is None:
        print(f"OHLCV CSV not found under: {data_dir}")
        return

    stem = ohlcv_path.stem
    timeframe = (
        stem[len(f"{symbol}_") :] if stem.startswith(f"{symbol}_") else "UNKNOWN"
    )

    entry_tf: Literal["15m", "1h", "4h"]
    tf_lower = timeframe.lower()
    if "15" in tf_lower:
        entry_tf = "15m"
    elif "4h" in tf_lower or "240" in tf_lower:
        entry_tf = "4h"
    else:
        entry_tf = "1h"

    context_files = None
    if entry_tf == "15m":
        ctx_1h = data_dir / f"{symbol}_1H.csv"
        ctx_4h = data_dir / f"{symbol}_4H.csv"
        if ctx_1h.exists() and ctx_4h.exists():
            context_files = {"1h": str(ctx_1h), "4h": str(ctx_4h)}
        else:
            # Fallback: run the single-timeframe bot on 15m data
            entry_tf = "1h"

    run_spot_backtest(
        symbol=symbol,
        timeframe=timeframe,
        ohlcv_file=str(ohlcv_path),
        max_candles=25000,
        starting_balance=100.0,
        entry_tf=entry_tf,
        context_files=context_files,
        print_rows=True,
    )


if __name__ == "__main__":
    main()
