# SpotTradeState, SpotTradeLifecycle
from dataclasses import dataclass
import pandas as pd

from crypto_trading.indicators.volatility import atr as atr_series


@dataclass
class SpotTradeState:
    in_position: bool = False
    entry_price: float | None = None
    entry_index: int | None = None
    highest_price: float | None = None
    atr_stop: float | None = None  # ATR-based stop
    partial_target: float | None = None  # Take partial profit price
    cooldown_until: int | None = None  # Bar index when next entry allowed
    entry_regime: str | None = None
    qty: float | None = None  # base-asset quantity for spot sizing
    entry_notional_usdt: float | None = None
    realized_pnl_usdt: float = 0.0
    partial_taken: bool = False
    just_entered: bool = False
    just_exited: bool = False


class SpotTradeLifecycle:
    def __init__(
        self,
        *,
        atr_multiplier: float = 3.0,  # ATR * multiplier for stop
        trail_pct: float | None = 0.02,  # trailing stop (% from high)
        trail_atr_mult: float | None = None,  # optional ATR trailing (Chandelier)
        partial_profit_pct: float = 0.03,  # take 1st portion at +3%
        partial_sell_fraction: float = 0.5,  # fraction of qty to sell at partial
        max_bars_in_trade: int = 48,
        cooldown_bars: int = 3,
        exit_on_regime_change: bool = True,
        exit_on_sell_signal: bool = True,
        volatility_sizing: bool = True,
    ):
        self.atr_multiplier = float(atr_multiplier)
        self.trail_pct = None if trail_pct is None else float(trail_pct)
        self.trail_atr_mult = None if trail_atr_mult is None else float(trail_atr_mult)
        self.partial_profit_pct = float(partial_profit_pct)
        self.partial_sell_fraction = float(partial_sell_fraction)
        self.max_bars = int(max_bars_in_trade)
        self.cooldown_bars = int(cooldown_bars)
        self.exit_on_regime_change = exit_on_regime_change
        self.exit_on_sell_signal = exit_on_sell_signal
        self.volatility_sizing = volatility_sizing

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR of last 'period' bars using project indicator."""
        col = f"atr_{int(period)}"
        if df is None or len(df) < 1:
            return 0.0

        if col in df.columns:
            v = df[col].iloc[-1]
            if not pd.isna(v):
                return float(v)

        if len(df) < int(period) + 2:
            return 0.0
        if not {"high", "low", "close"}.issubset(df.columns):
            return 0.0

        a = atr_series(df, int(period)).iloc[-1]
        if pd.isna(a):
            return 0.0
        return float(a)

    def _should_exit_on_regime_change(
        self, entry_regime: str | None, regime: str
    ) -> bool:
        """Return True if current regime invalidates holding a long position."""
        cur = str(regime)
        ent = None if entry_regime is None else str(entry_regime)

        # Long-only protection: exit immediately if the market turns bearish.
        if cur in {"TREND_DOWN", "CROSS_DOWN"}:
            return True

        if ent is None:
            return False

        # If we entered on a bullish regime, allow slight weakening but keep direction.
        if ent == "TREND_UP":
            return cur not in {"TREND_UP", "CROSS_UP"}

        if ent == "CROSS_UP":
            return cur not in {"CROSS_UP", "TREND_UP"}

        # Range trades only make sense in range.
        if ent == "RANGE":
            return cur != "RANGE"

        # Default: do not force-exit on regime change.
        return False

    def update(
        self,
        *,
        df_1h: pd.DataFrame,
        state: SpotTradeState,
        signal: str,
        regime: str,
        price: float,
        bar_index: int,
        atr_stop_override: float | None = None,
    ) -> tuple[SpotTradeState, dict | None]:
        """Spot lifecycle with ATR stop, trailing stop, partial profit, and cooldown.

        Returns: (updated_state, trade_event)
        trade_event is None unless an ENTRY/EXIT occurs.
        """
        trade_event: dict | None = None

        # Reset one-bar flags unless an event occurs this bar.
        state.just_entered = False
        state.just_exited = False
        # -----------------------
        # Cooldown check
        # -----------------------
        if state.cooldown_until is not None and bar_index < state.cooldown_until:
            return state, None  # Cannot re-enter yet

        # -----------------------
        # FLAT → ENTRY
        # -----------------------
        if not state.in_position and signal == "BUY":
            # ATR-based stop
            if atr_stop_override is not None:
                atr_stop = float(atr_stop_override)
            else:
                a = self.compute_atr(df_1h)
                atr_stop = price - self.atr_multiplier * a if a > 0 else price * 0.97
            partial_target = price * (1.0 + self.partial_profit_pct)

            if df_1h is not None and len(df_1h) > 0 and "high" in df_1h.columns:
                highest_price = float(df_1h["high"].iloc[-1])
            else:
                highest_price = float(price)

            new_state = SpotTradeState(
                in_position=True,
                entry_price=price,
                entry_index=bar_index,
                highest_price=highest_price,
                atr_stop=atr_stop,
                partial_target=partial_target,
                cooldown_until=None,
                entry_regime=regime,
                realized_pnl_usdt=0.0,
                partial_taken=False,
                just_entered=True,
                just_exited=False,
            )

            trade_event = {
                "type": "ENTRY",
                "price": float(price),
                "bar_index": int(bar_index),
                "entry_index": int(bar_index),
                "entry_price": float(price),
                "atr_stop": float(atr_stop),
                "regime": str(regime),
                "partial_taken": False,
                "pnl_usdt": 0.0,
            }
            return new_state, trade_event

        # -----------------------
        # IN POSITION → EXIT / MANAGEMENT
        # -----------------------
        if state.in_position:
            # Candle extremes (close-only can miss triggers).
            if df_1h is not None and len(df_1h) > 0 and "high" in df_1h.columns:
                last_high = float(df_1h["high"].iloc[-1])
            else:
                last_high = float(price)

            if df_1h is not None and len(df_1h) > 0 and "low" in df_1h.columns:
                last_low = float(df_1h["low"].iloc[-1])
            else:
                last_low = float(price)

            highest = max(state.highest_price or last_high, last_high)

            # 1️⃣ ATR STOP
            if state.atr_stop is not None and last_low <= state.atr_stop:
                exit_price = float(state.atr_stop)
                entry_price = state.entry_price
                qty = state.qty
                pnl_exit_usdt = 0.0
                if entry_price is not None and qty is not None:
                    pnl_exit_usdt = float(qty) * (exit_price - float(entry_price))

                pnl_total_usdt = float(state.realized_pnl_usdt) + float(pnl_exit_usdt)
                trade_event = {
                    "type": "EXIT",
                    "price": exit_price,
                    "bar_index": int(bar_index),
                    "entry_index": state.entry_index,
                    "entry_price": entry_price,
                    "regime": state.entry_regime,
                    "partial_taken": bool(state.partial_taken),
                    "qty": qty,
                    "pnl_usdt": float(pnl_exit_usdt),
                    "pnl_total_usdt": float(pnl_total_usdt),
                }

                # Trigger cooldown after stop-loss
                return (
                    SpotTradeState(
                        cooldown_until=int(bar_index) + int(self.cooldown_bars),
                        just_exited=True,
                    ),
                    trade_event,
                )

            # 2️⃣ Trailing stop
            trail_price: float | None = None
            if self.trail_pct is not None:
                trail_price = float(highest) * (1.0 - float(self.trail_pct))
            if self.trail_atr_mult is not None:
                a = self.compute_atr(df_1h)
                if a > 0:
                    trail_atr_price = float(highest) - float(
                        self.trail_atr_mult
                    ) * float(a)
                    trail_price = (
                        trail_atr_price
                        if trail_price is None
                        else max(float(trail_price), float(trail_atr_price))
                    )

            if trail_price is not None and last_low <= float(trail_price):
                exit_price = float(trail_price)
                entry_price = state.entry_price
                qty = state.qty
                pnl_exit_usdt = 0.0
                if entry_price is not None and qty is not None:
                    pnl_exit_usdt = float(qty) * (exit_price - float(entry_price))

                pnl_total_usdt = float(state.realized_pnl_usdt) + float(pnl_exit_usdt)
                trade_event = {
                    "type": "EXIT",
                    "price": exit_price,
                    "bar_index": int(bar_index),
                    "entry_index": state.entry_index,
                    "entry_price": entry_price,
                    "regime": state.entry_regime,
                    "partial_taken": bool(state.partial_taken),
                    "qty": qty,
                    "pnl_usdt": float(pnl_exit_usdt),
                    "pnl_total_usdt": float(pnl_total_usdt),
                }
                return (
                    SpotTradeState(
                        cooldown_until=int(bar_index) + int(self.cooldown_bars),
                        just_exited=True,
                    ),
                    trade_event,
                )

            # 3️⃣ Partial profit (real execution: sell fraction, realize pnl, reduce qty)
            if (
                state.partial_target is not None
                and last_high >= float(state.partial_target)
                and not bool(state.partial_taken)
            ):
                exec_price = float(state.partial_target)
                qty = state.qty
                qty_sold = None
                pnl_partial_usdt = 0.0

                if qty is not None and qty > 0 and state.entry_price is not None:
                    frac = max(0.0, min(1.0, float(self.partial_sell_fraction)))
                    qty_sold = float(qty) * float(frac)
                    qty_sold = min(float(qty_sold), float(qty))
                    pnl_partial_usdt = float(qty_sold) * (
                        exec_price - float(state.entry_price)
                    )
                    state.qty = float(qty) - float(qty_sold)
                    state.realized_pnl_usdt = float(state.realized_pnl_usdt) + float(
                        pnl_partial_usdt
                    )

                # Mark partial taken and bump stop to at least break-even.
                if state.entry_price is not None:
                    state.atr_stop = max(
                        float(state.atr_stop or 0.0), float(state.entry_price)
                    )
                state.partial_taken = True

                trade_event = {
                    "type": "PARTIAL",
                    "price": exec_price,
                    "bar_index": int(bar_index),
                    "entry_index": state.entry_index,
                    "entry_price": state.entry_price,
                    "regime": state.entry_regime,
                    "partial_taken": True,
                    "qty_sold": qty_sold,
                    "pnl_usdt": float(pnl_partial_usdt),
                }
                state.highest_price = highest
                return state, trade_event

            # 4️⃣ Time-based exit
            if state.entry_index is not None:
                if bar_index - state.entry_index >= self.max_bars:
                    exit_price = float(price)
                    entry_price = state.entry_price
                    qty = state.qty
                    pnl_exit_usdt = 0.0
                    if entry_price is not None and qty is not None:
                        pnl_exit_usdt = float(qty) * (exit_price - float(entry_price))

                    pnl_total_usdt = float(state.realized_pnl_usdt) + float(
                        pnl_exit_usdt
                    )
                    trade_event = {
                        "type": "EXIT",
                        "price": exit_price,
                        "bar_index": int(bar_index),
                        "entry_index": state.entry_index,
                        "entry_price": entry_price,
                        "regime": state.entry_regime,
                        "partial_taken": bool(state.partial_taken),
                        "qty": qty,
                        "pnl_usdt": float(pnl_exit_usdt),
                        "pnl_total_usdt": float(pnl_total_usdt),
                    }
                    return (
                        SpotTradeState(
                            cooldown_until=int(bar_index) + int(self.cooldown_bars),
                            just_exited=True,
                        ),
                        trade_event,
                    )

            # 5️⃣ Signal exit
            if self.exit_on_sell_signal and signal == "SELL":
                exit_price = float(price)
                entry_price = state.entry_price
                qty = state.qty
                pnl_exit_usdt = 0.0
                if entry_price is not None and qty is not None:
                    pnl_exit_usdt = float(qty) * (exit_price - float(entry_price))

                pnl_total_usdt = float(state.realized_pnl_usdt) + float(pnl_exit_usdt)
                trade_event = {
                    "type": "EXIT",
                    "price": exit_price,
                    "bar_index": int(bar_index),
                    "entry_index": state.entry_index,
                    "entry_price": entry_price,
                    "regime": state.entry_regime,
                    "partial_taken": bool(state.partial_taken),
                    "qty": qty,
                    "pnl_usdt": float(pnl_exit_usdt),
                    "pnl_total_usdt": float(pnl_total_usdt),
                }
                return (
                    SpotTradeState(
                        cooldown_until=int(bar_index) + int(self.cooldown_bars),
                        just_exited=True,
                    ),
                    trade_event,
                )

            # 6️⃣ Regime exit
            if self.exit_on_regime_change:
                if self._should_exit_on_regime_change(state.entry_regime, regime):
                    exit_price = float(price)
                    entry_price = state.entry_price
                    qty = state.qty
                    pnl_exit_usdt = 0.0
                    if entry_price is not None and qty is not None:
                        pnl_exit_usdt = float(qty) * (exit_price - float(entry_price))

                    pnl_total_usdt = float(state.realized_pnl_usdt) + float(
                        pnl_exit_usdt
                    )
                    trade_event = {
                        "type": "EXIT",
                        "price": exit_price,
                        "bar_index": int(bar_index),
                        "entry_index": state.entry_index,
                        "entry_price": entry_price,
                        "regime": state.entry_regime,
                        "partial_taken": bool(state.partial_taken),
                        "qty": qty,
                        "pnl_usdt": float(pnl_exit_usdt),
                        "pnl_total_usdt": float(pnl_total_usdt),
                    }
                    return (
                        SpotTradeState(
                            cooldown_until=int(bar_index) + int(self.cooldown_bars),
                            just_exited=True,
                        ),
                        trade_event,
                    )

            # Update highest price
            state.highest_price = highest
            return state, None

        return state, None  # No state change
