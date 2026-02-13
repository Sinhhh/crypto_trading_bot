from dataclasses import dataclass


@dataclass
class SimPosition:
    symbol: str
    side: str  # "LONG" or "SHORT"
    amount: float
    entry: float
    stop: float
    target: float
    margin: float = 0.0


class BrokerSim:
    """Spot-like broker simulator (USDT cash, supports LONG and SHORT positions).

    - Market orders fill at price ± slippage.
    - Fee deducted per trade.
    - Equity = cash + position value (long) or margin + PnL (short)
    """

    def __init__(
        self,
        starting_cash_usdt: float = 10_000.0,
        fee_pct: float = 0.001,  # 0.1%
        slippage_pct: float = 0.0005,  # 0.05%
        short_margin_pct: float = 0.1,  # 10% margin lock
    ):
        self.cash_usdt = float(starting_cash_usdt)
        self.positions: dict[str, SimPosition] = {}
        self.last_price: dict[str, float] = {}
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.short_margin_pct = float(short_margin_pct)

    def set_last_price(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = float(price)

    def equity_usdt(self) -> float:
        eq = self.cash_usdt
        for sym, pos in self.positions.items():
            px = self.last_price.get(sym, pos.entry)
            if pos.side == "LONG":
                eq += pos.amount * px
            elif pos.side == "SHORT":
                eq += pos.margin + pos.amount * (pos.entry - px)
        return float(eq)

    def get_position(self, symbol: str) -> SimPosition | None:
        return self.positions.get(symbol)

    def can_buy_notional(self, notional_usdt: float) -> bool:
        return self.cash_usdt >= notional_usdt

    # ---------------- Long ----------------
    def open_long(
        self, symbol: str, amount: float, price: float, stop: float, target: float
    ) -> dict:
        if symbol in self.positions:
            return {"ok": False, "reason": "POSITION_EXISTS"}

        amt = float(amount)
        px = float(price)
        if amt <= 0 or px <= 0:
            return {"ok": False, "reason": "BAD_AMOUNT_OR_PRICE"}

        # Apply slippage and fee
        px_effective = px * (1 + self.slippage_pct)
        notional = amt * px_effective
        fee = notional * self.fee_pct
        total_cost = notional + fee

        if not self.can_buy_notional(total_cost):
            return {"ok": False, "reason": "INSUFFICIENT_CASH"}

        self.cash_usdt -= total_cost
        self.positions[symbol] = SimPosition(
            symbol=symbol,
            side="LONG",
            amount=amt,
            entry=px_effective,
            stop=float(stop),
            target=float(target),
        )
        return {
            "ok": True,
            "filled_price": px_effective,
            "notional": notional,
            "fee": fee,
        }

    def close_long(self, symbol: str, price: float, reason: str) -> dict:
        pos = self.positions.get(symbol)
        if not pos or pos.side != "LONG":
            return {"ok": False, "reason": "NO_POSITION"}

        # Apply slippage and fee
        px_effective = price * (1 - self.slippage_pct)
        proceeds = pos.amount * px_effective
        fee = proceeds * self.fee_pct
        pnl = proceeds - fee - pos.amount * pos.entry

        self.cash_usdt += proceeds - fee
        del self.positions[symbol]

        return {
            "ok": True,
            "filled_price": px_effective,
            "proceeds": proceeds,
            "pnl": pnl,
            "reason": reason,
            "fee": fee,
        }

    # ---------------- Short ----------------
    def open_short(
        self, symbol: str, amount: float, price: float, stop: float, target: float
    ) -> dict:
        if symbol in self.positions:
            return {"ok": False, "reason": "POSITION_EXISTS"}

        amt = float(amount)
        px = float(price)
        if amt <= 0 or px <= 0:
            return {"ok": False, "reason": "BAD_AMOUNT_OR_PRICE"}

        # Slippage + fee
        px_effective = px * (1 - self.slippage_pct)
        notional = amt * px_effective
        fee = notional * self.fee_pct
        margin = notional * self.short_margin_pct
        if not self.can_buy_notional(fee + margin):
            return {"ok": False, "reason": "INSUFFICIENT_CASH"}

        self.cash_usdt -= fee + margin
        self.positions[symbol] = SimPosition(
            symbol=symbol,
            side="SHORT",
            amount=amt,
            entry=px_effective,
            stop=float(stop),
            target=float(target),
            margin=float(margin),
        )
        return {
            "ok": True,
            "filled_price": px_effective,
            "notional": notional,
            "fee": fee,
        }

    def close_short(self, symbol: str, price: float, reason: str) -> dict:
        pos = self.positions.get(symbol)
        if not pos or pos.side != "SHORT":
            return {"ok": False, "reason": "NO_POSITION"}

        px_effective = price * (1 + self.slippage_pct)
        pnl = pos.amount * (pos.entry - px_effective)
        fee = (pos.amount * px_effective) * self.fee_pct
        pnl -= fee

        self.cash_usdt += pos.margin + pnl
        del self.positions[symbol]

        return {
            "ok": True,
            "filled_price": px_effective,
            "proceeds": pos.margin + pnl,
            "pnl": pnl,
            "reason": reason,
            "fee": fee,
        }

    # ---------------- Stop / Target Check ----------------
    def check_stop_target(self, symbol: str) -> dict | None:
        pos = self.positions.get(symbol)
        if not pos:
            return None

        px = self.last_price.get(symbol)
        if px is None:
            return None

        if pos.side == "LONG":
            if px <= pos.stop:
                return self.close_long(symbol, px, reason="STOP")
            if px >= pos.target:
                return self.close_long(symbol, px, reason="TARGET")
        elif pos.side == "SHORT":
            if px >= pos.stop:
                return self.close_short(symbol, px, reason="STOP")
            if px <= pos.target:
                return self.close_short(symbol, px, reason="TARGET")

        return None
