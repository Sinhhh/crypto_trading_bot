from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class SimPosition:
    symbol: str
    side: str  # LONG only
    amount: float
    entry: float
    stop: float
    target: float
    opened_at: str


class BrokerSim:
    """Very small spot-like broker simulator (USDT cash, long-only positions).

    - Market orders fill at the provided price (no slippage/fees).
    - Equity = cash + sum(position_amount * last_price)
    """

    def __init__(self, starting_cash_usdt: float = 10_000.0):
        self.cash_usdt = float(starting_cash_usdt)
        self.positions: dict[str, SimPosition] = {}
        self.last_price: dict[str, float] = {}

    def set_last_price(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = float(price)

    def equity_usdt(self) -> float:
        eq = self.cash_usdt
        for sym, pos in self.positions.items():
            px = self.last_price.get(sym)
            if px is None:
                px = pos.entry
            eq += float(pos.amount) * float(px)
        return float(eq)

    def get_position(self, symbol: str) -> SimPosition | None:
        return self.positions.get(symbol)

    def can_buy_notional(self, notional_usdt: float) -> bool:
        return self.cash_usdt >= float(notional_usdt)

    def open_long(
        self, symbol: str, amount: float, price: float, stop: float, target: float
    ) -> dict:
        if symbol in self.positions:
            return {"ok": False, "reason": "POSITION_EXISTS"}

        amt = float(amount)
        px = float(price)
        if amt <= 0 or px <= 0:
            return {"ok": False, "reason": "BAD_AMOUNT_OR_PRICE"}

        notional = amt * px
        if not self.can_buy_notional(notional):
            return {"ok": False, "reason": "INSUFFICIENT_CASH"}

        self.cash_usdt -= notional
        self.positions[symbol] = SimPosition(
            symbol=symbol,
            side="LONG",
            amount=amt,
            entry=px,
            stop=float(stop),
            target=float(target),
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        return {"ok": True, "filled_price": px, "notional": notional}

    def close_long(self, symbol: str, price: float, reason: str) -> dict:
        pos = self.positions.get(symbol)
        if not pos:
            return {"ok": False, "reason": "NO_POSITION"}

        px = float(price)
        proceeds = float(pos.amount) * px
        self.cash_usdt += proceeds
        pnl = float(pos.amount) * (px - float(pos.entry))

        del self.positions[symbol]
        return {
            "ok": True,
            "filled_price": px,
            "proceeds": proceeds,
            "pnl": pnl,
            "reason": reason,
        }

    def check_stop_target(self, symbol: str) -> dict | None:
        pos = self.positions.get(symbol)
        if not pos:
            return None

        px = self.last_price.get(symbol)
        if px is None:
            return None

        if float(px) <= float(pos.stop):
            return self.close_long(symbol, px, reason="STOP")

        if float(px) >= float(pos.target):
            return self.close_long(symbol, px, reason="TARGET")

        return None
