from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    # Risk sizing: use stop distance and equity
    risk_pct: float = 0.01
    max_notional_pct: float = 0.95

    # Minimum notional guard (avoid dust trades)
    min_notional_usdt: float = 10.0


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def size_for_long(self, equity_usdt: float, entry: float, stop: float) -> float:
        """Compute base-asset amount for a long using risk% and stop distance.

        amount = (equity * risk_pct) / |entry - stop|
        Then capped by max_notional_pct.
        """
        equity = float(equity_usdt)
        if equity <= 0:
            return 0.0

        entry_f = float(entry)
        stop_f = float(stop)
        risk_per_unit = abs(entry_f - stop_f)
        if entry_f <= 0 or risk_per_unit <= 0:
            return 0.0

        risk_usdt = equity * float(self.cfg.risk_pct)
        amount = risk_usdt / risk_per_unit

        # Cap by notional
        max_amount_by_notional = (equity * float(self.cfg.max_notional_pct)) / entry_f
        amount = min(amount, max_amount_by_notional)

        # Notional guard
        if amount * entry_f < float(self.cfg.min_notional_usdt):
            return 0.0

        return float(amount)
