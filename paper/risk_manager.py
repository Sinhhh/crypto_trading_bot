from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    # Risk sizing: use stop distance and equity
    risk_pct: float = 0.01          # Max % of equity risked per trade
    max_notional_pct: float = 0.95  # Cap exposure to 95% of equity

    # Minimum notional guard (avoid dust trades)
    min_notional_usdt: float = 10.0 # Avoid tiny “dust” trades
    fee_pct: float = 0.0005
    slippage_pct: float = 0.001     # Slippage (0.1%)


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def size_for_long(self, equity_usdt: float, entry: float, stop: float) -> float:
        """
        Compute long position size with fees and slippage.
        """
        return self._compute_size(
            equity_usdt, entry, stop, direction="long"
        )

    def size_for_short(self, equity_usdt: float, entry: float, stop: float) -> float:
        """
        Compute short position size with fees and slippage.
        """
        return self._compute_size(
            equity_usdt, entry, stop, direction="short"
        )

    def _compute_size(self, equity_usdt: float, entry: float, stop: float, direction: str) -> float:
        equity = float(equity_usdt)
        entry_f = float(entry)
        stop_f = float(stop)

        if equity <= 0 or entry_f <= 0 or stop_f <= 0:
            return 0.0

        # Effective fill prices with slippage
        if direction == "long":
            entry_eff = entry_f * (1 + self.cfg.slippage_pct)
            stop_eff = stop_f * (1 - self.cfg.slippage_pct)
            price_risk_per_unit = entry_eff - stop_eff
        elif direction == "short":
            entry_eff = entry_f * (1 - self.cfg.slippage_pct)
            stop_eff = stop_f * (1 + self.cfg.slippage_pct)
            price_risk_per_unit = stop_eff - entry_eff
        else:
            raise ValueError("Direction must be 'long' or 'short'")

        if price_risk_per_unit <= 0:
            return 0.0

        # Compute risk in USDT
        risk_usdt = equity * self.cfg.risk_pct

        # Fees per unit for entry and exit at stop
        fees_per_unit = entry_eff * self.cfg.fee_pct + stop_eff * self.cfg.fee_pct
        total_risk_per_unit = price_risk_per_unit + fees_per_unit

        # Base size
        amount = risk_usdt / total_risk_per_unit

        # Cap by max_notional_pct
        max_amount_by_notional = (equity * self.cfg.max_notional_pct) / entry_eff
        amount = min(amount, max_amount_by_notional)

        # Enforce min_notional
        if amount * entry_eff < self.cfg.min_notional_usdt:
            return 0.0

        return float(amount)
