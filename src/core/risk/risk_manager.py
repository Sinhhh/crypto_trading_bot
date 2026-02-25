class RiskManagerV2:

    def __init__(self, broker, config):
        self.broker = broker
        self.cfg = config
        self.peak_equity = broker.equity_usdt()

    # =============================
    # Capital State
    # =============================

    def _update_peak(self):
        eq = self.broker.equity_usdt()
        if eq > self.peak_equity:
            self.peak_equity = eq

    def _drawdown_pct(self):
        self._update_peak()
        eq = self.broker.equity_usdt()

        if self.peak_equity == 0:
            return 0.0

        return (self.peak_equity - eq) / self.peak_equity

    def _dd_scaled_risk_pct(self):
        dd = self._drawdown_pct()

        if dd <= self.cfg.dd_soft_limit:
            return self.cfg.base_risk_pct

        if dd >= self.cfg.dd_hard_limit:
            return self.cfg.min_risk_pct

        # linear interpolation
        scale = 1 - (dd - self.cfg.dd_soft_limit) / (
            self.cfg.dd_hard_limit - self.cfg.dd_soft_limit
        )

        return self.cfg.min_risk_pct + scale * (
            self.cfg.base_risk_pct - self.cfg.min_risk_pct
        )

    # =============================
    # Portfolio Risk
    # =============================

    def _total_open_risk(self):
        total = 0.0
        for pos in self.broker.positions.values():
            stop_distance = abs(pos.entry - pos.stop)
            total += pos.amount * stop_distance
        return total

    def _symbol_open_risk(self, symbol: str):
        total = 0.0
        for pos in self.broker.positions.values():
            if pos.symbol == symbol:
                stop_distance = abs(pos.entry - pos.stop)
                total += pos.amount * stop_distance
        return total

    # =============================
    # Main Sizing Logic
    # =============================

    def compute_size(
        self,
        symbol: str,
        side: str,  # "LONG" or "SHORT"
        entry: float,
        stop: float,
        target: float,
        setup_score: float = 1.0,
    ) -> float:

        equity = self.broker.equity_usdt()

        # ----- 1. Asymmetry Filter -----
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0

        rr = abs(target - entry) / stop_distance
        if rr < self.cfg.min_rr:
            return 0.0

        # ----- 2. Dynamic Risk Budget -----
        risk_pct = self._dd_scaled_risk_pct()

        if self.cfg.enable_quality_scaling:
            setup_score = max(0.0, min(1.0, setup_score))
            risk_pct *= 0.5 + 0.5 * setup_score

        risk_usdt = equity * risk_pct

        # ----- 3. Convert Risk → Size -----
        amount = risk_usdt / stop_distance

        # ----- 4. Portfolio Caps -----
        if (
            self._total_open_risk() + risk_usdt
            > equity * self.cfg.max_portfolio_risk_pct
        ):
            return 0.0

        if (
            self._symbol_open_risk(symbol) + risk_usdt
            > equity * self.cfg.max_symbol_risk_pct
        ):
            return 0.0

        return amount
