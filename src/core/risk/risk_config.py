from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfigV2:

    # Base risk
    base_risk_pct: float = 0.01
    min_risk_pct: float = 0.0025
    max_portfolio_risk_pct: float = 0.05

    # Drawdown throttle
    dd_soft_limit: float = 0.05
    dd_hard_limit: float = 0.10

    # Symbol concentration
    max_symbol_risk_pct: float = 0.02

    # Asymmetry
    min_rr: float = 2.0

    # Quality scaling
    enable_quality_scaling: bool = True
