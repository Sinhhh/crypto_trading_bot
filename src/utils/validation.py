"""
Validation helpers to enforce framework rules.
"""


def validate_bias(bias: str) -> bool:
    return bias in ["BUY", "SELL", "SIDEWAY"]


def validate_rr(entry: float, stop: float, target: float, min_rr: float = 1.5) -> bool:
    """
    Validate Risk/Reward ratio.
    """
    risk = abs(entry - stop)
    reward = abs(target - entry)

    if risk == 0:
        return False

    return (reward / risk) >= min_rr
