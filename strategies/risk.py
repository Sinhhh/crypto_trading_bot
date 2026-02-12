"""
This module defines the risk management strategy for the trading bot. It calculates the position size based on the confidence level of a trade, which can be HIGH, MEDIUM, LOW, or NO_TRADE. The position size is determined as a percentage of the account risk per trade, with different multipliers for each confidence level.
"""


def position_size(confidence_level: str, base_risk: float = 1.0) -> float:
    """
    base_risk = % account risk per trade (e.g. 1%)
    """
    if confidence_level == "HIGH":
        return base_risk
    if confidence_level == "MEDIUM":
        return base_risk * 0.5
    if confidence_level == "LOW":
        return base_risk * 0.25
    return 0.0
