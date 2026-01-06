"""
Risk Manager for Spot Trading

Calculates position size based on account balance, risk per trade, and stop-loss distance.
"""


def calculate_position_size(
    balance_usdt: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float
) -> float:
    """
    Calculate the amount of asset to buy for a given risk.

    Parameters:
    - balance_usdt: total USDT available
    - risk_per_trade: fraction of balance to risk (e.g., 0.01 for 1%)
    - entry_price: entry price of the asset
    - stop_loss: stop-loss price

    Returns:
    - position_size: quantity of asset to buy
    """
    # Risk in USDT
    risk_amount = balance_usdt * risk_per_trade

    # Distance between entry and stop loss
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        return 0.0

    # Quantity = risk / stop distance
    position_size = risk_amount / stop_distance
    return position_size
