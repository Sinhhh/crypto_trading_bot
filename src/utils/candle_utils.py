"""
Candle utilities for 15M entry confirmation.
Pure price action – no indicators.
"""


def is_bullish_engulfing(prev, current):
    return (
        current["close"] > current["open"]
        and prev["close"] < prev["open"]
        and current["open"] < prev["close"]
        and current["close"] > prev["open"]
    )


def is_bearish_engulfing(prev, current):
    return (
        current["close"] < current["open"]
        and prev["close"] > prev["open"]
        and current["open"] > prev["close"]
        and current["close"] < prev["open"]
    )


def is_hammer(candle):
    body = abs(candle["close"] - candle["open"])
    lower_wick = (
        candle["open"] - candle["low"]
        if candle["close"] > candle["open"]
        else candle["close"] - candle["low"]
    )
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return lower_wick >= 2 * body and upper_wick <= body


def is_bearish_pinbar(candle):
    body = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return upper_wick >= 2 * body and lower_wick <= body


def is_inside_bar(prev, current):
    return current["high"] <= prev["high"] and current["low"] >= prev["low"]
