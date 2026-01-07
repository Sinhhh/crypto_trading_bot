from __future__ import annotations


def normalize_timeframe(timeframe: str) -> str:
    """Normalize timeframe strings.

    Examples:
    - "15M" -> "15m"
    - "15min" -> "15m"
    - "1H" -> "1h"
    """
    tf = str(timeframe).strip().lower()
    tf = tf.replace("min", "m")
    tf = tf.replace("minutes", "m")
    tf = tf.replace("hour", "h")
    tf = tf.replace("hours", "h")
    return tf


def timeframe_to_suffix(timeframe: str) -> str:
    """Convert a timeframe to the CSV filename suffix used in this repo."""
    tf = normalize_timeframe(timeframe)
    if not tf:
        return "TF"
    # Keep number, uppercase unit
    unit = tf[-1]
    if unit.isalpha():
        return tf[:-1] + unit.upper()
    return tf.upper()
