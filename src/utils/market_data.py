"""
Market data fetching utilities.
Fetches OHLCV candles from crypto exchanges.
"""

import ccxt
import pandas as pd


SUPPORTED_TIMEFRAMES = {"15M": "15m", "1H": "1h", "4H": "4h"}


def get_exchange(name="binance"):
    exchange_class = getattr(ccxt, name)
    exchange = exchange_class({"enableRateLimit": True})
    return exchange


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = 500,
    since: int | None = None,
    exchange_name: str = "binance",
) -> pd.DataFrame:
    """
    Fetch OHLCV data and return DataFrame.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    exchange = get_exchange(exchange_name)
    tf = SUPPORTED_TIMEFRAMES[timeframe]

    market_symbol = f"{symbol}/USDT"
    ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe=tf, since=since, limit=limit)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
