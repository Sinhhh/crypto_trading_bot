import pandas as pd


class CCXTExchange:
    """Thin ccxt wrapper for paper trading (public endpoints only by default)."""

    def __init__(
        self,
        exchange_id: str,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        try:
            import ccxt  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "ccxt is required. Install with: pip install ccxt"
            ) from e

        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Unknown ccxt exchange id: {exchange_id}")

        exchange_cls = getattr(ccxt, exchange_id)

        params = {
            "apiKey": api_key or "",
            "secret": api_secret or "",
            "enableRateLimit": True,
            "timeout": 30000,
        }

        # Spot-only guardrail: for Binance, explicitly force spot market data.
        # (Prevents accidentally using futures/perps defaults.)
        if exchange_id == "binance":
            params["options"] = {"defaultType": "spot"}

        self.exchange = exchange_cls(params)
        self.exchange_id = exchange_id
        self._markets_loaded = False

    def load_markets(self) -> None:
        if self._markets_loaded:
            return
        self.exchange.load_markets()
        self._markets_loaded = True

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        self.load_markets()
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_last_price(self, symbol: str) -> float | None:
        self.load_markets()
        ticker = self.exchange.fetch_ticker(symbol)
        last = ticker.get("last")
        if last is None:
            return None
        return float(last)
