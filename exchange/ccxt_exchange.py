from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketInfo:
    amount_precision: int | None
    min_amount: float | None


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

    def market_info(self, symbol: str) -> MarketInfo:
        self.load_markets()
        m = self.exchange.markets.get(symbol) or {}
        prec = None
        min_amt = None
        precision = m.get("precision") or {}
        if "amount" in precision and precision["amount"] is not None:
            prec = int(precision["amount"])
        limits = m.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        if amount_limits.get("min") is not None:
            min_amt = float(amount_limits["min"])
        return MarketInfo(amount_precision=prec, min_amount=min_amt)

    def round_amount(self, symbol: str, amount: float) -> float:
        info = self.market_info(symbol)
        if info.amount_precision is None:
            return float(amount)
        fmt = "{:.%df}" % info.amount_precision
        return float(fmt.format(amount))

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
