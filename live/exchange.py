from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketInfo:
    amount_precision: int | None
    min_amount: float | None


class LiveExchange:
    """ccxt exchange wrapper for live SPOT trading (Binance only).

    Provides:
    - OHLCV fetch (4h/1h/15m)
    - last price
    - balances
    - market buy/sell

    Spot-only guardrails are enabled.
    """

    def __init__(self, exchange_id: str, api_key: str | None, api_secret: str | None):
        try:
            import ccxt  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "ccxt is required. Install with: pip install ccxt"
            ) from e

        exchange_id = str(exchange_id).strip().lower()
        if exchange_id != "binance":
            raise ValueError(
                "Live trader is SPOT-only and currently supports only: binance"
            )

        exchange_cls = getattr(ccxt, exchange_id)
        params = {
            "apiKey": api_key or "",
            "secret": api_secret or "",
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"defaultType": "spot"},
        }

        self.exchange_id = exchange_id
        self.exchange = exchange_cls(params)
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
        if precision.get("amount") is not None:
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
        return float(last) if last is not None else None

    def fetch_free_balance(self, asset: str) -> float:
        self.load_markets()
        bal = self.exchange.fetch_balance() or {}
        free = (bal.get("free") or {}).get(asset)
        return float(free) if free is not None else 0.0

    def create_market_buy(self, symbol: str, amount: float) -> dict:
        self.load_markets()
        amount_r = self.round_amount(symbol, amount)
        return self.exchange.create_order(symbol, "market", "buy", amount_r)

    def create_market_sell(self, symbol: str, amount: float) -> dict:
        self.load_markets()
        amount_r = self.round_amount(symbol, amount)
        return self.exchange.create_order(symbol, "market", "sell", amount_r)

    @staticmethod
    def base_quote(symbol: str) -> tuple[str, str]:
        base, quote = symbol.split("/")
        return base.upper(), quote.upper()
