from __future__ import annotations


def main() -> None:
    # Delegate to the existing paper trading CLI.
    from crypto_trading.simulator.paper_trading import main as _main

    _main()


if __name__ == "__main__":
    main()
