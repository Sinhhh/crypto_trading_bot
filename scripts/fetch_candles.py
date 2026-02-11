"""Compatibility wrapper.

Keeps older invocation patterns working.
Implementation lives in `src.scripts.fetch_candles`.
"""

from src.scripts.fetch_candles import main


if __name__ == "__main__":
    main()
