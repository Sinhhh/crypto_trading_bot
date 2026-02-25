import os
import sys


def _ensure_repo_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def test_full_backtest_btc():
    _ensure_repo_on_path()

    import backtest

    df = backtest.backtest_symbol("BTC", log_skips=False)
    assert df is not None


def test_full_backtest_eth():
    _ensure_repo_on_path()

    import backtest

    df = backtest.backtest_symbol("ETH", log_skips=False)
    assert df is not None
