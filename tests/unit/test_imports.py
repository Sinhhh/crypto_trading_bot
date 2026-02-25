import os
import sys


def test_import_strategy_modules():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from core.strategies.smc_signal import generate_signal  # noqa: F401

    assert generate_signal is not None
