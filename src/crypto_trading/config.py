"""Configuration helpers.

Do NOT hardcode exchange keys in source control.

Set env vars instead:
- MEXC_API_KEY
- MEXC_SECRET_KEY
"""

from __future__ import annotations

import os


def get_api_key() -> str:
    return str(os.getenv("MEXC_API_KEY", ""))


def get_secret_key() -> str:
    return str(os.getenv("MEXC_SECRET_KEY", ""))

