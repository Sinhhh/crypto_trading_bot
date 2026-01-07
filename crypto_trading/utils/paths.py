from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """Return repository root by searching for common markers.

    Markers: `.git/`, `requirements.txt`.
    """
    here = Path(start) if start is not None else Path.cwd()
    here = here.resolve()

    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "requirements.txt").exists():
            return p

    # Fallback: current working directory
    return here


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(*parts: str, root: str | Path | None = None) -> Path:
    base = repo_root(root) / "data"
    return base.joinpath(*parts)


def reports_dir(*parts: str, root: str | Path | None = None) -> Path:
    base = repo_root(root) / "reports"
    return base.joinpath(*parts)
