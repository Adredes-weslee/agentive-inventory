r"""backend\app\api\v1\__init__.py

Versioned API routers for the FastAPI application."""

from importlib import import_module
from typing import Any

__all__ = [
    "approvals",
    "backtest",
    "catalog",
    "configs",
    "data",
    "forecasts",
    "health",
    "procure",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
