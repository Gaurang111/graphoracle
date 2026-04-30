"""Logging utilities — thin wrapper around loguru."""

from __future__ import annotations

import sys
from typing import Any


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with the given module name."""
    try:
        from loguru import logger
        return logger.bind(module=name)
    except ImportError:
        import logging
        return logging.getLogger(name)


def configure_logging(
    level: str = "INFO",
    sink: Any = sys.stderr,
    fmt: str = "{time:HH:mm:ss} | {level:<8} | {extra[module]} — {message}",
) -> None:
    """Configure loguru for the whole process."""
    try:
        from loguru import logger
        logger.remove()
        logger.add(sink, level=level, format=fmt)
    except ImportError:
        import logging
        logging.basicConfig(level=getattr(logging, level, logging.INFO))
