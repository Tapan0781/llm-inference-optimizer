"""Structured logging for the project.

Use :func:`get_logger` instead of ``print()`` anywhere inside ``src/``. The
logger writes a consistent, timestamped format to stderr and respects the
``LLMOPT_LOG_LEVEL`` environment variable (default ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    The logger is configured once per name with a single stderr handler and
    the level taken from the ``LLMOPT_LOG_LEVEL`` environment variable.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ready-to-use :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if name not in _configured:
        level_name = os.environ.get("LLMOPT_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

        _configured.add(name)

    return logger
