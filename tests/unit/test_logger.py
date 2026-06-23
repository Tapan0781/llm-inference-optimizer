"""CPU-safe unit tests for src.utils.logger."""

from __future__ import annotations

import logging

from src.utils.logger import get_logger


def test_get_logger_returns_logger() -> None:
    logger = get_logger("test.logger.a")
    assert isinstance(logger, logging.Logger)


def test_logger_has_single_handler_after_repeat_calls() -> None:
    name = "test.logger.b"
    first = get_logger(name)
    second = get_logger(name)
    assert first is second
    assert len(first.handlers) == 1


def test_logger_does_not_propagate() -> None:
    logger = get_logger("test.logger.c")
    assert logger.propagate is False
