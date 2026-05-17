"""
Centralized logging configuration for the RAG application.
"""
import logging
import sys
from typing import Optional


_loggers: dict = {}


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a named logger with consistent formatting.

    Args:
        name: Logger name (usually __name__ of the calling module).
        level: Optional log level override (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    log_level = level or "INFO"
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    _loggers[name] = logger
    return logger
