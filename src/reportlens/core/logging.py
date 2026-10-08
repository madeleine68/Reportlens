"""Logging setup shared by all modules."""

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once; safe to call repeatedly."""
    logging.basicConfig(level=level.upper(), format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, e.g. get_logger(__name__)."""
    return logging.getLogger(name)
