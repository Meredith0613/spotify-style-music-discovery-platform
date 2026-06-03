"""Logging helpers for local scripts, apps, and pipelines."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the provided module name."""

    logger = logging.getLogger(name)

    # This guard prevents duplicate handlers when modules are imported
    # multiple times during local development or tests.
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
