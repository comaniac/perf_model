"""
The format and config of logging.
"""

import logging
import time
from typing import Callable, Dict

LOGGER_TABLE: Dict[str, logging.Logger] = {}

FORMATTER = logging.Formatter('[%(asctime)s] %(levelname)7s %(name)s: %(message)s',
                              '%Y-%m-%d %H:%M:%S')
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setFormatter(FORMATTER)


def get_logger(name: str) -> logging.Logger:
    """Attach to the default logger."""

    if name in LOGGER_TABLE:
        return LOGGER_TABLE[name]

    logger = logging.getLogger(name)
    logger.parent = None
    logger.setLevel(logging.INFO)
    logger.addHandler(STREAM_HANDLER)

    LOGGER_TABLE[name] = logger
    return logger

def enable_log_file(file_name: str):
    """Add file handler to all loggers."""

    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(FORMATTER)

    for logger in LOGGER_TABLE.values():
        logger.addHandler(file_handler)

