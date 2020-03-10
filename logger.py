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

FILE_HANDLER = logging.FileHandler('run-{}.log'.format(int(time.time())))
FILE_HANDLER.setFormatter(FORMATTER)


def get_logger(name: str) -> logging.Logger:
    """Attach to the default logger."""

    if name in LOGGER_TABLE:
        return LOGGER_TABLE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(STREAM_HANDLER)
    logger.addHandler(FILE_HANDLER)

    LOGGER_TABLE[name] = logger
    return logger

