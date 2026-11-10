"""One logging setup for every module: timestamp, level, file:line, message.

    from frodo.log import get_logger
    log = get_logger(__name__)
    log.info("docked at %s", station)
"""
from __future__ import annotations

import logging

from .config import get_config

FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(filename)s:%(lineno)d | %(message)s"
_configured = False


def setup_logging(level: str | None = None, file: str | None = None) -> None:
    """Call once from an entry point. Arguments override the [logging] config section."""
    global _configured
    cfg = get_config().logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if file or cfg.file:
        handlers.append(logging.FileHandler(file or cfg.file))
    logging.basicConfig(level=(level or cfg.level).upper(), format=FORMAT, datefmt="%H:%M:%S",
                        handlers=handlers, force=True)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
