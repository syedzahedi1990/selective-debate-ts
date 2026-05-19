"""Tiny logging wrapper so callers don't reconfigure root logger themselves."""
from __future__ import annotations

import logging
import os


_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        level = os.environ.get("SDFTS_LOGLEVEL", "INFO").upper()
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )
        _configured = True
    return logging.getLogger(name)
