"""Root-logger configuration used by every module."""
from __future__ import annotations

import logging
import sys


def setup_logger(name: str = "crispr", level: str = "INFO") -> logging.Logger:
    root = logging.getLogger()
    if not any(getattr(h, "_crispr", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        handler._crispr = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper()))
    return logging.getLogger(name)
