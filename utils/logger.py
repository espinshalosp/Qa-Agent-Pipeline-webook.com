"""
utils/logger.py
===============
Structured logger factory for the QA Agent Platform.

Usage
-----
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("message", key="value")

Output
------
- Console  : human-readable via `rich`
- Log file : JSON lines written to logs/pipeline_{run_id}.log
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    _STRUCTLOG_AVAILABLE = False

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_configured = False
_log_file: Path | None = None


def configure(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    run_id: str | None = None,
) -> None:
    """
    Call once at pipeline startup to configure logging globally.

    Parameters
    ----------
    log_level : str
        One of DEBUG / INFO / WARNING / ERROR.
    log_dir : Path | None
        Directory where log files are written. If None, file logging is skipped.
    run_id : str | None
        Pipeline run ID used to name the log file.
    """
    global _configured, _log_file

    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # --- Console handler ---
    if _RICH_AVAILABLE:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(level)
    handlers.append(console_handler)

    # --- File handler (JSON lines) ---
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_label = run_id or "run"
        _log_file = log_dir / f"pipeline_{run_label}.log"
        file_handler = logging.FileHandler(_log_file, encoding="utf-8")
        file_handler.setLevel(level)
        # Simple format for file — structlog will add JSON if available
        formatter = logging.Formatter(
            fmt='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # --- Root logger ---
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a standard logger for the given module name.
    If `configure()` has not been called yet, configure with defaults.
    """
    global _configured
    if not _configured:
        configure()
    return logging.getLogger(name)
