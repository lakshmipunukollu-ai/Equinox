"""
Single-responsibility: configure and expose the Equinox logger.
Two handlers: console shows WARNING and above; file captures everything
from DEBUG up. All modules import get_logger() and log_trace() from here.
Never configure logging in any other module.

Level guide:
  Normal decision trace   → log_trace(..., level="info")    ← file only
  Missing field default   → log_trace(..., level="warning") ← console + file
  API error / retry       → log_trace(..., level="warning") ← console + file
  Auth failure           → log_trace(..., level="warning") ← console + file
  Graceful degradation   → log_trace(..., level="warning") ← console + file
  Fatal unexpected error → log_trace(..., level="error")    ← console + file
"""

import json
import logging
import os


def _setup_logger() -> logging.Logger:
    root = logging.getLogger("equinox")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Handler 1 — StreamHandler (console): WARNING+
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    root.addHandler(console)

    # Handler 2 — FileHandler (LOG_PATH from env, default logs/equinox.log): DEBUG+
    log_path = os.getenv("LOG_PATH", "logs/equinox.log")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(file_handler)

    return root


_setup_logger()


def get_logger(name: str) -> logging.Logger:
    """Returns logging.getLogger(f"equinox.{name}")."""
    return logging.getLogger(f"equinox.{name}")


def log_trace(
    stage: str,
    event: str,
    data: dict,
    level: str = "info",
) -> None:
    """
    Logs a narrative trace at the specified level.
    Format: [TRACE][{stage}] {event} | {json.dumps(data, default=str)}
    """
    logger = logging.getLogger("equinox.trace")
    msg = f"[TRACE][{stage}] {event} | {json.dumps(data, default=str)}"
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    log_level = level_map.get(level, logging.INFO)
    logger.log(log_level, msg)
