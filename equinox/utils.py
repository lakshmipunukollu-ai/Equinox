"""
Single-responsibility: pure helper functions shared across multiple layers.
No business logic. No imports from any other equinox module.
Uses stdlib logging directly — never imports log_trace to avoid circular imports.
"""

import unicodedata
from datetime import datetime, timezone

import dateutil.parser

import logging

_log = logging.getLogger("equinox.utils")


def _to_ascii(text: str) -> str:
    """Normalize string to ASCII, collapse whitespace."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_str = normalized.encode("ascii", errors="ignore").decode("ascii")
    stripped = ascii_str.strip()
    return " ".join(stripped.split())


def parse_utc_datetime(
    value: str | int | float | None, field_name: str
) -> datetime:
    """
    Parses a datetime value into a timezone-aware UTC datetime.
    Handles three input formats:
      - None or empty string → warns and returns datetime.now(UTC)
      - int or float         → treated as Unix timestamp in seconds (Kalshi close_ts)
      - str                  → parsed with dateutil.parser.parse (Polymarket endDateIso)
    Always returns a timezone-aware datetime. Never raises.
    """
    if value is None or value == "":
        _log.warning(
            "Field '%s' is missing — defaulting to current UTC time. "
            "Downstream date comparisons may be inaccurate.",
            field_name,
        )
        return datetime.now(timezone.utc)

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    try:
        result = dateutil.parser.parse(value)
    except (TypeError, ValueError):
        _log.warning(
            "Could not parse datetime value '%s' for field '%s' — "
            "defaulting to current UTC time.",
            value,
            field_name,
        )
        return datetime.now(timezone.utc)

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def parse_utc_datetime_from_fields(
    d: dict | None, field_names: list[str]
) -> datetime:
    """
    Try each key in field_names; parse first present value with parse_utc_datetime.
    If none present, default to current UTC (single warning using first field name).
    Use when the venue API may use different names for the same date (e.g. endDateIso, end_date, endDate, expiration_time).
    """
    if not d or not isinstance(d, dict):
        return parse_utc_datetime(
            None, field_names[0] if field_names else "endDateIso"
        )
    for name in field_names:
        val = d.get(name)
        if val is not None and val != "":
            return parse_utc_datetime(val, name)
    return parse_utc_datetime(
        None, field_names[0] if field_names else "endDateIso"
    )
