"""Unit tests for equivalence matcher."""

from datetime import datetime, timezone, timedelta

import pytest

from equinox.matcher.engine import find_matches
from equinox.models import Market


def _market(venue: str, mid: str, title: str, close_time: datetime) -> Market:
    return Market(
        id=f"{venue}:{mid}",
        venue=venue,
        title=title,
        category="test",
        yes_price=0.5,
        no_price=0.5,
        yes_bid=0.48,
        yes_ask=0.52,
        spread_width=0.04,
        volume=1000,
        liquidity=500,
        close_time=close_time,
        price_updated_at=None,
        price_source="mid",
        fee_rate=0.07 if venue == "kalshi" else 0.02,
        fee_model="additive" if venue == "kalshi" else "multiplicative",
        is_binary=True,
        url="https://example.com",
        raw={},
    )


def test_exact_match():
    """Identical normalized titles + 0 day delta → exact match."""
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)
    a = _market("kalshi", "K1", "Bitcoin above 80000", base)
    b = _market("polymarket", "P1", "Bitcoin above 80000", base)
    results = find_matches([a], [b])
    assert len(results) == 1
    assert results[0].method == "exact"
    assert results[0].score == 1.0


def test_fuzzy_match():
    """Similar titles + 2 day delta → fuzzy match."""
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)
    other = base + timedelta(days=2)
    a = _market("kalshi", "K1", "Bitcoin price above 80000 dollars", base)
    b = _market("polymarket", "P1", "Bitcoin above 80000 dollars price", other)
    results = find_matches([a], [b], fuzzy_threshold=0.75, date_window_days=3)
    assert len(results) == 1
    assert results[0].method == "fuzzy"
    assert results[0].score >= 0.75


def test_abbreviation_expansion():
    """BTC vs Bitcoin — abbreviation expansion enables match."""
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)
    # After expansion: "bitcoin above 80000" vs "bitcoin above 80000" — exact match
    a = _market("kalshi", "K1", "BTC above 80k", base)
    b = _market("polymarket", "P1", "Bitcoin above 80000", base)
    results = find_matches([a], [b], fuzzy_threshold=0.75, date_window_days=3)
    assert len(results) >= 1


def test_no_match_date_too_far():
    """Identical titles but 10 days apart → no match."""
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)
    other = base + timedelta(days=10)
    a = _market("kalshi", "K1", "Bitcoin above 80000", base)
    b = _market("polymarket", "P1", "Bitcoin above 80000", other)
    results = find_matches([a], [b], date_window_days=3)
    assert len(results) == 0


def test_no_match_score_too_low():
    """Completely unrelated titles → no match."""
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)
    a = _market("kalshi", "K1", "Bitcoin price", base)
    b = _market("polymarket", "P1", "Ethereum weather forecast", base)
    results = find_matches([a], [b])
    assert len(results) == 0


def test_category_not_a_hard_filter():
    """Different categories but similar titles → match found."""
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)
    a = _market("kalshi", "K1", "Bitcoin above 80000", base)
    a.category = "crypto"
    b = _market("polymarket", "P1", "Bitcoin above 80000", base + timedelta(days=1))
    b.category = "economics"
    results = find_matches([a], [b], date_window_days=3)
    assert len(results) >= 1
