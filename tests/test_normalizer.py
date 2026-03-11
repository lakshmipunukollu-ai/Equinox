"""Unit tests for Kalshi and Polymarket normalizers."""

import logging

import pytest

from equinox.normalizer import kalshi as kalshi_norm
from equinox.normalizer import polymarket as polymarket_norm


def test_kalshi_normalize_field_types(sample_kalshi_series):
    """id starts with kalshi:, yes_price 0-1, no_price=1-yes_price, etc."""
    markets = kalshi_norm.normalize(sample_kalshi_series)
    assert len(markets) >= 1
    m = markets[0]
    assert m.id.startswith("kalshi:")
    assert 0 <= m.yes_price <= 1.0
    assert m.no_price == pytest.approx(1.0 - m.yes_price, abs=0.0001)
    assert m.close_time.tzinfo is not None
    assert m.category == m.category.lower()
    assert "raw" in m.model_dump()
    assert m.fee_model == "additive"
    assert m.fee_rate == pytest.approx(0.07, abs=0.001)


def test_kalshi_normalize_returns_one_market_per_nested_market(sample_kalshi_series):
    """Series with 2 markets → 2 Market objects."""
    markets = kalshi_norm.normalize(sample_kalshi_series)
    assert len(markets) == 2


def test_kalshi_normalize_missing_yes_ask():
    """Series with yes_ask absent → yes_price defaults to 0.5."""
    series = [
        {
            "event_title": "Test",
            "category": "Test",
            "markets": [
                {
                    "ticker": "TICK",
                    "yes_subtitle": "Outcome",
                    "yes_bid": 50,
                    "yes_ask": None,
                    "last_price": None,
                    "volume": 0,
                    "close_ts": 1711382400,
                }
            ],
        }
    ]
    markets = kalshi_norm.normalize(series)
    assert len(markets) == 1
    assert markets[0].yes_price == pytest.approx(0.5, abs=0.0001)


def test_kalshi_normalize_zero_yes_bid():
    """yes_bid=0 is valid (near-certain NO) — not skipped."""
    series = [
        {
            "event_title": "Test",
            "category": "Test",
            "markets": [
                {
                    "ticker": "TICK",
                    "yes_subtitle": "Outcome",
                    "yes_bid": 0,
                    "yes_ask": None,
                    "last_price": None,
                    "volume": 0,
                    "close_ts": 1711382400,
                }
            ],
        }
    ]
    markets = kalshi_norm.normalize(series)
    assert len(markets) == 1
    assert markets[0].yes_price == pytest.approx(0.0, abs=0.0001)


def test_kalshi_normalize_decimal_price_used_as_is(caplog):
    """yes_bid=0.45 (float) → used as-is, WARNING logged."""
    series = [
        {
            "event_title": "Test",
            "category": "Test",
            "markets": [
                {
                    "ticker": "TICK",
                    "yes_subtitle": "Outcome",
                    "yes_bid": 0.45,
                    "yes_ask": None,
                    "last_price": None,
                    "volume": 0,
                    "close_ts": 1711382400,
                }
            ],
        }
    ]
    with caplog.at_level(logging.WARNING, logger="equinox.trace"):
        markets = kalshi_norm.normalize(series)
    assert len(markets) == 1
    assert markets[0].yes_price == pytest.approx(0.45, abs=0.0001)
    assert any(
        "0.45" in r.message or "decimal" in r.message.lower()
        for r in caplog.records
    )


def test_kalshi_normalize_over_100_cents_clamped(caplog):
    """yes_bid=150 → clamped to 0.5, WARNING logged."""
    series = [
        {
            "event_title": "Test",
            "category": "Test",
            "markets": [
                {
                    "ticker": "TICK",
                    "yes_subtitle": "Outcome",
                    "yes_bid": 150,
                    "yes_ask": None,
                    "last_price": None,
                    "volume": 0,
                    "close_ts": 1711382400,
                }
            ],
        }
    ]
    with caplog.at_level(logging.WARNING, logger="equinox.trace"):
        markets = kalshi_norm.normalize(series)
    assert len(markets) == 1
    assert markets[0].yes_price == pytest.approx(0.5, abs=0.0001)
    assert any(
        "150" in r.message
        or "clamp" in r.message.lower()
        or "exceeds" in r.message.lower()
        for r in caplog.records
    )


def test_polymarket_normalize_field_types(sample_polymarket_list):
    """Same field constraints, fee_model=multiplicative."""
    markets = polymarket_norm.normalize(sample_polymarket_list)
    assert len(markets) >= 1
    m = markets[0]
    assert m.id.startswith("polymarket:")
    assert 0 <= m.yes_price <= 1.0
    assert m.no_price == pytest.approx(1.0 - m.yes_price, abs=0.0001)
    assert m.fee_model == "multiplicative"
    assert m.fee_rate == pytest.approx(0.02, abs=0.001)


def test_polymarket_normalize_null_best_ask():
    """bestAsk=None → yes_price falls back to lastTradePrice."""
    raw = [
        {
            "id": "p1",
            "question": "Test",
            "category": "test",
            "bestBid": "0.40",
            "bestAsk": None,
            "lastTradePrice": "0.42",
            "volume": "1000",
            "liquidity": "500",
            "endDateIso": "2024-12-31T23:59:59Z",
            "slug": "test",
            "outcomes": ["Yes", "No"],
        }
    ]
    markets = polymarket_norm.normalize(raw)
    assert len(markets) == 1
    assert markets[0].yes_price == pytest.approx(0.42, abs=0.0001)


def test_polymarket_normalize_all_prices_null():
    """bestAsk=None, lastTradePrice=None → yes_price defaults to 0.5."""
    raw = [
        {
            "id": "p1",
            "question": "Test",
            "category": "test",
            "bestBid": None,
            "bestAsk": None,
            "lastTradePrice": None,
            "volume": "1000",
            "liquidity": "500",
            "endDateIso": "2024-12-31T23:59:59Z",
            "slug": "test",
            "outcomes": ["Yes", "No"],
        }
    ]
    markets = polymarket_norm.normalize(raw)
    assert len(markets) == 1
    assert markets[0].yes_price == pytest.approx(0.5, abs=0.0001)


def test_normalize_missing_date():
    """close_ts=None → close_time is timezone-aware, no exception."""
    series = [
        {
            "event_title": "Test",
            "category": "Test",
            "markets": [
                {
                    "ticker": "TICK",
                    "yes_subtitle": "Outcome",
                    "yes_bid": 50,
                    "yes_ask": 52,
                    "volume": 0,
                    "close_ts": None,
                }
            ],
        }
    ]
    markets = kalshi_norm.normalize(series)
    assert len(markets) == 1
    assert markets[0].close_time.tzinfo is not None
