"""Shared fixtures for Equinox tests."""

from datetime import datetime, timezone

import pytest

from equinox.models import Market, MatchResult
from equinox.normalizer import kalshi as kalshi_norm
from equinox.normalizer import polymarket as polymarket_norm

# Titles chosen to guarantee fuzzy match score >= 0.75 for test_api.py
SAMPLE_KALSHI_SERIES = [
    {
        "event_title": "Bitcoin price today",
        "category": "Crypto",
        "markets": [
            {
                "ticker": "KXBTCD-26MAR",
                "yes_subtitle": "Above $80,000",
                "yes_bid": 45,
                "yes_ask": 48,
                "last_price": 46,
                "volume": 5000,
                "close_ts": 1711382400,  # Unix timestamp
            },
            {
                "ticker": "KXBTCD-26MAR2",
                "yes_subtitle": "Below $80,000",
                "yes_bid": 52,
                "yes_ask": 55,
                "last_price": 53,
                "volume": 3000,
                "close_ts": 1711382400,
            },
        ],
    }
]

SAMPLE_POLYMARKET_LIST = [
    {
        "id": "poly-btc-1",
        "question": "Bitcoin above $80000 today",
        "category": "crypto",
        "bestBid": "0.44",
        "bestAsk": "0.47",
        "lastTradePrice": "0.45",
        "volume": "120000",
        "liquidity": "50000",
        "endDateIso": "2024-03-26T23:59:59Z",
        "slug": "bitcoin-80k",
        "outcomes": ["Yes", "No"],
    },
    {
        "id": "poly-btc-2",
        "question": "Bitcoin below $80000 by end of March",
        "category": "crypto",
        "bestBid": "0.50",
        "bestAsk": "0.53",
        "lastTradePrice": "0.51",
        "volume": "80000",
        "liquidity": "30000",
        "endDateIso": "2024-03-31T23:59:59Z",
        "slug": "bitcoin-80k-march",
        "outcomes": ["Yes", "No"],
    },
]


@pytest.fixture
def sample_kalshi_series():
    """Raw Kalshi V1 series list with 2 nested markets."""
    return SAMPLE_KALSHI_SERIES


@pytest.fixture
def sample_polymarket_list():
    """Raw Polymarket list with 2 markets."""
    return SAMPLE_POLYMARKET_LIST


@pytest.fixture
def sample_kalshi_markets(sample_kalshi_series):
    """Normalized Kalshi Market list."""
    return kalshi_norm.normalize(sample_kalshi_series)


@pytest.fixture
def sample_polymarket_markets(sample_polymarket_list):
    """Normalized Polymarket Market list."""
    return polymarket_norm.normalize(sample_polymarket_list)


@pytest.fixture
def sample_match_result(sample_kalshi_markets, sample_polymarket_markets):
    """One MatchResult between Kalshi and Polymarket."""
    return MatchResult(
        market_a=sample_kalshi_markets[0],
        market_b=sample_polymarket_markets[0],
        score=0.85,
        method="fuzzy",
        explanation="Fuzzy title similarity for Bitcoin price.",
    )


@pytest.fixture(autouse=True)
async def setup_db(tmp_path, monkeypatch):
    """Create DB in tmp_path for each test. Autouse."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    from equinox.store.db import init_db

    await init_db()
