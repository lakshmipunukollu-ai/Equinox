"""Unit tests for routing engine."""

from datetime import datetime, timezone

import pytest

from equinox.models import Market, MatchResult
from equinox.router.engine import route


def _market(
    venue: str,
    mid: str,
    title: str,
    yes_price: float = 0.5,
    liquidity: float = 5000,
    volume: float = 10000,
) -> Market:
    return Market(
        id=f"{venue}:{mid}",
        venue=venue,
        title=title,
        category="test",
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        yes_bid=yes_price - 0.02,
        yes_ask=yes_price + 0.02,
        spread_width=0.04,
        volume=volume,
        liquidity=liquidity,
        close_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        price_updated_at=None,
        price_source="mid",
        fee_rate=0.07 if venue == "kalshi" else 0.02,
        fee_model="additive" if venue == "kalshi" else "multiplicative",
        is_binary=True,
        url="https://example.com",
        raw={},
    )


def test_higher_liquidity_wins():
    """Higher liquidity market wins with order_size=100 when fees are equal."""
    # Use same fee_model and fee_rate so fee_score doesn't dominate
    high = _market("kalshi", "K1", "Market A", yes_price=0.5, liquidity=10000)
    low = _market("polymarket", "P1", "Market B", yes_price=0.5, liquidity=1000)
    # Override to equalize fees so liquidity is the differentiator
    high.fee_rate = 0.0
    high.fee_model = "multiplicative"
    low.fee_rate = 0.0
    low.fee_model = "multiplicative"
    match = MatchResult(
        market_a=high,
        market_b=low,
        score=0.9,
        method="fuzzy",
        explanation="Test",
    )
    decision = route([match], order_size=100.0)
    assert decision.selected_venue == "kalshi"
    # liquidity_score: 1 - 100/10000 = 0.99 vs 1 - 100/1000 = 0.90
    high_util = 100 / 10000
    low_util = 100 / 1000
    assert 1.0 - high_util > 1.0 - low_util


def test_reasoning_string_is_narrative():
    """Reasoning contains venue, dollar amount, score, numbers, simulation disclaimer."""
    m1 = _market("kalshi", "K1", "Market A", liquidity=5000)
    m2 = _market("polymarket", "P1", "Market B", liquidity=5000)
    match = MatchResult(
        market_a=m1,
        market_b=m2,
        score=0.9,
        method="fuzzy",
        explanation="Test",
    )
    decision = route([match], order_size=100.0)
    assert "kalshi" in decision.reasoning.lower() or "polymarket" in decision.reasoning.lower()
    assert "100" in decision.reasoning or "100.0" in decision.reasoning
    assert "score" in decision.reasoning.lower()
    assert "simulation" in decision.reasoning.lower() or "no real order" in decision.reasoning.lower()
    assert len(decision.reasoning.split()) >= 100


def test_alternatives_contains_all_non_winner():
    """4 unique candidates → alternatives has 3 entries."""
    k1 = _market("kalshi", "K1", "A", liquidity=5000)
    k2 = _market("kalshi", "K2", "B", liquidity=5000)
    p1 = _market("polymarket", "P1", "C", liquidity=5000)
    p2 = _market("polymarket", "P2", "D", liquidity=5000)
    match1 = MatchResult(market_a=k1, market_b=p1, score=0.9, method="fuzzy", explanation="1")
    match2 = MatchResult(market_a=k2, market_b=p2, score=0.9, method="fuzzy", explanation="2")
    decision = route([match1, match2])
    assert len(decision.alternatives) == 3


def test_empty_matches_raises_value_error():
    """route([], all_markets=None) → ValueError."""
    with pytest.raises(ValueError, match="No matched markets"):
        route([], all_markets=None)


def test_large_price_divergence_triggers_arb_warning():
    """25¢ divergence → price_divergence set, arb warning in reasoning."""
    m_a = _market("kalshi", "K1", "Market", yes_price=0.40, liquidity=5000)
    m_b = _market("polymarket", "P1", "Market", yes_price=0.65, liquidity=5000)
    match = MatchResult(
        market_a=m_a,
        market_b=m_b,
        score=0.9,
        method="fuzzy",
        explanation="Test",
    )
    decision = route([match])
    assert decision.price_divergence == pytest.approx(0.25, abs=0.001)
    assert decision.price_divergence > 0.10
    assert any(
        p in decision.reasoning.lower()
        for p in ["arbitrage", "divergence", "data quality"]
    )


def test_price_divergence_none_on_single_venue_fallback():
    """Single-venue fallback → price_divergence is None."""
    m = _market("kalshi", "K1", "Market", liquidity=5000)
    decision = route([], all_markets=[m])
    assert decision.price_divergence is None
