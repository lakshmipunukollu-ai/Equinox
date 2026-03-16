"""
Single-responsibility: select the best venue for a hypothetical order.
Venue-agnostic — zero if/else on venue names. No imports from venues/ or normalizer/.

Scoring formula (weights must match ASSUMPTIONS.md):
  final_score = (liquidity_score * 0.4) + (volume_score * 0.2) +
                (fee_score * 0.2) + (execution_score * 0.2)
  final_score *= source_weight
"""

import os
from datetime import datetime, timezone

from equinox.logger import log_trace
from equinox.models import Market, MatchResult, RoutingDecision

VOLUME_CAP = float(os.getenv("VOLUME_CAP", "100000"))
FEE_CAP = float(os.getenv("FEE_CAP", "0.10"))
SPREAD_WIDTH_CAP = float(os.getenv("SPREAD_WIDTH_CAP", "0.20"))
MAX_PRICE_AGE_HOURS = float(os.getenv("MAX_PRICE_AGE_HOURS", "24"))
MAX_DIVERGENCE = float(os.getenv("MAX_DIVERGENCE", "0.20"))


def _all_in_cost(market: Market) -> float:
    """All-in cost per share (yes_price + fee in same units)."""
    if market.fee_model == "additive":
        return min(market.yes_price + market.fee_rate, 1.0)
    return min(market.yes_price * (1.0 + market.fee_rate), 1.0)


def _execution_quality_score(market: Market, order_size: float, all_in: float) -> tuple[float, str]:
    """
    Compute execution quality score 0-100 and letter grade.
    Weights: spread 40%, liquidity utilization 40%, fee cost 20%.
    """
    score = 0.0
    spread = market.spread_width if market.spread_width is not None else 0.01
    if spread <= 0.001:
        score += 40
    elif spread <= 0.002:
        score += 30
    elif spread <= 0.005:
        score += 20
    else:
        score += 10

    utilization = order_size / max(market.liquidity, 1.0)
    if utilization < 0.01:
        score += 40
    elif utilization < 0.05:
        score += 30
    elif utilization < 0.10:
        score += 20
    else:
        score += 10

    if all_in < 0.001:
        score += 20
    elif all_in < 0.005:
        score += 15
    else:
        score += 5

    if score >= 90:
        return score, "A+"
    if score >= 80:
        return score, "A"
    if score >= 70:
        return score, "B+"
    if score >= 60:
        return score, "B"
    if score >= 50:
        return score, "C"
    return score, "D"


def _score_market(market: Market, order_size: float) -> tuple[float, dict]:
    """Score a market for routing. Returns (final_score, components_dict)."""
    # Liquidity: order-size-aware
    liquidity_utilization = order_size / max(market.liquidity, 1.0)
    liquidity_score = max(0.0, 1.0 - liquidity_utilization)

    # Volume: market health proxy (not order-size-aware)
    volume_score = min(market.volume / VOLUME_CAP, 1.0)

    # Fee-adjusted price: formula depends on fee_model
    if market.fee_model == "additive":
        all_in_price = market.yes_price + market.fee_rate
    else:
        all_in_price = market.yes_price * (1.0 + market.fee_rate)
    all_in_price = min(all_in_price, 1.0)

    # Fee score: friction only
    fee_cost = all_in_price - market.yes_price
    fee_score = max(0.0, 1.0 - (fee_cost / FEE_CAP))

    # Execution score
    if market.spread_width is not None:
        execution_score = max(0.0, 1.0 - (market.spread_width / SPREAD_WIDTH_CAP))
        execution_method = "spread_width"
    else:
        execution_score = max(0.0, 1.0 - abs(market.yes_price - 0.5) * 2.0)
        execution_method = "conviction_fallback"

    # Price source penalty
    source_weight = {
        "mid": 1.0,
        "ask": 1.0,
        "bid": 0.9,
        "last": 0.8,
        "default": 0.5,
    }.get(market.price_source, 1.0)

    # Staleness warning
    if market.price_updated_at is not None:
        age_seconds = (
            datetime.now(timezone.utc) - market.price_updated_at
        ).total_seconds()
        if age_seconds > 3600:
            log_trace(
                "route",
                f"Price for '{market.title}' is {age_seconds/3600:.1f}h old — "
                "may not reflect current market.",
                {"id": market.id, "age_hours": round(age_seconds / 3600, 2)},
                level="warning",
            )

    final = (
        (liquidity_score * 0.4)
        + (volume_score * 0.2)
        + (fee_score * 0.2)
        + (execution_score * 0.2)
    ) * source_weight
    final = round(max(0.0, final), 4)

    return final, {
        "liquidity_score": round(liquidity_score, 4),
        "liquidity_util": round(liquidity_utilization, 4),
        "volume_score": round(volume_score, 4),
        "fee_score": round(fee_score, 4),
        "all_in_price": round(all_in_price, 6),
        "execution_score": round(execution_score, 4),
        "execution_method": execution_method,
        "price_source": market.price_source,
        "source_weight": source_weight,
        "final_score": final,
    }


def route(
    matches: list[MatchResult],
    order_size: float = 100.0,
    all_markets: list[Market] | None = None,
) -> RoutingDecision:
    """Select best venue for a hypothetical order."""
    now = datetime.now(timezone.utc)
    max_age_seconds = MAX_PRICE_AGE_HOURS * 3600

    if all_markets:
        non_stale = [
            m
            for m in all_markets
            if m.price_updated_at is None
            or (now - m.price_updated_at).total_seconds() <= max_age_seconds
        ]
        if len(non_stale) < len(all_markets):
            log_trace(
                "route",
                f"Excluded {len(all_markets) - len(non_stale)} stale markets "
                f"(price older than {MAX_PRICE_AGE_HOURS}h).",
                {"excluded": len(all_markets) - len(non_stale)},
                level="warning",
            )
        all_markets = non_stale
        non_stale_ids = {m.id for m in all_markets}
        matches = [m for m in matches if m.market_a.id in non_stale_ids and m.market_b.id in non_stale_ids]

    valid_matches = []
    for m in matches:
        div = abs(m.market_a.yes_price - m.market_b.yes_price)
        if div > MAX_DIVERGENCE:
            log_trace(
                "route",
                "Match invalidated: price divergence exceeds MAX_DIVERGENCE.",
                {
                    "market_a_id": m.market_a.id,
                    "market_b_id": m.market_b.id,
                    "yes_a": m.market_a.yes_price,
                    "yes_b": m.market_b.yes_price,
                    "divergence": round(div, 4),
                    "max_allowed": MAX_DIVERGENCE,
                },
                level="warning",
            )
        else:
            valid_matches.append(m)
    matches = valid_matches

    if not matches:
        if all_markets:
            log_trace(
                "route",
                "No cross-venue matches found — falling back to single-venue routing "
                "across all available markets. Cross-venue price comparison is not possible.",
                {"market_count": len(all_markets), "order_size": order_size},
                level="warning",
            )
            tuples_fallback = [
                (m, *_score_market(m, order_size)) for m in all_markets
            ]
            seen: dict[str, tuple] = {}
            for entry in tuples_fallback:
                mkt, scr = entry[0], entry[1]
                if mkt.id not in seen or scr > seen[mkt.id][1]:
                    seen[mkt.id] = entry
            tuples = sorted(seen.values(), key=lambda t: t[1], reverse=True)
        else:
            log_trace(
                "route",
                "Cannot produce a routing decision — no equivalent markets were found "
                "and no individual markets were provided. Run find_matches first or try "
                "a different query.",
                {"order_size": order_size},
                level="warning",
            )
            raise ValueError("No matched markets to route — run find_matches first")
    else:
        raw_tuples = []
        for match in matches:
            raw_tuples.append(
                (match.market_a, *_score_market(match.market_a, order_size))
            )
            raw_tuples.append(
                (match.market_b, *_score_market(match.market_b, order_size))
            )
        seen = {}
        for entry in raw_tuples:
            mkt, scr = entry[0], entry[1]
            if mkt.id not in seen or scr > seen[mkt.id][1]:
                seen[mkt.id] = entry
        tuples = sorted(seen.values(), key=lambda t: t[1], reverse=True)

    winner, score, components = tuples[0]
    is_single_venue_fallback = not matches

    # For comparison table we need the best market from the *other* venue (not just second-best overall).
    if len(tuples) > 1:
        other_venue_tuples = [(m, sc, comp) for m, sc, comp in tuples[1:] if m.venue != winner.venue]
        if other_venue_tuples:
            loser, loser_score, loser_components = other_venue_tuples[0]
        else:
            loser, loser_score, loser_components = tuples[1]
    else:
        loser, loser_score, loser_components = winner, 0.0, components
        log_trace(
            "route",
            "Only one candidate found — loser comparison will mirror winner.",
            {"winner": winner.id},
            level="warning",
        )

    price_divergence: float | None = None
    arb_warning = ""
    if not is_single_venue_fallback and loser is not winner:
        winner_price = float(winner.yes_price)
        loser_price = float(loser.yes_price)
        price_divergence = round(abs(winner_price - loser_price), 4)
        assert price_divergence >= 0, "price_divergence must be non-negative"
        assert price_divergence == round(
            abs(winner_price - loser_price), 4
        ), "price_divergence must equal abs(winner_price - loser_price) in same unit"
        if price_divergence > 0.10:
            arb_warning = (
                f"⚠️ LARGE PRICE DIVERGENCE: {winner.venue.capitalize()} prices YES "
                f"at ${winner.yes_price:.4f} vs {loser.venue.capitalize()} at "
                f"${loser.yes_price:.4f} — a ${price_divergence:.4f} gap on "
                "what should be the same event. This may indicate: (1) a genuine "
                "arbitrage opportunity, (2) a stale price on one venue, or (3) a "
                "false match (different events incorrectly linked). Verify match "
                "quality before acting on this routing decision. "
            )
            log_trace(
                "route",
                arb_warning,
                {
                    "winner_price": winner.yes_price,
                    "loser_price": loser.yes_price,
                    "divergence": price_divergence,
                },
                level="warning",
            )

    if is_single_venue_fallback:
        lead_metric = (
            "overall score (single-venue — no cross-venue comparison available)"
        )
    else:
        _deltas = {
            "liquidity": components["liquidity_score"]
            - loser_components["liquidity_score"],
            "fee": components["fee_score"] - loser_components["fee_score"],
            "execution": components["execution_score"]
            - loser_components["execution_score"],
            "volume": components["volume_score"]
            - loser_components["volume_score"],
        }
        lead_metric = max(_deltas, key=_deltas.get)

    venues_considered = sorted(set(t[0].venue for t in tuples))

    time_to_close = winner.close_time - datetime.now(timezone.utc)
    if time_to_close.total_seconds() < 86400:
        log_trace(
            "route",
            f"Selected market '{winner.title}' expires in "
            f"{time_to_close.total_seconds()/3600:.1f} hours — very short window.",
            {
                "winner": winner.id,
                "hours_remaining": time_to_close.total_seconds() / 3600,
            },
            level="warning",
        )

    winner_all_in = components["all_in_price"]
    loser_all_in = loser_components["all_in_price"]
    estimated_savings = order_size * (loser_all_in - winner_all_in)
    estimated_savings = round(max(0.0, estimated_savings), 2)
    assert estimated_savings >= 0, (
        "Winner should always cost less: "
        f"winner_all_in={winner_all_in}, loser_all_in={loser_all_in}, "
        f"order_size={order_size}"
    )
    estimated_savings_text = f"Save ~${estimated_savings:.2f}" if estimated_savings > 0 else "—"

    winner_spread_desc = (
        f"${winner.spread_width:.4f} bid-ask spread"
        if winner.spread_width is not None
        else "no spread data (conviction proxy used)"
    )
    loser_spread_desc = (
        f"${loser.spread_width:.4f} bid-ask spread"
        if loser.spread_width is not None
        else "no spread data (conviction proxy used)"
    )
    cross_venue_note = (
        "No cross-venue match was available for this query — single-venue "
        "routing applied. Price and liquidity comparison across venues is "
        "not possible. "
        if is_single_venue_fallback
        else ""
    )

    # Fee explanation: compare all-in cost, not raw fee rates (additive vs multiplicative)
    winner_cheaper = winner_all_in < loser_all_in
    fee_explanation = (
        f"{winner.venue.capitalize()}'s {winner.fee_rate:.0%} {winner.fee_model} fee "
        f"costs less at this price level (all-in {winner_all_in:.4f}) than "
        f"{loser.venue.capitalize()}'s {loser.fee_rate:.0%} {loser.fee_model} fee "
        f"(all-in {loser_all_in:.4f}). "
        if winner_cheaper
        else f"All-in cost: {winner.venue.capitalize()} {winner_all_in:.4f} vs "
        f"{loser.venue.capitalize()} {loser_all_in:.4f}. "
    )

    def _source_note(market: Market, comps: dict) -> str:
        sw = comps["source_weight"]
        if sw < 1.0:
            return (
                f" [PRICE RELIABILITY PENALTY: {market.venue} price sourced "
                f"from '{market.price_source}' — reliability weight {sw:.1f}× "
                f"applied to final score. Unweighted score was "
                f"{comps['final_score']/sw:.3f}; weighted: {comps['final_score']:.3f}.]"
            )
        return ""

    reasoning = (
        f"Selected {winner.venue} market '{winner.title}' as the best venue "
        f"for a ${order_size:,.0f} order. "
        f"{cross_venue_note}"
        f"{winner.venue.capitalize()} yes_price: {winner.yes_price:.4f} "
        f"(source: {winner.price_source}) vs "
        f"{loser.venue.capitalize()} yes_price: {loser.yes_price:.4f} "
        f"(source: {loser.price_source}) — "
        f"a ${abs(winner.yes_price - loser.yes_price):.4f} mid-price difference. "
        f"{fee_explanation}"
        f"{arb_warning}"
        f"{winner.venue.capitalize()} execution: {winner_spread_desc} vs "
        f"{loser.venue.capitalize()} execution: {loser_spread_desc}. "
        f"Execution score: {components['execution_score']:.3f} vs "
        f"{loser_components['execution_score']:.3f} "
        f"(method: {components['execution_method']}). "
        f"{winner.venue.capitalize()} liquidity score: {components['liquidity_score']:.3f} "
        f"(${winner.liquidity:,.0f} available, "
        f"order utilization: {components['liquidity_util']:.1%}) vs "
        f"{loser.venue.capitalize()} liquidity score: {loser_components['liquidity_score']:.3f} "
        f"(${loser.liquidity:,.0f} available, "
        f"order utilization: {loser_components['liquidity_util']:.1%}). "
        f"{winner.venue.capitalize()} volume score: {components['volume_score']:.3f} vs "
        f"{loser.venue.capitalize()} volume score: {loser_components['volume_score']:.3f}. "
        f"{_source_note(winner, components)}"
        f"{_source_note(loser, loser_components)}"
        f"Composite scores: {winner.venue.capitalize()} {score:.3f} vs "
        f"{loser.venue.capitalize()} {loser_score:.3f}. "
        f"{winner.venue.capitalize()} is the recommended venue primarily due to "
        f"superior {lead_metric}. "
        f"Evaluated {len(tuples)} candidate(s) across "
        f"{len(venues_considered)} venue(s): {venues_considered}. "
        "— NOTE: This is a simulation. No real order has been placed."
    )

    alternatives = [
        {"id": m.id, "venue": m.venue, "title": m.title, **comps}
        for m, _, comps in tuples[1:]
    ]

    # Execution quality grades
    winner_eq_score, winner_eq_grade = _execution_quality_score(winner, order_size, winner_all_in)
    loser_eq_score, loser_eq_grade = _execution_quality_score(loser, order_size, loser_all_in)

    # Confidence breakdown (0-1 scale for UI bars). Overall = weighted average of components.
    if loser is not winner:
        min_price = min(winner.yes_price, loser.yes_price) or 0.01
        price_diff = abs(winner.yes_price - loser.yes_price)
        price_signal = 0.0 if price_diff == 0 else min(1.0, (price_diff / min_price))
        fee_advantage = min(1.0, abs(winner.fee_rate - loser.fee_rate) * 100)
    else:
        price_signal = 0.5
        fee_advantage = 0.5
    confidence_breakdown = {
        "price_signal": round(price_signal, 2),
        "fee_advantage": round(fee_advantage, 2),
        "liquidity_score": round(components["liquidity_score"], 2),
        "execution_score": round(components["execution_score"], 2),
    }
    # Overall = weighted average of all components (not just liquidity)
    overall = (
        confidence_breakdown["price_signal"] * 0.25
        + confidence_breakdown["fee_advantage"] * 0.25
        + confidence_breakdown["liquidity_score"] * 0.25
        + confidence_breakdown["execution_score"] * 0.25
    )
    confidence_breakdown["overall"] = round(min(1.0, max(0.0, overall)), 2)

    log_trace(
        "route",
        reasoning,
        {
            "selected": winner.id,
            "venue": winner.venue,
            "score": score,
            "order_size": order_size,
            "alternatives_count": len(alternatives),
        },
    )

    runner_up = loser if loser is not winner else None
    return RoutingDecision(
        selected_venue=winner.venue,
        selected_market=winner,
        score=score,
        reasoning=reasoning,
        alternatives=alternatives,
        price_divergence=price_divergence,
        simulation=True,
        runner_up_market=runner_up,
        winner_all_in_cost=round(winner_all_in, 4),
        loser_all_in_cost=round(loser_all_in, 4),
        estimated_savings=estimated_savings,
        estimated_savings_text=estimated_savings_text,
        winner_execution_quality=winner_eq_grade,
        loser_execution_quality=loser_eq_grade,
        confidence_breakdown=confidence_breakdown,
    )
