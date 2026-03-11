"""Single-responsibility: expose the Equinox pipeline as HTTP endpoints."""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from equinox.logger import log_trace
from equinox.matcher.engine import find_matches
from equinox.normalizer import kalshi as kalshi_norm
from equinox.normalizer import polymarket as polymarket_norm
from equinox.router.engine import route
from equinox.store.db import save_decision, save_markets, save_matches
from equinox.venues import kalshi, polymarket

router = APIRouter()


@router.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "venues": ["kalshi", "polymarket"]}


@router.get("/search")
async def search(query: str = Query(default="", description="Search term")):
    """Search both venues, normalize, match, and persist."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query parameter is required")

    results = await asyncio.gather(
        kalshi.fetch_markets(query, page_size=100),
        polymarket.fetch_markets(query, limit=100),
        return_exceptions=True,
    )
    kalshi_raw = results[0] if not isinstance(results[0], BaseException) else []
    polymarket_raw = results[1] if not isinstance(results[1], BaseException) else []

    if isinstance(results[0], BaseException):
        log_trace(
            "api",
            f"Kalshi fetch raised unexpected exception — returning empty results. "
            f"Error: {results[0]}",
            {"error": str(results[0])},
            level="error",
        )
    if isinstance(results[1], BaseException):
        log_trace(
            "api",
            f"Polymarket fetch raised unexpected exception — returning empty results. "
            f"Error: {results[1]}",
            {"error": str(results[1])},
            level="error",
        )

    kalshi_markets = kalshi_norm.normalize(kalshi_raw)[:100]
    polymarket_markets = polymarket_norm.normalize(polymarket_raw)[:100]
    await save_markets(kalshi_markets + polymarket_markets)

    matches = find_matches(kalshi_markets, polymarket_markets)
    await save_matches(matches)

    matched_ids_a = {m.market_a.id for m in matches}
    matched_ids_b = {m.market_b.id for m in matches}
    unmatched_kalshi = [m for m in kalshi_markets if m.id not in matched_ids_a]
    unmatched_polymarket = [m for m in polymarket_markets if m.id not in matched_ids_b]

    return {
        "query": query,
        "total_kalshi": len(kalshi_markets),
        "total_polymarket": len(polymarket_markets),
        "total_matches": len(matches),
        "matches": [m.model_dump(mode="json") for m in matches],
        "unmatched_kalshi": [m.model_dump(mode="json") for m in unmatched_kalshi],
        "unmatched_polymarket": [
            m.model_dump(mode="json") for m in unmatched_polymarket
        ],
    }


@router.get("/route")
async def route_endpoint(
    query: str = Query(default="", description="Search term"),
    order_size: float = Query(100.0, ge=0),
):
    """Route to best venue for a hypothetical order."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query parameter is required")
    results = await asyncio.gather(
        kalshi.fetch_markets(query, page_size=100),
        polymarket.fetch_markets(query, limit=100),
        return_exceptions=True,
    )
    kalshi_raw = results[0] if not isinstance(results[0], BaseException) else []
    polymarket_raw = results[1] if not isinstance(results[1], BaseException) else []

    if isinstance(results[0], BaseException):
        log_trace(
            "api",
            f"Kalshi fetch raised unexpected exception — returning empty results. "
            f"Error: {results[0]}",
            {"error": str(results[0])},
            level="error",
        )
    if isinstance(results[1], BaseException):
        log_trace(
            "api",
            f"Polymarket fetch raised unexpected exception — returning empty results. "
            f"Error: {results[1]}",
            {"error": str(results[1])},
            level="error",
        )

    kalshi_markets = kalshi_norm.normalize(kalshi_raw)[:100]
    polymarket_markets = polymarket_norm.normalize(polymarket_raw)[:100]
    all_markets = kalshi_markets + polymarket_markets
    matches = find_matches(kalshi_markets, polymarket_markets)

    try:
        decision = route(matches, order_size=order_size, all_markets=all_markets)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"No markets found for query '{query}' on any venue. "
            "Try a broader search term.",
        )

    await save_decision(decision)
    return decision.model_dump(mode="json")


@router.get("/debug/kalshi")
async def debug_kalshi(
    q: str = Query(default="", description="Search term"),
    page_size: int = Query(5, ge=1),
):
    """Raw Kalshi response for debugging."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q parameter is required")
    raw = await kalshi.fetch_markets(q, page_size=page_size)
    out = {
        "query": q,
        "raw_count": len(raw),
        "raw_response": raw,
    }
    if len(raw) == 0:
        out["warning"] = (
            "Kalshi returned 0 results. Check logs for HTTP errors or try a different query."
        )
    return out
