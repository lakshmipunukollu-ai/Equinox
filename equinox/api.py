"""Single-responsibility: expose the Equinox pipeline as HTTP endpoints."""

import asyncio
import time
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from equinox.logger import log_trace
from equinox.matcher.engine import find_matches
from equinox.models import Market, MatchResult
from equinox.normalizer import kalshi as kalshi_norm
from equinox.normalizer import polymarket as polymarket_norm
from equinox.router.engine import route
from equinox.store.db import save_decision, save_markets, save_matches
from equinox.venues import kalshi, polymarket


class RouteByMatchBody(BaseModel):
    market_a: dict
    market_b: dict
    order_size: float = 100.0


router = APIRouter()

# In-memory cache: max 20 entries, 5 min TTL. Key -> { "data": ..., "timestamp": ... }
CACHE_TTL_SEC = 5 * 60
CACHE_MAX_SIZE = 20
_search_cache: OrderedDict = OrderedDict()
_route_cache: OrderedDict = OrderedDict()

API_TIMEOUT_SEC = 8.0


def _cache_get(cache: OrderedDict, key: str):
    if key not in cache:
        return None
    entry = cache[key]
    if time.time() - entry["timestamp"] > CACHE_TTL_SEC:
        del cache[key]
        return None
    cache.move_to_end(key)
    return entry["data"]


def _cache_set(cache: OrderedDict, key: str, data):
    while len(cache) >= CACHE_MAX_SIZE and cache:
        cache.popitem(last=False)
    cache[key] = {"data": data, "timestamp": time.time()}
    cache.move_to_end(key)


async def _fetch_with_timeout(coro, label: str):
    """Run a single fetch with 8s timeout. Returns (result, error_msg)."""
    try:
        return await asyncio.wait_for(coro, timeout=API_TIMEOUT_SEC), None
    except asyncio.TimeoutError:
        log_trace("api", f"{label} timed out after {API_TIMEOUT_SEC}s", {}, level="warning")
        return [], f"{label} timed out after {API_TIMEOUT_SEC}s"
    except Exception as e:
        log_trace("api", f"{label} failed: {e}", {"error": str(e)}, level="error")
        return [], str(e)


@router.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "venues": ["kalshi", "polymarket"]}


@router.get("/search")
async def search(query: str = Query(default="", description="Search term")):
    """Search both venues, normalize, match, and persist. Cached 5 min. 8s timeout per venue."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query parameter is required")

    cache_key = f"search:{query.strip().lower()}"
    cached = _cache_get(_search_cache, cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    kalshi_task = _fetch_with_timeout(kalshi.fetch_markets(query, page_size=100), "Kalshi")
    poly_task = _fetch_with_timeout(polymarket.fetch_markets(query, limit=100), "Polymarket")
    (kalshi_result, kalshi_err), (poly_result, poly_err) = await asyncio.gather(kalshi_task, poly_task)
    kalshi_raw = kalshi_result if isinstance(kalshi_result, list) else []
    polymarket_raw = poly_result if isinstance(poly_result, list) else []

    if kalshi_err:
        log_trace("api", f"Kalshi fetch issue: {kalshi_err}", {"error": kalshi_err}, level="warning")
    if poly_err:
        log_trace("api", f"Polymarket fetch issue: {poly_err}", {"error": poly_err}, level="warning")

    kalshi_markets = kalshi_norm.normalize(kalshi_raw)[:100]
    polymarket_markets = polymarket_norm.normalize(polymarket_raw)[:100]
    await save_markets(kalshi_markets + polymarket_markets)

    matches = find_matches(kalshi_markets, polymarket_markets)
    await save_matches(matches)

    matched_ids_a = {m.market_a.id for m in matches}
    matched_ids_b = {m.market_b.id for m in matches}
    unmatched_kalshi = [m for m in kalshi_markets if m.id not in matched_ids_a]
    unmatched_polymarket = [m for m in polymarket_markets if m.id not in matched_ids_b]

    out = {
        "query": query,
        "total_kalshi": len(kalshi_markets),
        "total_polymarket": len(polymarket_markets),
        "total_matches": len(matches),
        "matches": [m.model_dump(mode="json") for m in matches],
        "unmatched_kalshi": [m.model_dump(mode="json") for m in unmatched_kalshi],
        "unmatched_polymarket": [m.model_dump(mode="json") for m in unmatched_polymarket],
        "kalshi_error": kalshi_err,
        "polymarket_error": poly_err,
    }
    _cache_set(_search_cache, cache_key, {k: v for k, v in out.items() if k not in ("kalshi_error", "polymarket_error")})
    return out


def build_route_export(decision: dict, query: str, order_size: float) -> dict:
    """Build the public API export shape: winner, loser, estimated_savings, etc."""
    from datetime import datetime, timezone
    winner = decision.get("selected_market") or {}
    loser = decision.get("runner_up_market")
    return {
        "query": query,
        "order_size": order_size,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recommended_venue": decision.get("selected_venue", ""),
        "confidence": decision.get("score", 0),
        "winner": {
            "venue": winner.get("venue"),
            "yes_price": winner.get("yes_price"),
            "liquidity": winner.get("liquidity"),
            "fee_rate": winner.get("fee_rate"),
            "fee_model": winner.get("fee_model"),
            "all_in_cost": decision.get("winner_all_in_cost"),
            "execution_quality": decision.get("winner_execution_quality"),
        },
        "loser": {
            "venue": loser.get("venue"),
            "yes_price": loser.get("yes_price"),
            "liquidity": loser.get("liquidity"),
            "fee_rate": loser.get("fee_rate"),
            "fee_model": loser.get("fee_model"),
            "all_in_cost": decision.get("loser_all_in_cost"),
            "execution_quality": decision.get("loser_execution_quality"),
        } if loser else None,
        "estimated_savings": decision.get("estimated_savings"),
        "estimated_savings_text": decision.get("estimated_savings_text"),
    }


async def get_route_decision(query: str, order_size: float) -> dict:
    """Fetch markets, route, and return decision dict. Uses cache."""
    cache_key = f"route:{query.strip().lower()}_{order_size}"
    cached = _cache_get(_route_cache, cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    kalshi_task = _fetch_with_timeout(kalshi.fetch_markets(query, page_size=100), "Kalshi")
    poly_task = _fetch_with_timeout(polymarket.fetch_markets(query, limit=100), "Polymarket")
    (kalshi_result, kalshi_err), (poly_result, poly_err) = await asyncio.gather(kalshi_task, poly_task)
    kalshi_raw = kalshi_result if isinstance(kalshi_result, list) else []
    polymarket_raw = poly_result if isinstance(poly_result, list) else []

    if kalshi_err:
        log_trace("api", f"Kalshi fetch issue: {kalshi_err}", {"error": kalshi_err}, level="warning")
    if poly_err:
        log_trace("api", f"Polymarket fetch issue: {poly_err}", {"error": poly_err}, level="warning")

    kalshi_markets = kalshi_norm.normalize(kalshi_raw)[:100]
    polymarket_markets = polymarket_norm.normalize(polymarket_raw)[:100]
    all_markets = kalshi_markets + polymarket_markets
    matches = find_matches(kalshi_markets, polymarket_markets)

    decision = route(matches, order_size=order_size, all_markets=all_markets)
    await save_decision(decision)
    out = decision.model_dump(mode="json")
    _cache_set(_route_cache, cache_key, out)
    return out


@router.get("/route")
async def route_endpoint(
    query: str = Query(default="", description="Search term"),
    order_size: float = Query(100.0, ge=0),
):
    """Route to best venue for a hypothetical order. Cached 5 min. 8s timeout per venue."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query parameter is required")
    try:
        out = await get_route_decision(query, order_size)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"No markets found for query '{query}' on any venue. "
            "Try a broader search term.",
        )
    return out


@router.post("/route-by-match")
async def route_by_match(body: RouteByMatchBody):
    """Run routing for a single match pair. Used when user clicks a match row."""
    try:
        market_a = Market.model_validate(body.market_a)
        market_b = Market.model_validate(body.market_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid market data: {e}")
    match = MatchResult(market_a=market_a, market_b=market_b, score=1.0, method="user_selected", explanation="User selected this pair")
    try:
        decision = route([match], order_size=body.order_size, all_markets=[market_a, market_b])
    except ValueError:
        raise HTTPException(status_code=404, detail="Could not produce routing for this pair.")
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
