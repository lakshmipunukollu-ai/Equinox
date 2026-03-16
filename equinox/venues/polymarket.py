"""
Single-responsibility: fetch raw market data from the Polymarket Gamma API.
Uses the public-search endpoint for query-based search. No authentication required.
Returns raw dicts only — no transformation.
Set POLYMARKET_API_BASE_URL to a proxy URL if the default is blocked or rate-limited (e.g. in cloud).
"""

import asyncio
import os

import httpx

from equinox.logger import log_trace

GAMMA_BASE = os.getenv("POLYMARKET_API_BASE_URL", "https://gamma-api.polymarket.com").rstrip("/")
SEARCH_PATH = "/public-search"


def _polymarket_timeout() -> float:
    """Timeout in seconds for Polymarket HTTP client. Use POLYMARKET_TIMEOUT_SECONDS (default 10)."""
    return min(float(os.getenv("POLYMARKET_TIMEOUT_SECONDS", "10")), 60.0)


async def fetch_markets(query: str, limit: int = 50) -> list[dict]:
    """Fetch raw Polymarket markets from Gamma API public-search endpoint."""
    url = f"{GAMMA_BASE}{SEARCH_PATH}"
    params = {"q": query, "limit_per_type": max(limit, 20)}
    timeout = _polymarket_timeout()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 429:
                for n, wait in enumerate([1, 2, 4], 1):
                    log_trace(
                        "fetch",
                        f"Polymarket rate limit hit on attempt {n} — waiting {wait}s.",
                        {"attempt": n, "query": query},
                        level="warning",
                    )
                    await asyncio.sleep(wait)
                    resp = await client.get(url, params=params)
                    if resp.status_code != 429:
                        break
                if resp.status_code == 429:
                    return []

            if resp.status_code >= 400:
                log_trace(
                    "fetch",
                    f"Polymarket HTTP error {resp.status_code} for query '{query}'.",
                    {"status": resp.status_code, "query": query},
                    level="warning",
                )
                return []

            body = resp.json()
            if not isinstance(body, dict) or "events" not in body:
                log_trace(
                    "fetch",
                    "Polymarket public-search response missing 'events' key.",
                    {"query": query},
                    level="warning",
                )
                return []

            events = body.get("events") or []
            result = []
            for ev in events:
                for m in ev.get("markets") or []:
                    result.append(m)
                    if len(result) >= limit:
                        break
                if len(result) >= limit:
                    break

            result = result[:limit]
            log_trace(
                "fetch",
                f"Retrieved {len(result)} Polymarket markets for query '{query}' (capped at {limit}).",
                {"count": len(result), "query": query},
            )
            return result
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        log_trace(
            "fetch",
            "Polymarket connection failed — API may be unreachable.",
            {"error": str(e), "query": query},
            level="error",
        )
        return []
