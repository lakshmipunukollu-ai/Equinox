"""
Single-responsibility: fetch raw market data from the Kalshi V1 API.
Handles optional RSA auth — attempts unauthenticated first, falls back to
signed request if 401 received and credentials are configured.
Returns raw dicts only — no transformation.
"""

import asyncio
import base64
import os
import time
from pathlib import Path

import httpx

from equinox.logger import log_trace

KALSHI_BASE = "https://api.elections.kalshi.com"
SEARCH_PATH = "/v1/search/series"


def _build_signed_headers(method: str, path: str) -> dict[str, str] | None:
    """Build RSA-signed headers. Returns None if credentials missing or key load fails."""
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not api_key_id or not pem_path or not Path(pem_path).exists():
        return None
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        private_key = serialization.load_pem_private_key(
            Path(pem_path).read_bytes(), password=None
        )
    except Exception as e:
        log_trace(
            "fetch",
            f"Failed to load Kalshi private key from {pem_path}: {e}. "
            "RSA signing disabled.",
            {"error": str(e)},
            level="warning",
        )
        return None

    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}{method}{path}"
    signature = private_key.sign(
        message.encode(), padding.PKCS1v15(), hashes.SHA256()
    )
    sig_b64 = base64.b64encode(signature).decode()
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
    }


async def fetch_markets(query: str, page_size: int = 50) -> list[dict]:
    """Fetch raw Kalshi series from V1 search endpoint."""
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    url = f"{KALSHI_BASE}{SEARCH_PATH}"
    params = {
        "query": query,
        "page_size": page_size,
        "order_by": "volume",
        "status": "open",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # First attempt: no auth
            resp = await client.get(url, params=params)
            if resp.status_code == 401 and api_key_id and pem_path:
                log_trace(
                    "fetch",
                    "Kalshi returned 401 — retrying with RSA signed headers. "
                    "Ensure KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH are set correctly "
                    "in .env if this persists.",
                    {"query": query},
                    level="warning",
                )
                headers = _build_signed_headers("GET", SEARCH_PATH)
                if headers:
                    resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 401:
                    log_trace(
                        "fetch",
                        "Kalshi authentication failed after signing attempt. "
                        "Check API credentials in .env.",
                        {"query": query},
                        level="warning",
                    )
                    return []

            if resp.status_code == 429:
                for n, wait in enumerate([1, 2, 4], 1):
                    log_trace(
                        "fetch",
                        f"Kalshi rate limit hit on attempt {n} — waiting {wait}s before retry.",
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
                    f"Kalshi HTTP error {resp.status_code} for query '{query}'.",
                    {"status": resp.status_code, "query": query},
                    level="warning",
                )
                return []

            try:
                body = resp.json()
            except Exception as e:
                log_trace(
                    "fetch",
                    f"Kalshi response not valid JSON: {e}",
                    {"query": query},
                    level="warning",
                )
                return []

            if "current_page" not in body:
                log_trace(
                    "fetch",
                    "Kalshi response missing expected 'current_page' key — "
                    "API response shape may have changed. Full response keys logged for debugging.",
                    {"keys": list(body.keys()), "query": query},
                    level="warning",
                )
                return []

            result = body["current_page"][:page_size]
            log_trace(
                "fetch",
                f"Retrieved {len(result)} Kalshi series for query '{query}' (capped at {page_size}).",
                {"count": len(result), "query": query},
            )
            return result
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        log_trace(
            "fetch",
            "Kalshi connection failed — API may be unreachable.",
            {"error": str(e), "query": query},
            level="error",
        )
        return []
