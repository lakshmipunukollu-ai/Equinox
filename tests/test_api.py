"""Integration tests for API endpoints."""

import os
import tempfile

import httpx
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from main import app
from tests.conftest import SAMPLE_KALSHI_SERIES, SAMPLE_POLYMARKET_LIST


@pytest.fixture
def throwaway_pem(tmp_path):
    """Generate a real RSA key for 401 retry test."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "test_key.pem"
    path.write_bytes(pem_bytes)
    yield str(path)


async def test_health_endpoint():
    """GET /api/equinox/health → 200, status ok."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/equinox/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


@respx.mock
async def test_kalshi_missing_current_page_key():
    """Kalshi returns body without current_page → total_kalshi=0, no crash."""
    respx.get("https://api.elections.kalshi.com/v1/search/series").mock(
        return_value=httpx.Response(200, json={"unexpected_key": []})
    )
    respx.get("https://gamma-api.polymarket.com/public-search").mock(
        return_value=httpx.Response(200, json={"events": []})
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/equinox/search?query=bitcoin")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("total_kalshi") == 0


@respx.mock
async def test_kalshi_401_triggers_retry_not_empty_return(throwaway_pem, monkeypatch):
    """401 on first call, 200 on retry with auth → non-empty result."""
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", throwaway_pem)

    respx.get("https://api.elections.kalshi.com/v1/search/series").mock(
        side_effect=[
            httpx.Response(401),
            httpx.Response(200, json={"current_page": SAMPLE_KALSHI_SERIES}),
        ]
    )
    respx.get("https://gamma-api.polymarket.com/public-search").mock(
        return_value=httpx.Response(200, json={"events": []})
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/equinox/search?query=bitcoin")

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("total_kalshi", 0) > 0
    assert len(respx.calls) >= 2


@respx.mock
async def test_search_returns_expected_shape():
    """Search returns matches, total_kalshi, total_polymarket, etc."""
    respx.get("https://api.elections.kalshi.com/v1/search/series").mock(
        return_value=httpx.Response(200, json={"current_page": SAMPLE_KALSHI_SERIES})
    )
    respx.get("https://gamma-api.polymarket.com/public-search").mock(
        return_value=httpx.Response(
            200,
            json={"events": [{"markets": SAMPLE_POLYMARKET_LIST}]},
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/equinox/search?query=bitcoin")
    assert resp.status_code == 200
    data = resp.json()
    assert "matches" in data
    assert "total_kalshi" in data
    assert "total_polymarket" in data
    assert "total_matches" in data


@respx.mock
async def test_route_returns_routing_decision():
    """Route returns selected_venue, reasoning."""
    respx.get("https://api.elections.kalshi.com/v1/search/series").mock(
        return_value=httpx.Response(200, json={"current_page": SAMPLE_KALSHI_SERIES})
    )
    respx.get("https://gamma-api.polymarket.com/public-search").mock(
        return_value=httpx.Response(
            200,
            json={"events": [{"markets": SAMPLE_POLYMARKET_LIST}]},
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/equinox/route?query=bitcoin")
    assert resp.status_code == 200
    data = resp.json()
    assert "selected_venue" in data
    assert data.get("reasoning")
    assert data.get("simulation") is True


@respx.mock
async def test_route_no_matches_returns_404():
    """Both venues empty → 404."""
    respx.get("https://api.elections.kalshi.com/v1/search/series").mock(
        return_value=httpx.Response(200, json={"current_page": []})
    )
    respx.get("https://gamma-api.polymarket.com/public-search").mock(
        return_value=httpx.Response(200, json={"events": []})
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/equinox/route?query=nonexistentxyz123")
    assert resp.status_code == 404
