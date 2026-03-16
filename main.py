"""Project Equinox — FastAPI entry point."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from equinox.api import router as equinox_router, build_route_export, get_route_decision
from equinox.logger import log_trace

FRONTEND_DIR = Path(__file__).parent / "frontend"
from equinox.matcher.engine import load_semantic_model
from equinox.store.db import init_db


def _validate_env_startup():
    """Log warnings for missing optional env vars so production can be audited."""
    missing = []
    if not os.getenv("HF_TOKEN"):
        missing.append("HF_TOKEN (optional; improves model download rate limits)")
    if not os.getenv("KALSHI_API_KEY_ID") or not os.getenv("KALSHI_PRIVATE_KEY_PATH"):
        missing.append("KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH (optional; required if Kalshi returns 401)")
    if not os.getenv("POLYMARKET_API_BASE_URL"):
        missing.append("POLYMARKET_API_BASE_URL (optional; set to proxy URL if Polymarket blocks cloud IPs)")
    if not os.getenv("POLYMARKET_TIMEOUT_SECONDS"):
        missing.append("POLYMARKET_TIMEOUT_SECONDS (optional; default 8; use 15–20 in production if fetches time out)")
    if missing:
        log_trace(
            "startup",
            "Some optional env vars are unset. App will run with defaults; set in Railway Variables if needed: " + "; ".join(missing),
            {"missing": missing},
            level="warning",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_env_startup()
    await init_db()
    await asyncio.to_thread(load_semantic_model)
    yield


app = FastAPI(
    title="Project Equinox",
    description="Cross-venue prediction market aggregation and routing prototype",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(equinox_router, prefix="/api/equinox")


@app.get("/api/route", include_in_schema=True)
async def public_route(
    q: str = Query(..., description="Search query"),
    size: float = Query(100.0, ge=0, description="Order size in USD"),
):
    """Public API: returns routing decision as JSON (winner, loser, estimated_savings, etc.)."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    try:
        decision = await get_route_decision(q.strip(), size)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"No markets found for query '{q}' on any venue.",
        )
    export = build_route_export(decision, q.strip(), size)
    return export


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return {"error": "frontend not built"}
    return FileResponse(index)
