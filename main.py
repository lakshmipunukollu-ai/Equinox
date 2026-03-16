"""Project Equinox — FastAPI entry point."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from equinox.api import router as equinox_router, build_route_export, get_route_decision

FRONTEND_DIR = Path(__file__).parent / "frontend"
from equinox.matcher.engine import load_semantic_model
from equinox.store.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
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
