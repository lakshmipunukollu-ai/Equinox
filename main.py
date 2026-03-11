"""Project Equinox — FastAPI entry point."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from equinox.api import router as equinox_router

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


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return {"error": "frontend not built"}
    return FileResponse(index)
