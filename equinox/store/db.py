"""
Single-responsibility: persist normalized markets, match results, and routing
decisions to a local SQLite database for post-run inspection and audit.
Uses aiosqlite for async compatibility with FastAPI.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import aiosqlite

from equinox.logger import log_trace
from equinox.models import Market, MatchResult, RoutingDecision


async def _get_conn():
    """Open connection. Reads DB_PATH at call time for test override."""
    path = os.getenv("DB_PATH", "equinox.db")
    return aiosqlite.connect(path)


async def init_db() -> None:
    """Create tables and enable WAL mode."""
    async with await _get_conn() as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id TEXT PRIMARY KEY,
                venue TEXT,
                title TEXT,
                category TEXT,
                yes_price REAL,
                volume REAL,
                liquidity REAL,
                close_time TEXT,
                url TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS match_results (
                id TEXT PRIMARY KEY,
                market_a_id TEXT,
                market_b_id TEXT,
                score REAL,
                method TEXT,
                explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS routing_decisions (
                id TEXT PRIMARY KEY,
                selected_market_id TEXT,
                selected_venue TEXT,
                score REAL,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
    log_trace(
        "store",
        "SQLite database initialized with WAL mode. Tables verified.",
        {"path": os.getenv("DB_PATH", "equinox.db")},
    )


async def save_markets(markets: list[Market]) -> None:
    """Upsert markets by id."""
    if not markets:
        return
    async with await _get_conn() as db:
        for m in markets:
            data = json.dumps(m.model_dump(mode="json"), default=str)
            await db.execute(
                """
                INSERT OR REPLACE INTO markets
                (id, venue, title, category, yes_price, volume, liquidity, close_time, url, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m.id,
                    m.venue,
                    m.title,
                    m.category,
                    m.yes_price,
                    m.volume,
                    m.liquidity,
                    m.close_time.isoformat(),
                    m.url,
                    data,
                ),
            )
        await db.commit()
    log_trace(
        "store",
        f"Saved {len(markets)} normalized markets to SQLite.",
        {"table": "markets", "count": len(markets)},
    )


async def save_matches(matches: list[MatchResult]) -> None:
    """Upsert match results."""
    if not matches:
        return
    async with await _get_conn() as db:
        for m in matches:
            mid = f"{m.market_a.id}::{m.market_b.id}"
            await db.execute(
                """
                INSERT OR REPLACE INTO match_results
                (id, market_a_id, market_b_id, score, method, explanation)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (mid, m.market_a.id, m.market_b.id, m.score, m.method, m.explanation),
            )
        await db.commit()
    log_trace(
        "store",
        f"Saved {len(matches)} match results to SQLite.",
        {"table": "match_results", "count": len(matches)},
    )


async def save_decision(decision: RoutingDecision) -> None:
    """Upsert routing decision with idempotent ID."""
    minute_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    raw = (
        f"{decision.selected_market.id}::{decision.score}::"
        f"{decision.selected_market.yes_price}::{minute_ts}"
    )
    decision_id = "route::" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    async with await _get_conn() as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO routing_decisions
            (id, selected_market_id, selected_venue, score, reasoning)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                decision.selected_market.id,
                decision.selected_venue,
                decision.score,
                decision.reasoning,
            ),
        )
        await db.commit()
    log_trace(
        "store",
        f"Saved routing decision for {decision.selected_venue} to SQLite.",
        {"table": "routing_decisions", "selected": decision.selected_market.id},
    )
