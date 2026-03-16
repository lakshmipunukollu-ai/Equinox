# Project Equinox — 2-Minute Demo Script

**Total: ~2 minutes** | Adjust pace as needed; timings are approximate.

---

## [0:00–0:20] **Intro & problem**

> "This is **Project Equinox** — a cross-venue prediction market aggregation and routing prototype.
>
> If you trade on prediction markets, you might use **Kalshi** and **Polymarket**. The problem: the same event often exists on both, with different prices, fees, and liquidity. There’s no single place that says *where* you should place your order for the best execution.
>
> Equinox does two things: it **finds equivalent markets** across venues, and it **routes you to the best venue** for your order size."

---

## [0:20–0:50] **Live demo — search and route**

> "I’ll type a query — say **bitcoin** — and an order size, like **100 dollars**. Then I click **Find Best Venue**.
>
> *(Click Find Best Venue.)*
>
> In the background we’re calling **Kalshi** and **Polymarket** in parallel, normalizing their responses into a common schema, and matching equivalent markets using a **hybrid matcher**: fuzzy string matching plus a **semantic model** so we catch things like ‘BTC above 100k’ vs ‘Bitcoin above $100,000.’ We persist those matches and the routing decision in SQLite.
>
> The UI shows: **how many markets** we found on each venue, **how many matched pairs**, and the **routing result** — which venue won, the confidence score, and **estimated savings** versus the other venue. You can expand a match row to see a side‑by‑side comparison and the reasoning."

---

## [0:50–1:25] **What the routing engine does**

> "The **routing engine** is venue-agnostic. It scores each market on **liquidity** (relative to your order size), **volume**, **fees**, and **execution quality** — including spread and all‑in cost. It picks a winner and a runner‑up and computes **all‑in cost per share** and **estimated savings** in dollars for your order size.
>
> So you get a clear recommendation: *use Kalshi* or *use Polymarket*, with a short explanation and, when we have a runner‑up, how much you’d save. We also expose a **public API**: `GET /api/route?q=bitcoin&size=100` returns that same routing decision as JSON — winner, loser, confidence, and estimated savings — so other tools or bots can consume it."

---

## [1:25–2:00] **Wrap-up**

> "Under the hood we have **FastAPI** for the backend, **sentence-transformers** for semantic matching, **SQLite** for markets and routing decisions, and a simple **single-page frontend** that talks to our search and route endpoints. There’s a debug endpoint for Kalshi if results look off, and the app can be deployed to **Railway** with the existing Procfile.
>
> So in short: one query, two venues, normalized and matched, with a routing recommendation and estimated savings. That’s Equinox."

---

## Stack

| Layer | Tech |
|-------|------|
| **Backend** | FastAPI, Uvicorn |
| **HTTP client** | httpx (async) |
| **Models / validation** | Pydantic ≥2 |
| **Matching** | rapidfuzz (fuzzy), sentence-transformers (all-MiniLM-L6-v2), scikit-learn |
| **Database** | SQLite via aiosqlite |
| **Auth (Kalshi)** | cryptography (API key + private key) |
| **Config** | python-dotenv, python-dateutil |
| **Frontend** | Vanilla HTML/CSS/JS (single page, no framework) |
| **Deploy** | Railway (Procfile, `requirements.txt`) |
| **Tests** | pytest, pytest-asyncio, respx |

---

## Quick reference — what to show

| Step | Action |
|------|--------|
| 1 | Open app (local or deployed). Point out search box and order size. |
| 2 | Enter e.g. `bitcoin`, size `100`, click **Find Best Venue**. |
| 3 | Highlight: Kalshi count, Polymarket count, match count, **recommended venue**, confidence, **estimated savings**. |
| 4 | (Optional) Expand a match row to show comparison and reasoning. |
| 5 | (Optional) Show or paste: `GET /api/route?q=bitcoin&size=100` and the JSON response. |

---

## Fallback if APIs are slow or down

> "If Kalshi or Polymarket is slow, we might see timeouts or zeros — we use an 8‑second timeout per venue and still return what we got. You can use the debug endpoint to see raw Kalshi responses. The routing logic and UI work the same; we’re just showing live data from both venues."
