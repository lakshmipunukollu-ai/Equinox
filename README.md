# Project Equinox

Cross-venue prediction market aggregation and routing prototype.

## Prerequisites

- Python 3.11+
- pip

## Setup

```bash
git clone <repo>
cd Equinox
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment

```bash
cp .env.example .env
# Defaults work for local development with no Kalshi credentials
# Add KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH only if you have them
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

## Endpoints

- `GET /api/equinox/health`
- `GET /api/equinox/search?query=bitcoin`
- `GET /api/equinox/route?query=bitcoin&order_size=100`

## Debugging

`GET /api/equinox/debug/kalshi?q=bitcoin`

Returns the raw unmodified Kalshi V1 response for a query. Use this when /search or /route returns 0 Kalshi results to diagnose whether the issue is the URL, the query term, auth, or the response shape.

## Logs

- **Console**: WARNING and above — API errors, missing fields, degraded modes
- **File**: `logs/equinox.log` — full DEBUG trace of every decision
- View live: `tail -f logs/equinox.log`

## Database

All normalized markets and decisions saved to `equinox.db`. Inspect with:

```bash
sqlite3 equinox.db "SELECT * FROM routing_decisions"
```

## Run Tests

```bash
pytest tests/ -v
```
