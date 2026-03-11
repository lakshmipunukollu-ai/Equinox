# Project Equinox — Assumptions and Methodology

## Why Kalshi V1 (not V2)

The Kalshi V2 public API silently ignores the `search` parameter — it accepts the query string but returns unfiltered results, making it useless for market discovery. V1 (`/v1/search/series`) powers the Kalshi frontend search bar and returns ranked results with nested market pricing included inline. V1 read endpoints require no authentication for public data, confirmed by direct inspection of Kalshi browser network traffic.

## Equivalence Matching Methodology

Markets are matched using a 3-level hybrid approach applied in order:

- **Level 1 (Exact)**: normalized title equality + expiry within 1 day. Score 1.0. Handles cases where both venues describe the market identically after cleanup.
- **Level 2 (Fuzzy)**: rapidfuzz token_sort_ratio >= 0.75 + expiry within 3 days. Handles word-order differences and minor wording variations between venues.
- **Level 3 (Semantic)**: sentence-transformers cosine similarity >= 0.82. Optional fallback for semantically equivalent but differently worded titles. Runs on each (a, b) pair that individually failed Levels 1 and 2 — does NOT require zero global matches. If 3 pairs match fuzzy, the remaining unmatched pairs still receive semantic evaluation. An implementation that gates Level 3 on "no results yet" is wrong and will silently drop valid matches.

Category is intentionally NOT used as a hard filter because Kalshi and Polymarket use inconsistent category names for the same events.

## Matching Threshold Justification

Fuzzy threshold 0.75: conservative by design to minimize false positives. A lower threshold risks matching unrelated markets (e.g. "Bitcoin" vs "Bitcoin Cash"). The 3-day date window accommodates minor expiry differences that still represent the same underlying event.

## Routing Scoring Formula

Four-component weighted formula (weights must match router/engine.py exactly):

```
final_score = (liquidity_score * 0.4) + (volume_score * 0.2) +
              (fee_score * 0.2) + (execution_score * 0.2)
final_score *= source_weight  (penalty for low-reliability price sources)
```

Component definitions:

- **liquidity_score (0.4)**: order-size-aware. `1.0 - (order_size / liquidity)`. A $100 order against $200 liquidity scores 0.5. Scores 0.0 if untradeable.
- **volume_score (0.2)**: market health signal only (not order-size-aware). `min(volume / VOLUME_CAP, 1.0)`. See volume/liquidity inconsistency note below.
- **fee_score (0.2)**: measures fee FRICTION only, not event probability. `fee_cost = all_in_price - yes_price` (venue-model-aware, see fee section). `fee_score = max(0.0, 1.0 - fee_cost / FEE_CAP)`. Anchors on the fee burden in ¢, NOT the total price.
- **execution_score (0.2)**: spread width when available; conviction proxy fallback. Rewards tight bid-ask spreads (lower execution cost).

NOTE: volume_score and liquidity_score are on different scales. volume_score uses an absolute cap (VOLUME_CAP) and ignores order_size. liquidity_score is order-size-relative and correctly captures tradability. A market with $90K volume and $100 liquidity would get volume_score≈0.9 but liquidity_score=0.0 — it would be completely untradeable yet score well on volume. Volume is treated as a market health proxy, not an execution metric. This asymmetry is intentional and documented here.

## Price Semantics: Mid vs Bid vs Ask

yes_price on the canonical Market model is the fair-value mid: `(yes_bid + yes_ask) / 2` when both are present. Falls back to yes_ask (buyer's execution price), then yes_bid, then last_price. NEVER use yes_bid as "the price" for a buyer — yes_bid is what a seller receives. Systematically using bid understates what a buyer actually pays and makes one venue appear cheaper than it is in routing comparisons.

## Scoring Cap Calibration and Venue Bias Risk

LIQUIDITY_CAP was removed when the liquidity formula changed from `min(liquidity/CAP, 1.0)` to the order-size-aware `1.0 - (order_size/liquidity)`. The new formula has no cap — it is self-calibrating relative to order size.

VOLUME_CAP (default 100,000) determines when a market saturates to score=1.0 on the volume component. This default is NOT empirically calibrated against real Kalshi or Polymarket volume distributions.

**Known bias risk**: if typical Kalshi volume is 500K contracts/day while typical Polymarket volume is $40K/day, Kalshi saturates at 1.0 on volume score while Polymarket scores 0.4 — not because Kalshi is more active, but because the cap is not scaled to each venue's actual volume distribution. To eliminate this bias, VOLUME_CAP should be set empirically by querying the p50/p90 volume distribution across each venue's active markets. Until then, treat cross-venue volume score comparisons as directional signals only. VOLUME_CAP is configurable in .env — adjust before drawing conclusions.

## Data Assumptions and Defaults

- **yes_price**: mid when both bid/ask present → ask → bid → last_price → default 0.5. Uses explicit `is not None` checks — not Python `or` truthiness — because 0 is a valid price for a near-certain NO market and would be silently skipped by an `or` chain.
- **volume**: defaults to 0.0 when absent
- **liquidity**: falls back to volume when absent (Kalshi V1 has no separate field)
- **category**: defaults to "unknown" when absent
- **close_time**: defaults to datetime.now(UTC) when absent or unparseable — this means date-proximity filtering may behave incorrectly for affected markets

## Kalshi Auth Behavior

Kalshi V1 read endpoints were observed to not require authentication based on direct inspection of browser network traffic. The adapter attempts the first request unauthenticated. If a 401 is received and credentials are configured in .env, it retries with RSA-signed headers. The auth mode that succeeded is logged at INFO level on every successful fetch so the operator can confirm which path is active. If no credentials are configured and a 401 is received, the adapter logs a WARNING and returns an empty list for that query.

## Volume Unit Incompatibility

Kalshi volume is denominated in contracts (number of shares traded), not USD. Polymarket volume is denominated in USD. These units are not directly comparable. The routing engine normalizes both to a 0–1 score using separate fixed denominators (Kalshi: 100,000 contracts, Polymarket: $100,000 USD). This normalization is approximate — a Kalshi contract value varies by market price. Cross-venue volume score comparisons should be treated as directional signals only, not precise equivalents.

## Polymarket Gamma API Topic Filtering

The Polymarket Gamma API's q= parameter does not filter results by topic — it returns broadly popular markets regardless of the query string. Without client-side filtering, the matcher receives unrelated markets from Polymarket and cross-venue matching never fires for topical queries.

Fix applied: fetch_markets() filters the raw API response to markets where the query string appears as a case-insensitive substring in the "question" field before returning results. The original_count and filtered_count are both logged for observability.

Known limitation of this approach: multi-word queries ("fed rate") won't match "Federal Reserve interest rate decision", and abbreviated queries ("btc") won't match "bitcoin" since the ABBREVIATIONS expansion in the matcher operates at match time, not fetch time. For the prototype's primary test queries (bitcoin, trump, fed rate as full words), this filter is sufficient. If abbreviated queries become a use case, expand the query before calling fetch_markets().

## Fee Rate Documentation

Fees are a first-class input to the routing formula. Without fees, a market at YES=0.44 with 7% taker fee costs the same all-in as YES=0.471 at 0% fee — the system would route to the wrong venue on headline price alone.

**Kalshi fee rate**: approximately 7¢ per contract on taker orders (KALSHI_FEE_RATE=0.07). This is an approximation based on publicly available pricing. The Kalshi V1 API does not expose per-market fee data in the search response. If Kalshi publishes fee data programmatically, the normalizer should read it from the market response instead of the environment variable.

**Polymarket fee rate**: approximately 2% protocol fee on taker orders (POLYMARKET_FEE_RATE=0.02). Similarly hardcoded — the Gamma API does not expose fee_rate per market. Update if Polymarket publishes this data.

Fee application formulas differ by venue and fee_model field:

- **Kalshi (fee_model="additive")**: `all_in_price = yes_price + fee_rate`
- **Polymarket (fee_model="multiplicative")**: `all_in_price = yes_price * (1 + fee_rate)`

NEVER apply the multiplicative formula to Kalshi — it is dimensionally wrong. The routing fee_score measures fee friction only: `fee_cost = all_in_price - yes_price`.

## Binary Market Policy

This system only handles binary YES/NO prediction markets. For binary markets: yes_price + no_price = 1.0 (by definition). yes_price represents the probability of the YES outcome. no_price = 1.0 - yes_price is meaningful.

Multi-outcome markets (e.g. "Who wins the 2024 election?" with 5 candidates) are out of scope. For those markets: no_price = 1.0 - yes_price is meaningless (other outcomes exist); yes_price from bestAsk is a single-leg price, not event probability; cross-venue comparison is undefined without normalizing across all legs.

Polymarket multi-outcome markets are detected by checking the 'outcomes' field. If len(outcomes) != 2, the market is skipped during normalization with a WARNING. Kalshi V1 /search/series only returns binary YES/NO contracts — no detection needed.

## AI Usage Disclosure

sentence-transformers (all-MiniLM-L6-v2, 90MB, runs locally) is used as an optional Level 3 equivalence signal. It computes semantic similarity between market titles using dense vector embeddings. It is disabled automatically if the library is not installed — the system degrades to Levels 1 and 2 only. The model is loaded asynchronously during FastAPI lifespan startup via asyncio.to_thread() to avoid blocking the event loop. First startup may take 30–60 seconds while the model downloads. This is the only AI component in the system.
