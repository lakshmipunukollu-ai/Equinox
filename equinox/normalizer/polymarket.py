"""
Single-responsibility: map raw Polymarket Gamma API dicts into canonical Market
objects. bestAsk is unreliable on illiquid markets — fallback chain documented.
Never raises — uses documented defaults for all missing fields.
"""

import json as _json
import os

from equinox.logger import log_trace
from equinox.models import Market
from equinox.utils import _to_ascii, parse_utc_datetime


def _safe_float(val: str | int | float | None, default: float | None) -> float | None:
    """Safely parse to float. Returns default for None, empty string, or parse failure."""
    if val is None:
        return default
    s = str(val).strip()
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def normalize(markets_list: list[dict]) -> list[Market]:
    """Convert Polymarket list to canonical Market list."""
    results: list[Market] = []
    for raw in markets_list:
        id_ = f"polymarket:{raw.get('id', 'unknown')}"
        venue = "polymarket"
        title = _to_ascii(raw.get("question", ""))
        category = _to_ascii(raw.get("category", "unknown").lower())

        yes_bid = _safe_float(raw.get("bestBid"), None)
        yes_ask = _safe_float(raw.get("bestAsk"), None)
        last = _safe_float(raw.get("lastTradePrice"), None)

        # Polymarket: mid when both; else bestAsk (buyer price); else lastTradePrice; else bid; else 0.5.
        # Skip bid when ask is missing — bestAsk unreliable on illiquid markets.
        if yes_bid is not None and yes_ask is not None:
            yes_price = round((yes_bid + yes_ask) / 2.0, 6)
        elif yes_ask is not None:
            yes_price = yes_ask
        elif last is not None:
            yes_price = last
        elif yes_bid is not None:
            yes_price = yes_bid
        else:
            yes_price = 0.5

        spread_width = (
            round(yes_ask - yes_bid, 6)
            if yes_bid is not None and yes_ask is not None
            else None
        )

        price_updated_at = (
            parse_utc_datetime(raw.get("updatedAt"), "updatedAt")
            if raw.get("updatedAt")
            else None
        )

        if yes_bid is not None and yes_ask is not None:
            price_source = "mid"
        elif yes_ask is not None:
            price_source = "ask"
        elif last is not None:
            price_source = "last"
        elif yes_bid is not None:
            price_source = "bid"
        else:
            price_source = "default"

        fee_rate = float(os.getenv("POLYMARKET_FEE_RATE", "0.02"))
        fee_model = "multiplicative"

        outcomes = raw.get("outcomes", [])
        if isinstance(outcomes, str):
            try:
                outcomes = _json.loads(outcomes)
            except Exception:
                outcomes = []
        is_binary = len(outcomes) == 2 or not outcomes

        if not is_binary:
            log_trace(
                "normalize",
                f"Skipping Polymarket market '{title}' — detected {len(outcomes)} "
                "outcomes (not binary). Multi-outcome markets are out of scope.",
                {"id": id_, "outcome_count": len(outcomes)},
                level="warning",
            )
            continue

        no_price = round(1.0 - yes_price, 6)
        volume = _safe_float(raw.get("volume"), 0.0) or 0.0
        # Liquidity: try multiple API field names (Gamma API structure can vary)
        liquidity = (
            _safe_float(raw.get("liquidity"), None)
            or _safe_float(raw.get("volume"), None)
            or _safe_float(raw.get("volumeNum"), None)
            or _safe_float(raw.get("liquidityNum"), None)
            or _safe_float(raw.get("volumeNumber"), None)
            or _safe_float(raw.get("depth"), None)
        )
        if liquidity is None or liquidity < 0:
            liquidity = volume if volume else 0.0
        if liquidity == 0 and volume > 0:
            liquidity = volume  # use volume as liquidity proxy when liquidity field missing
        close_time = parse_utc_datetime(raw.get("endDateIso"), "endDateIso")
        url = ""  # Platform links removed until API URL fields confirmed working

        log_trace(
            "normalize",
            f"Normalized Polymarket market '{title}' with yes_price={yes_price:.4f} "
            f"({price_source}), bid={yes_bid}, ask={yes_ask}, "
            f"spread_width={spread_width}, fee_rate={fee_rate}.",
            {
                "id": id_,
                "yes_price": yes_price,
                "price_source": price_source,
                "spread_width": spread_width,
                "fee_rate": fee_rate,
            },
        )

        results.append(
            Market(
                id=id_,
                venue=venue,
                title=title,
                category=category,
                yes_price=yes_price,
                no_price=no_price,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                spread_width=spread_width,
                volume=volume,
                liquidity=liquidity,
                close_time=close_time,
                price_updated_at=price_updated_at,
                price_source=price_source,
                fee_rate=fee_rate,
                fee_model=fee_model,
                is_binary=is_binary,
                url=url,
                raw=raw,
            )
        )
    return results
