"""
Single-responsibility: map raw Kalshi V1 series dicts into canonical Market
objects. A Kalshi series contains multiple markets (one per outcome/strike).
Each nested market produces one canonical Market.
Never raises — uses documented defaults for all missing fields.
"""

import os

from equinox.logger import log_trace
from equinox.models import Market
from equinox.utils import _to_ascii, parse_utc_datetime


def _cents_to_decimal(v: int | float | None, ticker: str = "") -> float | None:
    """Convert Kalshi integer cents to decimal probability. Guards against decimal format."""
    if v is None:
        return None
    if isinstance(v, float) and 0.0 < v < 1.0:
        return v
    result = v / 100.0
    if result > 1.0:
        return 0.5
    return result


def normalize(series_list: list[dict]) -> list[Market]:
    """Convert Kalshi series list to canonical Market list."""
    results: list[Market] = []
    for series in series_list:
        for market in series.get("markets", []):
            ticker = market.get("ticker") or market.get("market_ticker") or "unknown"
            id_ = f"kalshi:{ticker}"
            venue = "kalshi"
            title_parts = []
            base = series.get("series_title") or series.get("event_title") or ""
            if base:
                title_parts.append(base)
            event_sub = series.get("event_subtitle", "")
            if event_sub:
                title_parts.append(event_sub)
            yes_sub = market.get("yes_subtitle", "")
            if yes_sub:
                title_parts.append(yes_sub)
            title = _to_ascii(" ".join(title_parts))
            category = _to_ascii(series.get("category", "unknown").lower())

            _bid_raw = market.get("yes_bid")
            _ask_raw = market.get("yes_ask")
            _last_raw = market.get("last_price")

            yes_bid = _cents_to_decimal(_bid_raw)
            yes_ask = _cents_to_decimal(_ask_raw)
            _last = _cents_to_decimal(_last_raw)

            for field_name, raw_v, converted in [
                ("yes_bid", _bid_raw, yes_bid),
                ("yes_ask", _ask_raw, yes_ask),
                ("last_price", _last_raw, _last),
            ]:
                if raw_v is None:
                    continue
                if isinstance(raw_v, float) and 0.0 < raw_v < 1.0:
                    log_trace(
                        "normalize",
                        f"Kalshi {field_name}={raw_v} appears to be decimal fraction "
                        f"for ticker '{ticker}' — expected integer cents. Using as-is.",
                        {"ticker": ticker, "field": field_name, "raw": raw_v},
                        level="warning",
                    )
                elif converted == 0.5 and raw_v is not None and raw_v / 100.0 > 1.0:
                    log_trace(
                        "normalize",
                        f"Kalshi {field_name}={raw_v} / 100 = {raw_v/100:.4f} > 1.0 "
                        f"for ticker '{ticker}' — clamped to 0.5.",
                        {"ticker": ticker, "field": field_name, "raw": raw_v},
                        level="warning",
                    )

            if yes_bid is not None and yes_ask is not None:
                yes_price = round((yes_bid + yes_ask) / 2.0, 6)
            elif yes_ask is not None:
                yes_price = yes_ask
            elif yes_bid is not None:
                yes_price = yes_bid
            elif _last is not None:
                yes_price = _last
            else:
                yes_price = 0.5
                log_trace(
                    "normalize",
                    f"Kalshi market '{ticker}' has no bid, ask, or last_price — "
                    "defaulting yes_price to 0.5.",
                    {"ticker": ticker},
                    level="warning",
                )

            spread_width = (
                round(yes_ask - yes_bid, 6)
                if yes_bid is not None and yes_ask is not None
                else None
            )

            price_updated_at = None

            if yes_bid is not None and yes_ask is not None:
                price_source = "mid"
            elif yes_ask is not None:
                price_source = "ask"
            elif yes_bid is not None:
                price_source = "bid"
            elif market.get("last_price") is not None:
                price_source = "last"
            else:
                price_source = "default"

            fee_rate = float(os.getenv("KALSHI_FEE_RATE", "0.07"))
            fee_model = "additive"
            is_binary = True

            no_price = round(1.0 - yes_price, 6)
            volume = float(market.get("volume", 0))
            liquidity = float(market.get("volume", 0))
            close_time = parse_utc_datetime(market.get("close_ts"), "close_ts")
            url = ""  # Platform links removed until API URL fields confirmed working

            log_trace(
                "normalize",
                f"Normalized Kalshi market '{title}' with yes_price={yes_price:.4f} "
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
                    raw=market,
                )
            )
    return results
