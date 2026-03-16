"""
Single-responsibility: define all data structures that cross layer boundaries.
No raw dicts may be passed between layers — only these models.
No business logic here.
"""

from datetime import datetime

from pydantic import BaseModel, model_validator


class Market(BaseModel):
    id: str
    venue: str
    title: str
    category: str
    yes_price: float
    no_price: float
    yes_bid: float | None
    yes_ask: float | None
    spread_width: float | None
    volume: float
    liquidity: float
    close_time: datetime
    price_updated_at: datetime | None
    price_source: str
    fee_rate: float
    fee_model: str
    is_binary: bool
    url: str
    raw: dict


class MatchResult(BaseModel):
    market_a: Market
    market_b: Market
    score: float
    method: str
    explanation: str

    @model_validator(mode="after")
    def venues_must_differ(self) -> "MatchResult":
        if self.market_a.venue == self.market_b.venue:
            raise ValueError(
                f"MatchResult requires markets from different venues — "
                f"got '{self.market_a.venue}' for both market_a ({self.market_a.id}) "
                f"and market_b ({self.market_b.id}). A market cannot match itself."
            )
        return self


class RoutingDecision(BaseModel):
    selected_venue: str
    selected_market: Market
    score: float
    reasoning: str
    alternatives: list[dict]
    price_divergence: float | None
    simulation: bool = True
    runner_up_market: Market | None = None  # loser for plain-English summary and savings
    # Fee/savings: all-in cost and estimated savings (winner is always cheaper)
    winner_all_in_cost: float | None = None
    loser_all_in_cost: float | None = None
    estimated_savings: float | None = None
    estimated_savings_text: str | None = None  # Plain text e.g. "Save ~$5.00" for UI (no HTML)
    # Execution quality letter grade (A+, A, B+, etc.)
    winner_execution_quality: str | None = None
    loser_execution_quality: str | None = None
    # Confidence breakdown for UI (0-1 each)
    confidence_breakdown: dict | None = None
