"""
Single-responsibility: detect equivalent markets across venues using a
3-level hybrid approach. Imports only from equinox.models and equinox.utils.
Never imports from equinox.venues or equinox.normalizer.
"""

import itertools
import os
import re
from datetime import timedelta

from rapidfuzz import fuzz

from equinox.logger import log_trace
from equinox.models import Market, MatchResult
from equinox.utils import _to_ascii

ABBREVIATIONS = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "usd": "dollars",
    "pct": "percent",
    "us": "united states",
    "80k": "80000",
    # NBA — safe multi-word expansions only (single-word cities excluded to
    # avoid corrupting non-NBA markets like "Boston Fed", "Dallas Fed survey")
    "oklahoma city": "oklahoma city thunder",
    "golden state": "golden state warriors",
    "san antonio": "san antonio spurs",
    "new orleans": "new orleans pelicans",
    # NBA generic term expansion
    "pro basketball": "nba",
    # Cross-venue synonym: Kalshi uses "Championship", Polymarket uses "Finals"
    "championship": "finals",
}

_ST_MODEL = None


def load_semantic_model() -> None:
    """Load sentence-transformers model. Called during FastAPI lifespan startup."""
    global _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        log_trace("match", "Semantic model loaded successfully.", {})
    except ImportError:
        log_trace(
            "match",
            "sentence-transformers not installed — Level 3 disabled.",
            {},
            level="warning",
        )


def _normalize_title(title: str) -> str:
    """Normalize title for comparison: ASCII, lowercase, expand abbreviations."""
    t = _to_ascii(title).lower()
    # Strip punctuation, keep digits and spaces
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in t)
    tokens = cleaned.split()
    expanded = [ABBREVIATIONS.get(t, t) for t in tokens]
    result = " ".join(expanded)
    # Multi-word expansion pass (longest keys first to avoid partial overlaps)
    for key in sorted(
        (k for k in ABBREVIATIONS if len(k.split()) > 1), key=len, reverse=True
    ):
        value = ABBREVIATIONS[key]
        suffix = value[len(key) :].strip()
        if suffix:
            pattern = (
                r"\b" + re.escape(key) + r"(?!\s+" + re.escape(suffix) + r")"
            )
        else:
            pattern = r"\b" + re.escape(key) + r"\b"
        result = re.sub(pattern, value, result)
    return result


def find_matches(
    markets_a: list[Market],
    markets_b: list[Market],
    fuzzy_threshold: float | None = None,
    semantic_threshold: float | None = None,
    date_window_days: int | None = None,
) -> list[MatchResult]:
    """Find equivalent markets across two venue lists using 3-level hybrid matching."""
    if fuzzy_threshold is None:
        fuzzy_threshold = float(os.getenv("FUZZY_THRESHOLD", "0.75"))
    if semantic_threshold is None:
        semantic_threshold = float(os.getenv("SEMANTIC_THRESHOLD", "0.82"))
    if date_window_days is None:
        date_window_days = int(os.getenv("DATE_WINDOW_DAYS", "3"))

    results: list[MatchResult] = []
    unmatched_pairs: list[tuple[Market, Market]] = []

    for a, b in itertools.product(markets_a, markets_b):
        norm_a = _normalize_title(a.title)
        norm_b = _normalize_title(b.title)
        delta = abs(a.close_time - b.close_time)

        # Level 1 — Exact
        if norm_a == norm_b and delta <= timedelta(days=1):
            explanation = (
                f"Exact title match after normalization — both markets describe "
                f"'{norm_a}'. Expiry dates are {delta.days} day(s) apart, "
                "within the 1-day exact-match window."
            )
            log_trace("match", explanation, {"a": a.id, "b": b.id, "method": "exact"})
            results.append(
                MatchResult(
                    score=1.0,
                    method="exact",
                    market_a=a,
                    market_b=b,
                    explanation=explanation,
                )
            )
            continue

        # Level 2 — Fuzzy
        ratio = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
        date_ok = delta <= timedelta(days=date_window_days)
        if ratio >= fuzzy_threshold and date_ok:
            explanation = (
                f"Fuzzy title similarity of {ratio:.0%} — '{a.title}' and "
                f"'{b.title}' share the same underlying event. Expiry dates "
                f"are {delta.days} day(s) apart, within the {date_window_days}-day window."
            )
            log_trace(
                "match",
                explanation,
                {"a": a.id, "b": b.id, "score": ratio, "method": "fuzzy"},
            )
            results.append(
                MatchResult(
                    score=ratio,
                    method="fuzzy",
                    market_a=a,
                    market_b=b,
                    explanation=explanation,
                )
            )
            continue
        else:
            reason = (
                f"fuzzy score {ratio:.0%} below {fuzzy_threshold:.0%} threshold"
                if ratio < fuzzy_threshold
                else f"expiry dates {delta.days} days apart exceeds {date_window_days}-day window"
            )
            log_trace(
                "match",
                f"Rejected pair — {reason}. '{a.title}' vs '{b.title}'.",
                {"a": a.id, "b": b.id, "score": ratio, "date_ok": date_ok},
            )
            unmatched_pairs.append((a, b))

    # Level 3 — Semantic
    if _ST_MODEL is None and unmatched_pairs:
        log_trace(
            "match",
            f"Level 3 semantic matching skipped — sentence-transformers not installed. "
            f"{len(unmatched_pairs)} unmatched pair(s) were not evaluated semantically. "
            "Install sentence-transformers to enable deeper matching.",
            {"unmatched": len(unmatched_pairs)},
            level="warning",
        )
    elif _ST_MODEL is not None and unmatched_pairs:
        from sklearn.metrics.pairwise import cosine_similarity

        unmatched_a_ids = {a.id for a, _ in unmatched_pairs}
        unmatched_b_ids = {b.id for _, b in unmatched_pairs}
        embeddings_a = {
            a.id: _ST_MODEL.encode(_normalize_title(a.title))
            for a in markets_a
            if a.id in unmatched_a_ids
        }
        embeddings_b = {
            b.id: _ST_MODEL.encode(_normalize_title(b.title))
            for b in markets_b
            if b.id in unmatched_b_ids
        }

        for a, b in unmatched_pairs:
            emb_a = embeddings_a[a.id]
            emb_b = embeddings_b[b.id]
            sim = float(cosine_similarity([emb_a], [emb_b])[0][0])
            if sim >= semantic_threshold:
                explanation = (
                    f"Semantic similarity of {sim:.0%} — despite different wording, "
                    f"'{a.title}' and '{b.title}' describe the same underlying event."
                )
                log_trace(
                    "match",
                    explanation,
                    {"a": a.id, "b": b.id, "score": sim, "method": "semantic"},
                )
                results.append(
                    MatchResult(
                        score=sim,
                        method="semantic",
                        market_a=a,
                        market_b=b,
                        explanation=explanation,
                    )
                )
            else:
                log_trace(
                    "match",
                    f"Semantic similarity of {sim:.0%} is below the {semantic_threshold:.0%} "
                    f"threshold — '{a.title}' and '{b.title}' are not equivalent.",
                    {"a": a.id, "b": b.id, "score": sim},
                )

    log_trace(
        "match",
        f"Matching complete — evaluated {len(markets_a) * len(markets_b)} pairs, "
        f"found {len(results)} equivalent market(s).",
        {"total_pairs": len(markets_a) * len(markets_b), "matches_found": len(results)},
    )

    # ── Deduplicate to one-to-one: keep highest-scoring pair per market id ──
    seen: dict[str, float] = {}  # market_id → best score claimed
    deduped: list[MatchResult] = []
    for match in sorted(results, key=lambda m: m.score, reverse=True):
        id_a = match.market_a.id
        id_b = match.market_b.id
        if id_a in seen or id_b in seen:
            continue
        seen[id_a] = match.score
        seen[id_b] = match.score
        deduped.append(match)
    if len(deduped) < len(results):
        log_trace(
            "match",
            f"Deduplicated {len(results)} raw matches to {len(deduped)} "
            f"one-to-one pairs. {len(results) - len(deduped)} lower-scoring "
            f"duplicate pairs dropped.",
            {"raw": len(results), "deduped": len(deduped)},
        )
    results = deduped

    return results
