import math
from datetime import datetime, timezone
from ..preprocessing.textutils import words
from ..preprocessing.claims import STOP
from ..schemas.schemas import Source
from .common import domain_of

# Publisher TYPE is a weak signal (max 0.15 of the score). A famous publisher is never "always true".
TYPE_WEIGHT = {"wire": 1.0, "factchecker": 1.0, "government": 0.9, "newspaper": 0.8, "broadcaster": 0.8}


def relevance(claim: str, src: Source) -> float:
    """Fraction of the claim's content words found in the source title+snippet.
    Lexical baseline; replace with multilingual sentence embeddings (e.g. an e5 model) for semantic similarity."""
    cw = {w for w in words(claim) if w not in STOP and len(w) > 2}
    if not cw:
        return 0.0
    sw = set(words(f"{src.title} {src.snippet or ''}"))
    return len(cw & sw) / len(cw)


def recency(dt) -> float:
    if dt is None:
        return 0.3
    age = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
    return math.exp(-age / 30)


def rank(claim: str, sources: list, min_relevance: float = 0.25, limit: int = 12, sem_fn=None) -> list:
    """Dedup, score, filter, sort. relevance = word overlap, blended 50/50 with embedding cosine when sem_fn is given."""
    seen, uniq = set(), []
    for s in sources:
        key = (domain_of(str(s.url)), " ".join(words(s.title))[:80])
        if key in seen:                      # drop exact syndicated duplicates
            continue
        seen.add(key)
        uniq.append(s)
    lex = [relevance(claim, s) for s in uniq]
    sims = None
    if sem_fn and uniq:
        try:
            sims = sem_fn(claim, [f"{s.title}. {s.snippet or ''}" for s in uniq])
        except Exception:
            sims = None                      # never let the embedder break a request
    ranked = []
    for i, s in enumerate(uniq):
        rel = lex[i] if sims is None else 0.5 * lex[i] + 0.5 * max(0.0, float(sims[i]))
        if rel < min_relevance and s.provenance != "factcheck":
            continue
        tw = TYPE_WEIGHT.get(s.source_type or "", 0.5)
        s.relevance = round(0.6 * rel + 0.25 * recency(s.published_at) + 0.15 * tw, 3)
        ranked.append(s)
    ranked.sort(key=lambda s: -s.relevance)
    return ranked[:limit]
