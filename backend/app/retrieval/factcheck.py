import httpx
from ..schemas.schemas import EvidenceStatus, Source
from .common import parse_iso, safe_http_url, source_type_for

URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
PROVIDER = "google_factcheck"


async def search_factchecks(query: str, lang: str, api_key, client: httpx.AsyncClient,
                            timeout: float = 8, max_age_days: int = 365):
    if not api_key:
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error="API key not configured")
    try:
        r = await client.get(URL, timeout=timeout, params={
            "query": query, "languageCode": lang, "pageSize": 10,
            "maxAgeDays": max_age_days, "key": api_key})
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:      # includes timeouts
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error=type(e).__name__)
    out = []
    for claim in data.get("claims", []):
        for rv in claim.get("claimReview", []):
            url = safe_http_url(rv.get("url"))
            if not url:
                continue                              # no URL -> no card
            out.append(Source(
                title=rv.get("title") or (claim.get("text") or "")[:120] or "Fact-check",
                publisher=(rv.get("publisher") or {}).get("name") or "Unknown publisher",
                url=url, published_at=parse_iso(rv.get("reviewDate")),
                snippet=claim.get("text"), rating=rv.get("textualRating"),
                provenance="factcheck", source_type=source_type_for(url) or "factchecker"))
    return out, EvidenceStatus(provider=PROVIDER, ok=True)
