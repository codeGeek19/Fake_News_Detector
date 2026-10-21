import httpx
from ..schemas.schemas import EvidenceStatus, Source
from .common import parse_iso, safe_http_url, source_type_for

URL = "https://newsapi.org/v2/everything"
PROVIDER = "newsapi"
# Free plan: ~24h delay, dev-only, 1-month lookback -> weak for breaking claims. Optional.


async def search_newsapi(query: str, lang: str, api_key, client: httpx.AsyncClient, timeout: float = 8):
    if lang != "en":     # NewsAPI does not index Hindi or Marathi
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error="disabled")
    if not api_key:
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error="API key not configured")
    try:
        r = await client.get(URL, timeout=timeout, headers={"X-Api-Key": api_key},
                             params={"q": query, "language": lang, "sortBy": "relevancy", "pageSize": 10})
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error=type(e).__name__)
    out = []
    for a in data.get("articles", []):
        url = safe_http_url(a.get("url"))
        if not url or not a.get("title"):
            continue
        out.append(Source(title=a["title"], publisher=(a.get("source") or {}).get("name") or "Unknown publisher",
                          url=url, published_at=parse_iso(a.get("publishedAt")), snippet=a.get("description"),
                          provenance="newsapi", source_type=source_type_for(url)))
    return out, EvidenceStatus(provider=PROVIDER, ok=True)
