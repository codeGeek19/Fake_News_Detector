from datetime import datetime, timezone
import httpx
from ..schemas.schemas import EvidenceStatus, Source
from .common import domain_of, safe_http_url, source_type_for

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
PROVIDER = "gdelt"


def _seendate(s):
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


async def search_gdelt(query: str, lang: str, enabled: bool, client: httpx.AsyncClient, timeout: float = 8):
    if not enabled:
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error="disabled")
    if not query.strip():
        return [], EvidenceStatus(provider=PROVIDER, ok=True)
    q = f"({query}) sourcelang:{ {'hi': 'hindi', 'mr': 'marathi'}.get(lang, 'english') }"
    try:
        r = await client.get(URL, timeout=timeout, params={
            "query": q, "mode": "ArtList", "format": "json", "maxrecords": 15,
            "sort": "DateDesc", "timespan": "14d"})
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:      # GDELT returns plain text on bad queries -> ValueError
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error=type(e).__name__)
    out = []
    for a in data.get("articles", []):
        url = safe_http_url(a.get("url"))
        if not url or not a.get("title"):
            continue
        out.append(Source(title=a["title"], publisher=a.get("domain") or domain_of(url), url=url,
                          published_at=_seendate(a.get("seendate")),   # = when GDELT first saw it
                          snippet=a["title"], provenance="gdelt", source_type=source_type_for(url)))
    return out, EvidenceStatus(provider=PROVIDER, ok=True)
