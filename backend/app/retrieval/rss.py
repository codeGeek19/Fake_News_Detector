import re
from datetime import datetime, timezone
import feedparser
import httpx
from ..schemas.schemas import EvidenceStatus, Source
from .common import safe_http_url, source_type_for

URL = "https://news.google.com/rss/search"
PROVIDER = "google_news_rss"
# Google News RSS is intended for personal/non-commercial feed reading. Check its terms before
# deploying publicly; it can be switched off with ENABLE_GOOGLE_NEWS_RSS=false.


async def search_rss(query: str, lang: str, enabled: bool, client: httpx.AsyncClient, timeout: float = 8):
    if not enabled:
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error="disabled")
    if not query.strip():
        return [], EvidenceStatus(provider=PROVIDER, ok=True)
    hl, ceid = {"hi": ("hi", "IN:hi"), "mr": ("mr", "IN:mr")}.get(lang, ("en-IN", "IN:en"))
    try:
        r = await client.get(URL, timeout=timeout, params={"q": f"{query} when:14d", "hl": hl, "gl": "IN", "ceid": ceid})
        r.raise_for_status()
        feed = feedparser.parse(r.text)           # parse text we fetched ourselves, so our timeout applies
    except httpx.HTTPError as e:
        return [], EvidenceStatus(provider=PROVIDER, ok=False, error=type(e).__name__)
    out = []
    for e in feed.entries[:15]:
        url = safe_http_url(e.get("link"))
        if not url:
            continue
        pub = (e.get("source") or {}).get("title") or "Unknown publisher"
        title = re.sub(r"\s+-\s+" + re.escape(pub) + r"$", "", e.get("title", "")).strip()
        pp = e.get("published_parsed")
        out.append(Source(title=title or e.get("title", ""), publisher=pub, url=url, snippet=title,
                          published_at=datetime(*pp[:6], tzinfo=timezone.utc) if pp else None,
                          provenance="rss", source_type=source_type_for(url)))
    return out, EvidenceStatus(provider=PROVIDER, ok=True)
