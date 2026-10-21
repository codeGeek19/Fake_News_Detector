import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional

_SRC = None


def source_types() -> dict:
    global _SRC
    if _SRC is None:
        p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sources.json")
        _SRC = {k: v for k, v in json.load(open(p, encoding="utf-8")).items() if not k.startswith("_")}
    return _SRC


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def source_type_for(url: str) -> Optional[str]:
    d = domain_of(url)
    for key, typ in source_types().items():
        if d == key or d.endswith("." + key):
            return typ
    return None


def safe_http_url(url: Optional[str]) -> Optional[str]:
    """Only http(s) URLs from providers are ever displayed. We never fetch them."""
    if not url:
        return None
    p = urlparse(url)
    return url if p.scheme in ("http", "https") and p.hostname else None


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
