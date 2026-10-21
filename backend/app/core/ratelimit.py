import collections
import time
from fastapi import HTTPException, Request
from .config import get_settings

_hits: dict = collections.defaultdict(collections.deque)


def rate_limit(request: Request) -> None:
    """Sliding-window limiter per client IP. Behind a proxy, configure it to forward the real IP."""
    limit = get_settings().rate_limit_per_min
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")
    q.append(now)
