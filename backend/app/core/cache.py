"""Tiny in-memory TTL cache for retrieval results (swap for Redis if you scale out)."""
import threading
from cachetools import TTLCache
from .config import get_settings

_lock = threading.Lock()
_cache: TTLCache | None = None


def _get() -> TTLCache:
    global _cache
    if _cache is None:
        _cache = TTLCache(maxsize=512, ttl=get_settings().cache_ttl)
    return _cache


def cache_get(key):
    with _lock:
        return _get().get(key)


def cache_set(key, value):
    with _lock:
        _get()[key] = value


def cache_clear():
    with _lock:
        _get().clear()
