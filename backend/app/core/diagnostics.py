"""Model loading with retry + visible errors.
A model that fails to load (offline, out of memory, bad name) is NOT cached as failed forever: it is retried after a cool-down,
and the last error is exposed at GET /api/health so you can see why a component is missing."""
import threading
import time

_lock = threading.Lock()
_cache: dict = {}
_errors: dict = {}


def record_error(component: str, name: str, exc: Exception) -> None:
    _errors[f"{component}:{name}"] = f"{type(exc).__name__}: {str(exc)[:240]}"


def clear_error(component: str, name: str) -> None:
    _errors.pop(f"{component}:{name}", None)


def errors() -> dict:
    return dict(_errors)


def reset() -> None:
    with _lock:
        _cache.clear()
        _errors.clear()


def load_cached(component: str, name, factory, retry_s: float = 120.0):
    """Return factory() result, cached. A failure returns None and is retried after retry_s seconds."""
    if not name:
        return None
    key = (component, name)
    now = time.time()
    with _lock:
        hit = _cache.get(key)
    if hit is not None and (hit[0] is not None or now - hit[1] < retry_s):
        return hit[0]
    try:
        value = factory()
        clear_error(component, name)
    except Exception as e:                        # noqa: BLE001 (any load problem must degrade gracefully)
        record_error(component, name, e)
        value = None
    with _lock:
        _cache[key] = (value, now)
    return value
