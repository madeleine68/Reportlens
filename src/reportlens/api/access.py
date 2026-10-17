"""Who may ask, and how often: protects the free-tier LLM quota when the demo is public.

- DEMO_PASSWORD: if set, /ask and /ask/stream need it in the X-Demo-Password header (compared in
  constant time). A wrong password is rejected before it counts against any limit.
- RATE_LIMIT_PER_HOUR: questions per visitor in any rolling hour (visitor = client IP).
- DAILY_QUESTION_CAP: questions per calendar day (UTC) for everyone together; the hard limit.

Counts live in memory: right for a single server process (one Hugging Face Space), reset when it
restarts. The visitor's IP is the first X-Forwarded-For entry (the Space sits behind a proxy),
which a client can fake; the daily cap is what bounds the cost.
"""

import datetime as dt
import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from reportlens.core.config import get_settings

PASSWORD_HEADER = "X-Demo-Password"
HOUR_S = 3600.0


def password_required() -> bool:
    """True when a demo password is set (an empty DEMO_PASSWORD= counts as unset)."""
    password = get_settings().demo_password
    return password is not None and bool(password.get_secret_value())


def check_password(given: str | None) -> bool:
    """True if no password is required, or `given` matches it."""
    if not password_required():
        return True
    expected = get_settings().demo_password.get_secret_value()  # type: ignore[union-attr]
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


def client_key(request: Request) -> str:
    """The visitor's IP: the first X-Forwarded-For entry behind a proxy, else the socket's."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if first := forwarded.split(",")[0].strip():
        return first
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Per-visitor rolling-hour limit plus a daily cap for everyone; thread-safe."""

    def __init__(self, clock=time.monotonic, today=lambda: dt.datetime.now(dt.UTC).date()):
        self._clock, self._today = clock, today
        self._lock = threading.Lock()
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._day: dt.date | None = None
        self._day_count = 0

    def allow(self, key: str, per_hour: int, per_day: int) -> tuple[bool, str, int]:
        """(allowed, reason if not, seconds to wait). A refused question is not counted."""
        with self._lock:
            now, today = self._clock(), self._today()
            if today != self._day:
                self._day, self._day_count = today, 0
            if per_day and self._day_count >= per_day:
                return (
                    False,
                    "The demo's daily question limit is reached; try again tomorrow.",
                    3600,
                )
            recent = self._recent[key]
            while recent and now - recent[0] >= HOUR_S:
                recent.popleft()
            if per_hour and len(recent) >= per_hour:
                wait = int(HOUR_S - (now - recent[0])) + 1
                return False, f"Limit of {per_hour} questions per hour reached; try later.", wait
            recent.append(now)
            self._day_count += 1
            return True, "", 0


limiter = RateLimiter()


def require_access(request: Request) -> None:
    """FastAPI dependency for the question endpoints: password first, then the limits."""
    if not check_password(request.headers.get(PASSWORD_HEADER)):
        raise HTTPException(status_code=401, detail="Wrong or missing demo password.")
    settings = get_settings()
    allowed, reason, wait = limiter.allow(
        client_key(request), settings.rate_limit_per_hour, settings.daily_question_cap
    )
    if not allowed:
        raise HTTPException(status_code=429, detail=reason, headers={"Retry-After": str(wait)})
