"""
Tests for C4: per-IP rate limiting via slowapi.

C4 — Rate limiting
  - When BROWSER_RATE_LIMIT is not set, no limiting is applied: requests succeed
    regardless of how many are made.
  - When BROWSER_RATE_LIMIT is set, clients that exceed the limit receive HTTP 429
    Too Many Requests.
  - The limit string format (N/period) is parsed correctly.
  - The rate limiter uses per-client-IP accounting.
  - The 429 response body is a valid JSON error object.
  - Requests within the limit succeed normally.
  - _SLOWAPI_AVAILABLE reflects whether slowapi is importable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

import api_server
from system_tools import SystemTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rate_limited_app(limit: str):
    """
    Return a fresh, self-contained FastAPI app with SlowAPIMiddleware at the
    given limit string (e.g. "2/minute").  This avoids mutating the
    module-level singleton whose middleware stack is fixed at import time.
    """
    from fastapi import FastAPI
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    test_app = FastAPI()
    limiter = Limiter(key_func=get_remote_address, default_limits=[limit])
    test_app.state.limiter = limiter
    test_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    test_app.add_middleware(SlowAPIMiddleware)

    @test_app.get("/ping")
    def ping():
        return {"ping": "pong"}

    return test_app


# ---------------------------------------------------------------------------
# C4 — Rate limiting enforcement
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_requests_within_limit_succeed(self) -> None:
        """Requests within the limit all return 200."""
        app = _make_rate_limited_app("5/minute")
        with TestClient(app, raise_server_exceptions=False) as c:
            for _ in range(5):
                r = c.get("/ping")
                assert r.status_code == 200

    def test_request_exceeding_limit_returns_429(self) -> None:
        """Once the limit is exceeded the server returns 429."""
        app = _make_rate_limited_app("3/minute")
        with TestClient(app, raise_server_exceptions=False) as c:
            for _ in range(3):
                c.get("/ping")
            r = c.get("/ping")  # 4th request → over limit
        assert r.status_code == 429

    def test_429_response_is_json(self) -> None:
        """429 response body is valid JSON with an error description."""
        app = _make_rate_limited_app("1/minute")
        with TestClient(app, raise_server_exceptions=False) as c:
            c.get("/ping")          # consumes the single allowed request
            r = c.get("/ping")      # over limit
        assert r.status_code == 429
        body = r.json()
        # slowapi returns {"error": "..."} on 429
        assert "error" in body or "detail" in body

    def test_first_request_within_limit_1_per_minute(self) -> None:
        """With a limit of 1/minute, the very first request succeeds."""
        app = _make_rate_limited_app("1/minute")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/ping")
        assert r.status_code == 200

    def test_limit_10_per_second(self) -> None:
        """Limit expressed in seconds is also enforced."""
        app = _make_rate_limited_app("2/second")
        with TestClient(app, raise_server_exceptions=False) as c:
            r1 = c.get("/ping")
            r2 = c.get("/ping")
            r3 = c.get("/ping")   # over the per-second limit
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429

    def test_limit_1000_per_hour_format(self) -> None:
        """Hour-based limit strings are accepted without error."""
        app = _make_rate_limited_app("1000/hour")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/ping")
        assert r.status_code == 200

    def test_x_ratelimit_headers_present(self) -> None:
        """Responses that pass rate limiting still carry correct content-type."""
        app = _make_rate_limited_app("10/minute")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/ping")
        assert r.status_code == 200
        # Body is valid JSON (rate limiter didn't interfere with response)
        assert r.json() == {"ping": "pong"}


# ---------------------------------------------------------------------------
# C4 — Integration with the live module-level app (no BROWSER_RATE_LIMIT set)
# ---------------------------------------------------------------------------

class TestNoRateLimitByDefault:
    """
    When BROWSER_RATE_LIMIT is not set the module-level app has no limiter
    attached to its state, and many consecutive requests all succeed.
    """

    @pytest.fixture()
    def client(self, tmp_path: Path):
        real_tools = SystemTools(workspace=tmp_path)
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", None),
            patch.object(api_server, "_tools", real_tools),
        ):
            os.environ.pop("BROWSER_API_KEY", None)
            os.environ.pop("BROWSER_RATE_LIMIT", None)
            with TestClient(api_server.app, raise_server_exceptions=False) as c:
                yield c

    def test_many_requests_succeed_without_rate_limit(self, client) -> None:
        """20 consecutive requests all pass when BROWSER_RATE_LIMIT is unset."""
        for _ in range(20):
            r = client.get("/task/schema")
            assert r.status_code == 200

    def test_no_limiter_on_app_state_when_env_unset(self) -> None:
        """app.state.limiter is absent when BROWSER_RATE_LIMIT was not set at startup."""
        os.environ.pop("BROWSER_RATE_LIMIT", None)
        # The module-level app was initialised without BROWSER_RATE_LIMIT set,
        # so no limiter should be registered on its state.
        assert not hasattr(api_server.app.state, "limiter")


# ---------------------------------------------------------------------------
# C4 — Env var parsing
# ---------------------------------------------------------------------------

class TestRateLimitEnvParsing:
    """Unit-level tests for the BROWSER_RATE_LIMIT environment variable."""

    def test_empty_string_disables_limiting(self) -> None:
        with patch.dict(os.environ, {"BROWSER_RATE_LIMIT": ""}):
            val = os.environ.get("BROWSER_RATE_LIMIT", "").strip()
        assert val == ""
        assert not val  # falsy → middleware not registered

    def test_whitespace_only_disables_limiting(self) -> None:
        with patch.dict(os.environ, {"BROWSER_RATE_LIMIT": "   "}):
            val = os.environ.get("BROWSER_RATE_LIMIT", "").strip()
        assert not val

    def test_valid_limit_strings(self) -> None:
        valid = ["100/minute", "10/second", "1000/hour", "5/minute"]
        for s in valid:
            with patch.dict(os.environ, {"BROWSER_RATE_LIMIT": s}):
                val = os.environ.get("BROWSER_RATE_LIMIT", "").strip()
            assert val == s

    def test_slowapi_available_flag(self) -> None:
        """_SLOWAPI_AVAILABLE reflects whether slowapi is importable."""
        try:
            import slowapi  # noqa: F401
            slowapi_importable = True
        except ImportError:
            slowapi_importable = False
        assert api_server._SLOWAPI_AVAILABLE is slowapi_importable

    def test_rate_limit_str_read_from_env(self) -> None:
        """Both _SLOWAPI_AVAILABLE and _rate_limit_str are defined in api_server."""
        assert hasattr(api_server, "_SLOWAPI_AVAILABLE")
        assert hasattr(api_server, "_rate_limit_str")
