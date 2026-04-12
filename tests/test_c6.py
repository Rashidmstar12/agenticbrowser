"""
Tests for C6: request body size limiting via MaxBodySizeMiddleware.

C6 — Request body size limiting
  - When BROWSER_MAX_BODY_SIZE is not set (or 0), no limit is applied.
  - When set, requests whose body exceeds the limit receive HTTP 413.
  - Content-Length header is used for a fast pre-check.
  - Bodies streamed without Content-Length are counted on the fly.
  - The limit is respected exactly at the boundary (size == limit is allowed,
    size == limit + 1 is rejected).
  - _max_body_size() correctly parses the env var.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

import api_server
from api_server import MaxBodySizeMiddleware, _max_body_size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(limit_bytes: int | None = None):
    """Return a minimal FastAPI app with MaxBodySizeMiddleware registered."""
    test_app = FastAPI()
    test_app.add_middleware(MaxBodySizeMiddleware)

    @test_app.post("/echo")
    async def echo(request: Request):
        from fastapi.responses import JSONResponse
        body = await request.body()
        return JSONResponse({"length": len(body)})

    @test_app.get("/ping")
    def ping():
        return {"ping": "pong"}

    return test_app


def _client(limit: int):
    """Return a TestClient whose env has BROWSER_MAX_BODY_SIZE=<limit>."""
    env = {**os.environ, "BROWSER_MAX_BODY_SIZE": str(limit)}
    with patch.dict(os.environ, env):
        app = _make_app()
        return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# C6 — _max_body_size() parsing
# ---------------------------------------------------------------------------

class TestMaxBodySizeParsing:
    def test_unset_returns_zero(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "BROWSER_MAX_BODY_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            assert _max_body_size() == 0

    def test_empty_string_returns_zero(self) -> None:
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": ""}):
            assert _max_body_size() == 0

    def test_zero_returns_zero(self) -> None:
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": "0"}):
            assert _max_body_size() == 0

    def test_positive_integer_parsed(self) -> None:
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": "1048576"}):
            assert _max_body_size() == 1048576

    def test_invalid_value_returns_zero(self) -> None:
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": "abc"}):
            assert _max_body_size() == 0

    def test_negative_value_clamped_to_zero(self) -> None:
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": "-100"}):
            assert _max_body_size() == 0

    def test_whitespace_stripped(self) -> None:
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": "  512  "}):
            assert _max_body_size() == 512


# ---------------------------------------------------------------------------
# C6 — Content-Length pre-check
# ---------------------------------------------------------------------------

class TestContentLengthPreCheck:
    def test_body_at_limit_is_accepted(self) -> None:
        limit = 100
        body = b"x" * limit
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": str(limit)}):
            app = _make_app()
            with TestClient(app) as c:
                r = c.post("/echo", content=body)
        assert r.status_code == 200
        assert r.json()["length"] == limit

    def test_body_below_limit_is_accepted(self) -> None:
        limit = 100
        body = b"x" * (limit - 1)
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": str(limit)}):
            app = _make_app()
            with TestClient(app) as c:
                r = c.post("/echo", content=body)
        assert r.status_code == 200

    def test_body_one_byte_over_limit_is_rejected(self) -> None:
        limit = 100
        body = b"x" * (limit + 1)
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": str(limit)}):
            app = _make_app()
            with TestClient(app) as c:
                r = c.post("/echo", content=body)
        assert r.status_code == 413

    def test_413_response_body_contains_limit(self) -> None:
        limit = 64
        body = b"x" * (limit + 1)
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": str(limit)}):
            app = _make_app()
            with TestClient(app) as c:
                r = c.post("/echo", content=body)
        assert "64" in r.text

    def test_get_request_with_no_body_is_always_accepted(self) -> None:
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": "1"}):
            app = _make_app()
            with TestClient(app) as c:
                r = c.get("/ping")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# C6 — No limit when env var is unset
# ---------------------------------------------------------------------------

class TestNoLimitByDefault:
    def test_large_body_accepted_when_no_limit_set(self) -> None:
        body = b"x" * 10_000
        env = {k: v for k, v in os.environ.items() if k != "BROWSER_MAX_BODY_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            app = _make_app()
            with TestClient(app) as c:
                r = c.post("/echo", content=body)
        assert r.status_code == 200
        assert r.json()["length"] == 10_000

    def test_large_body_accepted_when_limit_is_zero(self) -> None:
        body = b"y" * 5_000
        with patch.dict(os.environ, {"BROWSER_MAX_BODY_SIZE": "0"}):
            app = _make_app()
            with TestClient(app) as c:
                r = c.post("/echo", content=body)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# C6 — Module symbol verification
# ---------------------------------------------------------------------------

class TestModuleSymbols:
    def test_max_body_size_middleware_exported(self) -> None:
        assert hasattr(api_server, "MaxBodySizeMiddleware")

    def test_max_body_size_function_exported(self) -> None:
        assert hasattr(api_server, "_max_body_size")
        assert callable(api_server._max_body_size)

    def test_middleware_is_base_http_middleware_subclass(self) -> None:
        from starlette.middleware.base import BaseHTTPMiddleware
        assert issubclass(MaxBodySizeMiddleware, BaseHTTPMiddleware)
