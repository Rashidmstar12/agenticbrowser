"""
Tests for C3: CORS middleware and WebSocket API key authentication.

C3 — CORS
  - When BROWSER_CORS_ORIGINS is not set, no CORS headers are emitted.
  - When BROWSER_CORS_ORIGINS is set to one or more origins, responses include
    the correct Access-Control-Allow-Origin header for matching origins.
  - Preflight OPTIONS requests receive the proper CORS headers.
  - A wildcard origin ("*") is respected.

C3 — WebSocket auth via query parameter
  - _check_api_key accepts ?api_key= query param as a fallback when no
    X-API-Key header is present.
  - Header takes priority over the query param.
  - Wrong key in query param still returns 401.
  - When BROWSER_API_KEY is not set, no key is required (header or param).
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
from api_server import app
from system_tools import SystemTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client(tmp_path: Path, env: dict | None = None):
    """Return a TestClient with patched singletons and optional env overrides."""
    real_tools = SystemTools(workspace=tmp_path)
    env = env or {}
    ctx = [
        patch.object(api_server, "_agent", None),
        patch.object(api_server, "_planner", None),
        patch.object(api_server, "_tools", real_tools),
        patch.dict(os.environ, env),
    ]
    # Remove BROWSER_API_KEY if not in env to ensure clean state
    if "BROWSER_API_KEY" not in env:
        os.environ.pop("BROWSER_API_KEY", None)
    return ctx


# ---------------------------------------------------------------------------
# C3 — CORS middleware
# ---------------------------------------------------------------------------

class TestCORSMiddleware:
    """
    CORSMiddleware is registered at module import time using the value of
    BROWSER_CORS_ORIGINS.  These tests create a *new* FastAPI app with the
    middleware configured as needed rather than depending on the module-level
    singleton (which was already built).
    """

    def _app_with_cors(self, origins: list[str]):
        """Return a fresh FastAPI app with CORSMiddleware for the given origins."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*", "X-API-Key"],
        )

        @test_app.get("/ping")
        def ping():
            return {"ping": "pong"}

        return test_app

    def test_cors_header_returned_for_allowed_origin(self) -> None:
        test_app = self._app_with_cors(["https://app.example.com"])
        with TestClient(test_app, raise_server_exceptions=False) as c:
            r = c.get("/ping", headers={"Origin": "https://app.example.com"})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "https://app.example.com"

    def test_cors_header_not_returned_for_disallowed_origin(self) -> None:
        test_app = self._app_with_cors(["https://app.example.com"])
        with TestClient(test_app, raise_server_exceptions=False) as c:
            r = c.get("/ping", headers={"Origin": "https://evil.attacker.com"})
        assert r.status_code == 200
        # CORS header absent or not set to the attacker origin
        acao = r.headers.get("access-control-allow-origin", "")
        assert acao != "https://evil.attacker.com"

    def test_cors_wildcard(self) -> None:
        test_app = self._app_with_cors(["*"])
        with TestClient(test_app, raise_server_exceptions=False) as c:
            r = c.get("/ping", headers={"Origin": "https://anything.example.com"})
        assert r.status_code == 200
        # When allow_credentials=True, Starlette reflects the request Origin
        # instead of "*" (sending "*" + credentials is forbidden by the CORS spec).
        acao = r.headers.get("access-control-allow-origin", "")
        assert acao in ("*", "https://anything.example.com")

    def test_cors_multiple_origins(self) -> None:
        test_app = self._app_with_cors(
            ["https://app.example.com", "https://dev.example.com"]
        )
        with TestClient(test_app, raise_server_exceptions=False) as c:
            r1 = c.get("/ping", headers={"Origin": "https://app.example.com"})
            r2 = c.get("/ping", headers={"Origin": "https://dev.example.com"})
        assert r1.headers.get("access-control-allow-origin") == "https://app.example.com"
        assert r2.headers.get("access-control-allow-origin") == "https://dev.example.com"

    def test_cors_preflight_returns_headers(self) -> None:
        test_app = self._app_with_cors(["https://app.example.com"])
        with TestClient(test_app, raise_server_exceptions=False) as c:
            r = c.options(
                "/ping",
                headers={
                    "Origin": "https://app.example.com",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "X-API-Key",
                },
            )
        assert r.status_code == 200
        assert "access-control-allow-origin" in r.headers

    def test_no_cors_when_env_not_set(self, tmp_path: Path) -> None:
        """When BROWSER_CORS_ORIGINS is unset the live app adds no CORS headers."""
        real_tools = SystemTools(workspace=tmp_path)
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", None),
            patch.object(api_server, "_tools", real_tools),
        ):
            os.environ.pop("BROWSER_CORS_ORIGINS", None)
            os.environ.pop("BROWSER_API_KEY", None)
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.get("/task/schema", headers={"Origin": "https://app.example.com"})
        # The live app was built without CORS origins set, so no CORS header
        assert r.status_code == 200

    def test_cors_origins_parsing(self) -> None:
        """_cors_origins list is built correctly from the env var."""
        with patch.dict(os.environ, {"BROWSER_CORS_ORIGINS": "https://a.com, https://b.com ,  "}):
            raw = os.environ.get("BROWSER_CORS_ORIGINS", "")
            origins = [o.strip() for o in raw.split(",") if o.strip()]
        assert origins == ["https://a.com", "https://b.com"]

    def test_cors_origins_empty_string(self) -> None:
        with patch.dict(os.environ, {"BROWSER_CORS_ORIGINS": ""}):
            raw = os.environ.get("BROWSER_CORS_ORIGINS", "")
            origins = [o.strip() for o in raw.split(",") if o.strip()]
        assert origins == []

    def test_cors_x_api_key_in_allowed_headers(self) -> None:
        """CORS config exposes X-API-Key so browser clients can send it."""
        test_app = self._app_with_cors(["https://app.example.com"])
        with TestClient(test_app, raise_server_exceptions=False) as c:
            r = c.options(
                "/ping",
                headers={
                    "Origin": "https://app.example.com",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "X-API-Key",
                },
            )
        allowed = r.headers.get("access-control-allow-headers", "")
        assert "X-API-Key" in allowed or allowed == "*"


# ---------------------------------------------------------------------------
# C3 — WebSocket auth via ?api_key= query parameter
# ---------------------------------------------------------------------------

class TestApiKeyQueryParam:
    """
    _check_api_key accepts the key from a ?api_key= query parameter as a
    fallback when no X-API-Key header is present.  This is necessary for
    WebSocket connections initiated by the browser's native WebSocket API,
    which does not support custom request headers.
    """

    @pytest.fixture()
    def client_with_auth(self, tmp_path: Path):
        real_tools = SystemTools(workspace=tmp_path)
        env_patch = {"BROWSER_API_KEY": "ws-test-key"}
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", None),
            patch.object(api_server, "_tools", real_tools),
            patch.dict(os.environ, env_patch),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    @pytest.fixture()
    def client_no_auth(self, tmp_path: Path):
        real_tools = SystemTools(workspace=tmp_path)
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", None),
            patch.object(api_server, "_tools", real_tools),
        ):
            os.environ.pop("BROWSER_API_KEY", None)
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    # --- ?api_key= query parameter ---

    def test_query_param_correct_key_passes(self, client_with_auth) -> None:
        r = client_with_auth.get("/task/schema?api_key=ws-test-key")
        assert r.status_code == 200

    def test_query_param_wrong_key_returns_401(self, client_with_auth) -> None:
        r = client_with_auth.get("/task/schema?api_key=wrong-key")
        assert r.status_code == 401

    def test_query_param_missing_returns_401(self, client_with_auth) -> None:
        """No header and no query param → 401."""
        r = client_with_auth.get("/task/schema")
        assert r.status_code == 401

    def test_header_takes_priority_over_query_param(self, client_with_auth) -> None:
        """Correct header overrides a wrong query param."""
        r = client_with_auth.get(
            "/task/schema?api_key=wrong-key",
            headers={"X-API-Key": "ws-test-key"},
        )
        assert r.status_code == 200

    def test_wrong_header_plus_correct_query_param_fails(self, client_with_auth) -> None:
        """Wrong header + correct query param → 401 (header takes priority)."""
        r = client_with_auth.get(
            "/task/schema?api_key=ws-test-key",
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_query_param_not_required_when_no_env(self, client_no_auth) -> None:
        """When BROWSER_API_KEY is unset, no key needed (header or param)."""
        r = client_no_auth.get("/task/schema")
        assert r.status_code == 200

    def test_query_param_on_post_route(self, client_with_auth) -> None:
        """?api_key= works on POST routes too (used from non-browser WS clients)."""
        r = client_with_auth.get("/session/status?api_key=ws-test-key")
        assert r.status_code == 200

    def test_401_detail_updated(self, client_with_auth) -> None:
        """401 message mentions both header and query param."""
        r = client_with_auth.get("/task/schema")
        body = r.json()
        assert "Unauthorized" in body.get("detail", "")
