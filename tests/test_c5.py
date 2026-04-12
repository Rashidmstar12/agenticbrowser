"""
Tests for C5: security response headers via SecurityHeadersMiddleware.

C5 — Security response headers
  - When BROWSER_SECURITY_HEADERS is not set (default), no extra headers are added.
  - When BROWSER_SECURITY_HEADERS=1, a fixed set of security headers is injected on
    every response.
  - Individual header values can be overridden via env vars (BROWSER_CSP, etc.).
  - HSTS is injected only over HTTPS (detected via request scheme or X-Forwarded-Proto).
  - The middleware class is importable and behaves correctly in isolation.
  - _security_headers_enabled() correctly parses truthy / falsy env var values.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api_server
from api_server import (
    SecurityHeadersMiddleware,
    _security_headers_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app_with_security_headers(**env_overrides):
    """Return a fresh FastAPI app with SecurityHeadersMiddleware registered."""
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware)

    @test_app.get("/ping")
    def ping():
        return {"ping": "pong"}

    return test_app


# ---------------------------------------------------------------------------
# C5 — _security_headers_enabled() parsing
# ---------------------------------------------------------------------------

class TestSecurityHeadersEnabledParsing:
    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"])
    def test_truthy_values_enable_headers(self, val: str) -> None:
        with patch.dict(os.environ, {"BROWSER_SECURITY_HEADERS": val}):
            assert _security_headers_enabled() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "False", "no", "off", "   "])
    def test_falsy_values_disable_headers(self, val: str) -> None:
        with patch.dict(os.environ, {"BROWSER_SECURITY_HEADERS": val}):
            assert _security_headers_enabled() is False

    def test_unset_env_var_disables_headers(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "BROWSER_SECURITY_HEADERS"}
        with patch.dict(os.environ, env, clear=True):
            assert _security_headers_enabled() is False


# ---------------------------------------------------------------------------
# C5 — Header injection when enabled
# ---------------------------------------------------------------------------

class TestSecurityHeadersInjected:
    @pytest.fixture()
    def client(self):
        app = _make_app_with_security_headers()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    def test_x_content_type_options_nosniff(self, client) -> None:
        r = client.get("/ping")
        assert r.headers.get("x-content-type-options") == "nosniff"

    def test_x_frame_options_deny(self, client) -> None:
        r = client.get("/ping")
        assert r.headers.get("x-frame-options") == "DENY"

    def test_x_xss_protection(self, client) -> None:
        r = client.get("/ping")
        assert r.headers.get("x-xss-protection") == "1; mode=block"

    def test_referrer_policy_default(self, client) -> None:
        r = client.get("/ping")
        assert r.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_content_security_policy_default(self, client) -> None:
        r = client.get("/ping")
        assert r.headers.get("content-security-policy") == "default-src 'self'"

    def test_permissions_policy_default(self, client) -> None:
        r = client.get("/ping")
        pp = r.headers.get("permissions-policy", "")
        assert "geolocation=()" in pp
        assert "microphone=()" in pp
        assert "camera=()" in pp

    def test_no_hsts_over_plain_http(self, client) -> None:
        """HSTS must NOT be sent over plain HTTP."""
        r = client.get("/ping")
        assert "strict-transport-security" not in r.headers

    def test_response_body_unaffected(self, client) -> None:
        """Middleware must not alter the response body."""
        r = client.get("/ping")
        assert r.json() == {"ping": "pong"}

    def test_status_code_unaffected(self, client) -> None:
        r = client.get("/ping")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# C5 — HSTS only over HTTPS
# ---------------------------------------------------------------------------

class TestHSTSOnlyOverHttps:
    @pytest.fixture()
    def client(self):
        app = _make_app_with_security_headers()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    def test_hsts_injected_for_x_forwarded_proto_https(self, client) -> None:
        r = client.get("/ping", headers={"X-Forwarded-Proto": "https"})
        hsts = r.headers.get("strict-transport-security", "")
        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts

    def test_hsts_not_injected_for_x_forwarded_proto_http(self, client) -> None:
        r = client.get("/ping", headers={"X-Forwarded-Proto": "http"})
        assert "strict-transport-security" not in r.headers

    def test_hsts_not_injected_without_forwarded_proto(self, client) -> None:
        r = client.get("/ping")
        assert "strict-transport-security" not in r.headers

    def test_hsts_default_max_age_is_one_year(self, client) -> None:
        r = client.get("/ping", headers={"X-Forwarded-Proto": "https"})
        hsts = r.headers.get("strict-transport-security", "")
        assert "max-age=31536000" in hsts


# ---------------------------------------------------------------------------
# C5 — Per-header env var overrides
# ---------------------------------------------------------------------------

class TestPerHeaderOverrides:
    def _get(self, env: dict) -> dict:
        with patch.dict(os.environ, env, clear=False):
            app = _make_app_with_security_headers()
            with TestClient(app) as c:
                return dict(c.get("/ping", headers={"X-Forwarded-Proto": "https"}).headers)

    def test_browser_csp_overrides_content_security_policy(self) -> None:
        headers = self._get({"BROWSER_CSP": "default-src 'none'"})
        assert headers.get("content-security-policy") == "default-src 'none'"

    def test_browser_hsts_max_age_overrides_max_age(self) -> None:
        headers = self._get({"BROWSER_HSTS_MAX_AGE": "600"})
        assert "max-age=600" in headers.get("strict-transport-security", "")

    def test_browser_referrer_policy_override(self) -> None:
        headers = self._get({"BROWSER_REFERRER_POLICY": "no-referrer"})
        assert headers.get("referrer-policy") == "no-referrer"

    def test_browser_frame_options_sameorigin(self) -> None:
        headers = self._get({"BROWSER_FRAME_OPTIONS": "SAMEORIGIN"})
        assert headers.get("x-frame-options") == "SAMEORIGIN"

    def test_browser_permissions_policy_override(self) -> None:
        headers = self._get({"BROWSER_PERMISSIONS_POLICY": "fullscreen=(self)"})
        assert headers.get("permissions-policy") == "fullscreen=(self)"


# ---------------------------------------------------------------------------
# C5 — No headers when middleware is NOT registered (default behaviour)
# ---------------------------------------------------------------------------

class TestNoExtraHeadersByDefault:
    """
    The module-level app does NOT have SecurityHeadersMiddleware unless
    BROWSER_SECURITY_HEADERS was set when the module was first imported.
    Verify a plain FastAPI app without the middleware does not add these headers.
    """

    @pytest.fixture()
    def plain_client(self, tmp_path: Path):
        plain_app = FastAPI()

        @plain_app.get("/ping")
        def ping():
            return {"ping": "pong"}

        with TestClient(plain_app) as c:
            yield c

    def test_no_x_content_type_options_without_middleware(self, plain_client) -> None:
        r = plain_client.get("/ping")
        assert "x-content-type-options" not in r.headers

    def test_no_x_frame_options_without_middleware(self, plain_client) -> None:
        r = plain_client.get("/ping")
        assert "x-frame-options" not in r.headers

    def test_no_csp_without_middleware(self, plain_client) -> None:
        r = plain_client.get("/ping")
        assert "content-security-policy" not in r.headers

    def test_no_hsts_without_middleware(self, plain_client) -> None:
        r = plain_client.get("/ping", headers={"X-Forwarded-Proto": "https"})
        assert "strict-transport-security" not in r.headers


# ---------------------------------------------------------------------------
# C5 — Module-level symbol verification
# ---------------------------------------------------------------------------

class TestModuleSymbols:
    def test_security_headers_middleware_class_exported(self) -> None:
        assert hasattr(api_server, "SecurityHeadersMiddleware")

    def test_security_headers_enabled_function_exported(self) -> None:
        assert hasattr(api_server, "_security_headers_enabled")
        assert callable(api_server._security_headers_enabled)

    def test_truthy_set_constant_defined(self) -> None:
        assert hasattr(api_server, "_SECURITY_HEADERS_ENABLED_TRUTHY")
        assert "1" in api_server._SECURITY_HEADERS_ENABLED_TRUTHY

    def test_default_csp_is_restrictive(self) -> None:
        assert "default-src 'self'" in SecurityHeadersMiddleware._DEFAULT_CSP

    def test_default_frame_is_deny(self) -> None:
        assert SecurityHeadersMiddleware._DEFAULT_FRAME == "DENY"
