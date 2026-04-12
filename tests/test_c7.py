"""
Tests for C7: structured JSON access logging via AccessLogMiddleware.

C7 — Structured JSON access logging
  - When BROWSER_ACCESS_LOG is not set, no access-log middleware is registered.
  - When enabled, each request emits exactly one JSON log line at INFO level
    to the ``agenticbrowser.access`` logger.
  - Each record contains: timestamp, method, path, status, duration_ms, client_ip.
  - client_ip is taken from X-Forwarded-For when present; falls back to client.host.
  - The log record is valid JSON.
  - _access_log_enabled() correctly parses truthy / falsy env var values.
  - AccessLogMiddleware does not alter the response body or status code.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api_server
from api_server import _ACCESS_LOGGER, AccessLogMiddleware, _access_log_enabled

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app():
    """Return a minimal FastAPI app with AccessLogMiddleware registered."""
    test_app = FastAPI()
    test_app.add_middleware(AccessLogMiddleware)

    @test_app.get("/ping")
    def ping():
        return {"ping": "pong"}

    @test_app.post("/echo")
    async def echo(request):
        from fastapi.responses import PlainTextResponse
        body = await request.body()
        return PlainTextResponse(body.decode())

    @test_app.get("/error")
    def error():
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="boom")

    return test_app


# ---------------------------------------------------------------------------
# C7 — _access_log_enabled() parsing
# ---------------------------------------------------------------------------

class TestAccessLogEnabledParsing:
    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"])
    def test_truthy_values_enable_logging(self, val: str) -> None:
        with patch.dict(os.environ, {"BROWSER_ACCESS_LOG": val}):
            assert _access_log_enabled() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "False", "no", "off"])
    def test_falsy_values_disable_logging(self, val: str) -> None:
        with patch.dict(os.environ, {"BROWSER_ACCESS_LOG": val}):
            assert _access_log_enabled() is False

    def test_unset_env_var_disables_logging(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "BROWSER_ACCESS_LOG"}
        with patch.dict(os.environ, env, clear=True):
            assert _access_log_enabled() is False


# ---------------------------------------------------------------------------
# C7 — One JSON log line per request
# ---------------------------------------------------------------------------

class TestAccessLogEmission:
    @pytest.fixture()
    def app_and_client(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield app, c

    def _capture_log(self, client, method="get", path="/ping", **kwargs):
        """Make a request and return the parsed JSON record from the access logger."""
        records: list = []
        return records

    def test_one_record_emitted_per_request(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
        access_records = [r for r in caplog.records if r.name == "agenticbrowser.access"]
        assert len(access_records) == 1

    def test_log_record_is_valid_json(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
        access_records = [r for r in caplog.records if r.name == "agenticbrowser.access"]
        assert len(access_records) == 1
        record = json.loads(access_records[0].getMessage())
        assert isinstance(record, dict)

    def test_log_record_contains_method(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
        record = json.loads(
            [r for r in caplog.records if r.name == "agenticbrowser.access"][0].getMessage()
        )
        assert record["method"] == "GET"

    def test_log_record_contains_path(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
        record = json.loads(
            [r for r in caplog.records if r.name == "agenticbrowser.access"][0].getMessage()
        )
        assert record["path"] == "/ping"

    def test_log_record_contains_status_code(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
        record = json.loads(
            [r for r in caplog.records if r.name == "agenticbrowser.access"][0].getMessage()
        )
        assert record["status"] == 200

    def test_log_record_contains_duration_ms(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
        record = json.loads(
            [r for r in caplog.records if r.name == "agenticbrowser.access"][0].getMessage()
        )
        assert "duration_ms" in record
        assert isinstance(record["duration_ms"], (int, float))
        assert record["duration_ms"] >= 0

    def test_log_record_contains_timestamp(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
        record = json.loads(
            [r for r in caplog.records if r.name == "agenticbrowser.access"][0].getMessage()
        )
        assert "timestamp" in record
        assert "T" in record["timestamp"]

    def test_log_record_contains_client_ip(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
        record = json.loads(
            [r for r in caplog.records if r.name == "agenticbrowser.access"][0].getMessage()
        )
        assert "client_ip" in record

    def test_response_body_unaffected(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            r = c.get("/ping")
        assert r.json() == {"ping": "pong"}

    def test_response_status_unaffected(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            r = c.get("/ping")
        assert r.status_code == 200

    def test_error_status_code_logged_correctly(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/error")
        access_records = [r2 for r2 in caplog.records if r2.name == "agenticbrowser.access"]
        assert len(access_records) == 1
        record = json.loads(access_records[0].getMessage())
        assert record["status"] == 500

    def test_post_method_logged(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.post("/echo", content=b"hello")
        access_records = [r for r in caplog.records if r.name == "agenticbrowser.access"]
        record = json.loads(access_records[0].getMessage())
        assert record["method"] == "POST"
        assert record["path"] == "/echo"

    def test_x_forwarded_for_used_as_client_ip(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping", headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8"})
        record = json.loads(
            [r for r in caplog.records if r.name == "agenticbrowser.access"][0].getMessage()
        )
        assert record["client_ip"] == "1.2.3.4"

    def test_multiple_requests_emit_multiple_records(self, app_and_client, caplog) -> None:
        _, c = app_and_client
        with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
            c.get("/ping")
            c.get("/ping")
            c.get("/ping")
        access_records = [r for r in caplog.records if r.name == "agenticbrowser.access"]
        assert len(access_records) == 3


# ---------------------------------------------------------------------------
# C7 — Logger name and level
# ---------------------------------------------------------------------------

class TestLoggerName:
    def test_access_logger_name_is_correct(self) -> None:
        assert _ACCESS_LOGGER.name == "agenticbrowser.access"

    def test_access_logger_emits_at_info_level(self, caplog) -> None:
        app = _make_app()
        with TestClient(app) as c:
            with caplog.at_level(logging.INFO, logger="agenticbrowser.access"):
                c.get("/ping")
        access_records = [r for r in caplog.records if r.name == "agenticbrowser.access"]
        assert len(access_records) == 1
        assert access_records[0].levelno == logging.INFO


# ---------------------------------------------------------------------------
# C7 — Module symbol verification
# ---------------------------------------------------------------------------

class TestModuleSymbols:
    def test_access_log_middleware_exported(self) -> None:
        assert hasattr(api_server, "AccessLogMiddleware")

    def test_access_log_enabled_function_exported(self) -> None:
        assert hasattr(api_server, "_access_log_enabled")
        assert callable(api_server._access_log_enabled)

    def test_access_logger_exported(self) -> None:
        assert hasattr(api_server, "_ACCESS_LOGGER")

    def test_middleware_is_base_http_middleware_subclass(self) -> None:
        from starlette.middleware.base import BaseHTTPMiddleware
        assert issubclass(AccessLogMiddleware, BaseHTTPMiddleware)
