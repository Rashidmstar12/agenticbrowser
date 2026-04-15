"""
tests/test_gui_auth_usability.py

Tests for the GUI auth usability fix:
1. /config endpoint (auth-exempt, returns auth_required flag)
2. X-API-Key header threading through the api() helper (REST calls)
3. ?api_key= query parameter appended to /ws/task when a key is stored
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

import api_server
from api_server import app, _AUTH_EXEMPT_PATHS
from system_tools import SystemTools


# ---------------------------------------------------------------------------
# Helpers (reuse mock-agent pattern from sibling test files)
# ---------------------------------------------------------------------------

def _make_agent(*, page_active: bool = True):
    from unittest.mock import MagicMock
    agent = MagicMock()
    agent._page = MagicMock() if page_active else None
    agent.headless = True
    agent._pages = [MagicMock()] if page_active else []
    agent._recording = False
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Example"}
    agent.navigate.return_value = {"url": "https://example.com", "status": "ok"}
    agent.list_tabs.return_value = {"tabs": [], "count": 0}
    if page_active:
        agent._page.url = "https://example.com"
    return agent


def _make_planner(steps=None):
    from unittest.mock import MagicMock
    planner = MagicMock()
    planner.plan.return_value = steps or [{"action": "navigate", "url": "https://example.com"}]
    results = [{"action": "navigate", "url": "https://example.com", "status": "ok"}]
    planner.execute.return_value = results
    return planner


# ===========================================================================
# 1.  /config endpoint
# ===========================================================================

class TestConfigEndpoint:
    """Verify that GET /config returns the correct auth_required flag."""

    def test_config_returns_auth_not_required_when_no_key(self):
        """When BROWSER_API_KEY is not set, auth_required should be False."""
        with patch.object(api_server, "_API_KEY", None):
            with TestClient(app) as c:
                resp = c.get("/config")
        assert resp.status_code == 200
        assert resp.json() == {"auth_required": False}

    def test_config_returns_auth_required_when_key_set(self):
        """When BROWSER_API_KEY is set, auth_required should be True."""
        with patch.object(api_server, "_API_KEY", "secret-key"):
            with TestClient(app) as c:
                resp = c.get("/config")
        assert resp.status_code == 200
        assert resp.json() == {"auth_required": True}

    def test_config_is_accessible_without_api_key_header(self):
        """/config must not require authentication even when BROWSER_API_KEY is set."""
        with patch.object(api_server, "_API_KEY", "some-secret"):
            with TestClient(app) as c:
                # No X-API-Key header — must still return 200
                resp = c.get("/config")
        assert resp.status_code == 200

    def test_config_exempt_path_registered(self):
        """/config must be in _AUTH_EXEMPT_PATHS so the middleware skips it."""
        assert "/config" in _AUTH_EXEMPT_PATHS

    def test_config_empty_string_key_treated_as_no_key(self):
        """An empty _API_KEY string is falsy — auth_required should be False."""
        with patch.object(api_server, "_API_KEY", ""):
            with TestClient(app) as c:
                resp = c.get("/config")
        assert resp.status_code == 200
        assert resp.json()["auth_required"] is False


# ===========================================================================
# 2.  REST API calls authenticated via X-API-Key
# ===========================================================================

class TestRestApiKeyHeader:
    """Verify that REST routes accept the X-API-Key header for auth."""

    def test_rest_call_with_correct_key_succeeds(self, tmp_path):
        """A REST route must return 200 when the correct key is supplied."""
        agent = _make_agent()
        secret = "rest-secret"

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_API_KEY", secret):
            with TestClient(app) as c:
                resp = c.get(
                    "/session/status",
                    headers={"X-API-Key": secret},
                )
        assert resp.status_code == 200

    def test_rest_call_without_key_returns_401(self, tmp_path):
        """A REST route must return 401 when BROWSER_API_KEY is set and no key given."""
        secret = "rest-secret"

        with patch.object(api_server, "_API_KEY", secret):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/session/status")
        assert resp.status_code == 401

    def test_rest_call_with_wrong_key_returns_401(self, tmp_path):
        """A REST route must return 401 when the wrong key is supplied."""
        with patch.object(api_server, "_API_KEY", "correct-key"):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get(
                    "/session/status",
                    headers={"X-API-Key": "wrong-key"},
                )
        assert resp.status_code == 401

    def test_rest_call_without_key_env_succeeds(self, tmp_path):
        """When BROWSER_API_KEY is not configured, any request succeeds."""
        agent = _make_agent()

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_API_KEY", None):
            with TestClient(app) as c:
                resp = c.get("/session/status")
        assert resp.status_code == 200


# ===========================================================================
# 3.  /ws/task with ?api_key= query parameter
# ===========================================================================

class TestWsTaskApiKeyQueryParam:
    """Verify that the ?api_key= query parameter is sufficient to authenticate /ws/task."""

    def test_ws_task_api_key_query_param_accepted(self, tmp_path):
        """?api_key=<correct-key> must allow the WS to proceed."""
        agent   = _make_agent()
        planner = _make_planner()
        tools   = SystemTools(workspace=tmp_path)
        secret  = "ws-query-key"

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools), \
             patch.object(api_server, "_API_KEY", secret):
            with TestClient(app, raise_server_exceptions=False) as c:
                with c.websocket_connect(f"/ws/task?api_key={secret}") as ws:
                    ws.send_json({"intent": "navigate to google"})
                    messages = []
                    while True:
                        msg = ws.receive_json()
                        messages.append(msg)
                        if msg["type"] in ("done", "error"):
                            break

        types = [m["type"] for m in messages]
        assert "planned" in types
        assert "done" in types

    def test_ws_task_wrong_query_key_rejected(self, tmp_path):
        """?api_key=<wrong-key> must cause the WS to be closed."""
        secret = "correct-ws-key"

        with patch.object(api_server, "_API_KEY", secret):
            with TestClient(app, raise_server_exceptions=False) as c:
                with pytest.raises(Exception):
                    with c.websocket_connect("/ws/task?api_key=wrong") as ws:
                        ws.send_json({"intent": "test"})
                        ws.receive_json()

    def test_ws_task_no_key_when_required_rejected(self, tmp_path):
        """No key at all when BROWSER_API_KEY is configured must be rejected."""
        with patch.object(api_server, "_API_KEY", "required-key"):
            with TestClient(app, raise_server_exceptions=False) as c:
                with pytest.raises(Exception):
                    with c.websocket_connect("/ws/task") as ws:
                        ws.send_json({"intent": "test"})
                        ws.receive_json()

    def test_ws_task_no_key_required_allows_connection(self, tmp_path):
        """When BROWSER_API_KEY is not set, ?api_key= is not required."""
        agent   = _make_agent()
        planner = _make_planner()
        tools   = SystemTools(workspace=tmp_path)

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools), \
             patch.object(api_server, "_API_KEY", None):
            with TestClient(app, raise_server_exceptions=False) as c:
                with c.websocket_connect("/ws/task") as ws:
                    ws.send_json({"intent": "navigate to google"})
                    messages = []
                    while True:
                        msg = ws.receive_json()
                        messages.append(msg)
                        if msg["type"] in ("done", "error"):
                            break

        types = [m["type"] for m in messages]
        assert "done" in types


# ===========================================================================
# 4.  GUI HTML sanity checks
# ===========================================================================

class TestGuiHtml:
    """Verify that the GUI HTML contains the expected auth-related markup."""

    @pytest.fixture
    def html(self) -> str:
        html_path = Path(__file__).parent.parent / "gui" / "index.html"
        return html_path.read_text(encoding="utf-8")

    def test_api_key_input_field_present(self, html):
        """The sidebar must contain an API key input element."""
        assert 'id="api-key-input"' in html

    def test_save_api_key_function_present(self, html):
        """saveApiKey() function must exist in the GUI script."""
        assert "function saveApiKey(" in html

    def test_init_api_key_called_on_load(self, html):
        """initApiKey() must be called in DOMContentLoaded."""
        assert "initApiKey()" in html

    def test_ws_url_uses_api_key_when_set(self, html):
        """The WS URL construction must reference state.apiKey."""
        assert "state.apiKey" in html
        assert "?api_key=" in html

    def test_api_helper_includes_x_api_key_header(self, html):
        """The api() helper must set X-API-Key when state.apiKey is present."""
        assert "X-API-Key" in html
        assert "state.apiKey" in html

    def test_local_storage_key_persisted(self, html):
        """localStorage should be used to persist the API key across reloads."""
        assert "localStorage.setItem('browser_api_key'" in html
        assert "localStorage.getItem('browser_api_key')" in html

    def test_config_endpoint_fetched_on_init(self, html):
        """initApiKey() must fetch /config to check whether auth is required."""
        assert "'/config'" in html or '"/config"' in html

    def test_auth_required_warning_toast(self, html):
        """A warning toast must be shown when auth is required but no key is set."""
        assert "BROWSER_API_KEY" in html
        assert "warn" in html
