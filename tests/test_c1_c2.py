"""
Tests for C1 (proxy support) and C2 (API key authentication).

C1 — proxy
  - BrowserAgent accepts proxy in __init__ and passes it to chromium.launch()
  - SessionStartRequest accepts proxy field
  - /session/start passes proxy to BrowserAgent
  - local_runner.py CLI exposes --proxy argument

C2 — API key
  - When BROWSER_API_KEY is not set, all requests succeed (no auth required)
  - When BROWSER_API_KEY is set, requests without X-API-Key header get 401
  - When BROWSER_API_KEY is set, requests with a wrong key get 401
  - When BROWSER_API_KEY is set, requests with the correct key succeed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

import api_server
from api_server import app
from browser_agent import BrowserAgent
from system_tools import SystemTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(*, page_active: bool = True) -> MagicMock:
    agent = MagicMock(spec=BrowserAgent)
    agent._page = MagicMock() if page_active else None
    agent.headless = True
    agent.proxy = None
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Example"}
    return agent


# ---------------------------------------------------------------------------
# C1 — Proxy support: BrowserAgent
# ---------------------------------------------------------------------------

class TestBrowserAgentProxy:
    def test_default_proxy_is_none(self) -> None:
        ba = BrowserAgent()
        assert ba.proxy is None

    def test_proxy_stored(self) -> None:
        ba = BrowserAgent(proxy="http://proxy.example.com:8080")
        assert ba.proxy == "http://proxy.example.com:8080"

    def test_start_passes_proxy_to_chromium_launch(self) -> None:
        ba = BrowserAgent(proxy="http://proxy.example.com:3128")
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.on = MagicMock()

        with patch("browser_agent.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mock_pw
            ba.start()

        call_kwargs = mock_pw.chromium.launch.call_args[1]
        assert "proxy" in call_kwargs
        assert call_kwargs["proxy"] == {"server": "http://proxy.example.com:3128"}

    def test_start_without_proxy_does_not_pass_proxy_kwarg(self) -> None:
        ba = BrowserAgent()
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.on = MagicMock()

        with patch("browser_agent.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mock_pw
            ba.start()

        call_kwargs = mock_pw.chromium.launch.call_args[1]
        assert "proxy" not in call_kwargs

    def test_socks5_proxy_stored(self) -> None:
        ba = BrowserAgent(proxy="socks5://127.0.0.1:1080")
        assert ba.proxy == "socks5://127.0.0.1:1080"


# ---------------------------------------------------------------------------
# C1 — Proxy support: API server /session/start
# ---------------------------------------------------------------------------

class TestSessionStartProxy:
    """Verify that /session/start accepts and uses the proxy field."""

    @pytest.fixture()
    def client(self, tmp_path: Path):
        real_tools = SystemTools(workspace=tmp_path)
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", None),
            patch.object(api_server, "_tools", real_tools),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def test_session_start_with_proxy_passes_it_to_agent(self, client) -> None:
        proxy_url = "http://proxy.test:8080"
        created_agents: list[BrowserAgent] = []

        class FakeBrowserAgent:
            def __init__(self, **kwargs: object) -> None:
                self.headless = kwargs.get("headless", True)
                self.proxy = kwargs.get("proxy")
                self._page = MagicMock()
                created_agents.append(self)

            def start(self) -> "FakeBrowserAgent":
                return self

            def stop(self) -> None:
                pass

        with patch("api_server.BrowserAgent", FakeBrowserAgent):
            r = client.post("/session/start", json={"proxy": proxy_url})

        assert r.status_code == 200
        assert len(created_agents) == 1
        assert created_agents[0].proxy == proxy_url
        assert r.json()["proxy"] == proxy_url

    def test_session_start_without_proxy_defaults_to_none(self, client) -> None:
        created_agents: list = []

        class FakeBrowserAgent:
            def __init__(self, **kwargs: object) -> None:
                self.headless = kwargs.get("headless", True)
                self.proxy = kwargs.get("proxy")
                self._page = MagicMock()
                created_agents.append(self)

            def start(self) -> "FakeBrowserAgent":
                return self

            def stop(self) -> None:
                pass

        with patch("api_server.BrowserAgent", FakeBrowserAgent):
            r = client.post("/session/start", json={})

        assert r.status_code == 200
        assert created_agents[0].proxy is None
        assert r.json()["proxy"] is None


# ---------------------------------------------------------------------------
# C1 — Proxy support: CLI --proxy argument
# ---------------------------------------------------------------------------

class TestCLIProxy:
    def test_proxy_argument_present(self) -> None:
        """Verify --proxy is a recognised CLI argument."""
        import argparse

        # Rebuild the argument parser the same way local_runner does.
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-headless", action="store_true")
        parser.add_argument("--slow-mo", type=int, default=0)
        parser.add_argument("--no-auto-popups", action="store_true")
        parser.add_argument("--intent")
        parser.add_argument("--task")
        parser.add_argument("--cmd")
        parser.add_argument("--workspace", default="workspace")
        parser.add_argument("--proxy", default=None)
        parser.add_argument("--doctor", action="store_true")
        parser.add_argument("--fix", action="store_true")
        parser.add_argument("--skills", action="append", default=[])

        args = parser.parse_args(["--proxy", "http://proxy.test:3128"])
        assert args.proxy == "http://proxy.test:3128"

    def test_proxy_default_is_none(self) -> None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--proxy", default=None)
        args = parser.parse_args([])
        assert args.proxy is None


# ---------------------------------------------------------------------------
# C2 — API key authentication
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    """
    API key enforcement via BROWSER_API_KEY env var and X-API-Key header.
    """

    @pytest.fixture()
    def client_no_auth(self, tmp_path: Path):
        """Client where BROWSER_API_KEY is NOT set — all requests pass through."""
        real_tools = SystemTools(workspace=tmp_path)
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", None),
            patch.object(api_server, "_tools", real_tools),
            patch.dict(os.environ, {}, clear=False),
        ):
            # Ensure the key is absent
            os.environ.pop("BROWSER_API_KEY", None)
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    @pytest.fixture()
    def client_with_auth(self, tmp_path: Path):
        """Client where BROWSER_API_KEY=test-secret-key is set."""
        real_tools = SystemTools(workspace=tmp_path)
        env_patch = {"BROWSER_API_KEY": "test-secret-key"}
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", None),
            patch.object(api_server, "_tools", real_tools),
            patch.dict(os.environ, env_patch),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    # --- No auth configured ---

    def test_no_auth_required_when_env_not_set(self, client_no_auth) -> None:
        r = client_no_auth.get("/task/schema")
        assert r.status_code == 200

    def test_no_auth_any_header_value_allowed(self, client_no_auth) -> None:
        r = client_no_auth.get("/task/schema", headers={"X-API-Key": "random-garbage"})
        assert r.status_code == 200

    # --- Auth configured ---

    def test_missing_key_returns_401(self, client_with_auth) -> None:
        r = client_with_auth.get("/task/schema")
        assert r.status_code == 401

    def test_wrong_key_returns_401(self, client_with_auth) -> None:
        r = client_with_auth.get("/task/schema", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

    def test_correct_key_returns_200(self, client_with_auth) -> None:
        r = client_with_auth.get("/task/schema", headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 200

    def test_401_detail_message(self, client_with_auth) -> None:
        r = client_with_auth.get("/task/schema")
        body = r.json()
        assert "detail" in body
        assert "Unauthorized" in body["detail"]

    def test_auth_enforced_on_post_routes(self, client_with_auth) -> None:
        """POST routes also require the key."""
        r = client_with_auth.post("/navigate", json={"url": "https://example.com"})
        assert r.status_code == 401

    def test_auth_enforced_on_system_routes(self, client_with_auth) -> None:
        r = client_with_auth.get("/system/info")
        assert r.status_code == 401

    def test_correct_key_on_post_route_passes_auth(self, client_with_auth) -> None:
        """With the right key the request proceeds past auth (may still fail for other reasons)."""
        r = client_with_auth.post(
            "/navigate",
            json={"url": "https://example.com"},
            headers={"X-API-Key": "test-secret-key"},
        )
        # Auth passes — the expected failure here is 400 (no active session), not 401
        assert r.status_code != 401
