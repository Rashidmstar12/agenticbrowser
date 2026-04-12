"""
Integration tests for the FastAPI routes in api_server.py.

Strategy
--------
- The browser-dependent routes (session/*, navigate, click, etc.) are tested
  by patching the global singletons (_agent, _planner) with MagicMock objects
  so no real Chromium process is launched.
- The system-tools routes (/system/*) create a real SystemTools instance
  scoped to a temp directory via a fixture that overrides _tools.
- The /task/schema route needs no patching at all.

Tests verify:
  * HTTP status codes (200 vs 400 vs 422)
  * Response structure and key presence
  * Error propagation (e.g. 400 when session is inactive, 422 on bad input)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import httpx TestClient — installed together with pytest-asyncio / httpx.
from fastapi.testclient import TestClient

import api_server
from api_server import app
from system_tools import SystemTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(*, page_active: bool = True) -> MagicMock:
    """Return a MagicMock that passes the `_page is not None` guard."""
    agent = MagicMock()
    agent._page = MagicMock() if page_active else None
    agent.headless = True
    agent.get_page_info.return_value = {
        "url": "https://example.com",
        "title": "Example",
    }
    agent.navigate.return_value = {"url": "https://example.com", "status": "ok"}
    agent.close_popups.return_value = {"dismissed": 0}
    agent.click.return_value = {"selector": "button", "status": "ok"}
    agent.type_text.return_value = {"selector": "input", "status": "ok"}
    agent.fill.return_value = {"selector": "input", "status": "ok"}
    agent.press_key.return_value = {"key": "Enter", "status": "ok"}
    agent.hover.return_value = {"selector": "a", "status": "ok"}
    agent.select_option.return_value = {"selector": "select", "value": "opt", "status": "ok"}
    agent.scroll.return_value = {"x": 0, "y": 500, "status": "ok"}
    agent.scroll_to_element.return_value = {"selector": "h1", "status": "ok"}
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Ex"}
    agent.get_text.return_value = "page text"
    agent.get_html.return_value = "<p>html</p>"
    agent.get_attribute.return_value = "value"
    agent.query_all.return_value = [{"tag": "div", "text": "item"}]
    agent.evaluate.return_value = 42
    agent.screenshot.return_value = {"path": None, "data": "base64=="}
    agent.wait_for_selector.return_value = {"selector": "div", "status": "ok"}
    agent.wait_for_load_state.return_value = {"state": "networkidle", "status": "ok"}
    agent.extract_links.return_value = {"links": [], "count": 0}
    agent.extract_table.return_value = {"headers": [], "rows": [], "count": 0}
    # assert_text / assert_url: MagicMock blocks attribute access for names
    # starting with "assert", so use setattr to bypass the safety check.
    setattr(agent, "assert_text", MagicMock(return_value={"found": True, "text": "hello"}))
    setattr(agent, "assert_url",  MagicMock(return_value={"matched": True, "url": "https://example.com"}))
    agent.wait_text.return_value = {"found": True, "text": "hello"}
    agent.get_cookies.return_value = []
    agent.add_cookies.return_value = None
    agent.new_tab.return_value = {"index": 1, "url": "about:blank"}
    agent.switch_tab.return_value = {"index": 0, "status": "ok"}
    agent.close_tab.return_value = {"closed_index": 0, "status": "ok"}
    agent.list_tabs.return_value = {"tabs": [], "count": 0}
    return agent


def _make_planner(steps: list[dict] | None = None, *, fail: bool = False) -> MagicMock:
    planner = MagicMock()
    steps = steps or [{"action": "navigate", "url": "https://example.com"}]
    planner.plan.return_value = steps
    if fail:
        ok_result = [{"status": "error", "error": "step failed"}]
    else:
        ok_result = [{"status": "ok", "result": {}}]
    planner.execute.return_value = ok_result
    return planner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path: Path):
    """Test client with fresh in-memory state: no real browser, real SystemTools."""
    real_tools = SystemTools(workspace=tmp_path)
    with (
        patch.object(api_server, "_agent", None),
        patch.object(api_server, "_planner", None),
        patch.object(api_server, "_tools", real_tools),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, real_tools


@pytest.fixture()
def client_with_session(tmp_path: Path):
    """Test client with an active (mocked) browser session."""
    agent = _make_agent()
    planner = _make_planner()
    real_tools = SystemTools(workspace=tmp_path)
    with (
        patch.object(api_server, "_agent", agent),
        patch.object(api_server, "_planner", planner),
        patch.object(api_server, "_tools", real_tools),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, agent, planner, real_tools


# ---------------------------------------------------------------------------
# /task/schema — no session required
# ---------------------------------------------------------------------------

class TestTaskSchema:
    def test_returns_schema(self, client) -> None:
        c, _ = client
        r = c.get("/task/schema")
        assert r.status_code == 200
        body = r.json()
        assert "schema" in body
        assert "navigate" in body["schema"]
        assert "click" in body["schema"]

    def test_schema_contains_all_known_actions(self, client) -> None:
        c, _ = client
        from task_planner import STEP_SCHEMA
        r = c.get("/task/schema")
        schema = r.json()["schema"]
        for action in STEP_SCHEMA:
            assert action in schema, f"Missing action: {action}"


# ---------------------------------------------------------------------------
# /session/status — no active session
# ---------------------------------------------------------------------------

class TestSessionStatus:
    def test_inactive_returns_active_false(self, client) -> None:
        c, _ = client
        r = c.get("/session/status")
        assert r.status_code == 200
        assert r.json()["active"] is False


# ---------------------------------------------------------------------------
# Routes requiring active session — 400 when session inactive
# ---------------------------------------------------------------------------

class TestRequiresSession:
    ROUTES_POST = [
        ("/navigate",          {"url": "https://example.com"}),
        ("/popups/close",      {}),
        ("/click",             {"selector": "button"}),
        ("/fill",              {"selector": "input", "value": "x"}),
        ("/press_key",         {"key": "Enter"}),
        ("/scroll",            {}),
        ("/page/text",         {"selector": "body"}),
    ]

    @pytest.mark.parametrize("path,body", ROUTES_POST)
    def test_400_without_session(self, client, path: str, body: dict) -> None:
        c, _ = client
        r = c.post(path, json=body)
        assert r.status_code == 400, f"{path} should return 400 without session"

    def test_get_page_info_400_without_session(self, client) -> None:
        c, _ = client
        r = c.get("/page/info")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Navigation route with active session
# ---------------------------------------------------------------------------

class TestNavigateRoute:
    def test_navigate_returns_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/navigate", json={"url": "https://example.com"})
        assert r.status_code == 200
        agent.navigate.assert_called_once_with("https://example.com", wait_until="domcontentloaded")

    def test_navigate_custom_wait_until(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/navigate", json={"url": "https://example.com", "wait_until": "load"})
        assert r.status_code == 200
        _, kwargs = agent.navigate.call_args
        assert kwargs.get("wait_until") == "load" or agent.navigate.call_args[0][1] == "load"


# ---------------------------------------------------------------------------
# Click, fill, press_key
# ---------------------------------------------------------------------------

class TestInteractionRoutes:
    def test_click_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/click", json={"selector": "#btn"})
        assert r.status_code == 200
        agent.click.assert_called_once()

    def test_fill_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/fill", json={"selector": "input", "value": "hello"})
        assert r.status_code == 200
        agent.fill.assert_called_once_with("input", "hello")

    def test_press_key_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/press_key", json={"key": "Enter"})
        assert r.status_code == 200

    def test_hover_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/hover", json={"selector": "a.link"})
        assert r.status_code == 200

    def test_scroll_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/scroll", json={"y": 300})
        assert r.status_code == 200

    def test_select_option_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/select_option", json={"selector": "select#s", "value": "opt1"})
        assert r.status_code == 200
        agent.select_option.assert_called_once_with("select#s", "opt1")


# ---------------------------------------------------------------------------
# Page info routes
# ---------------------------------------------------------------------------

class TestPageInfoRoutes:
    def test_page_info_200(self, client_with_session) -> None:
        c, *_ = client_with_session
        r = c.get("/page/info")
        assert r.status_code == 200
        body = r.json()
        assert "url" in body or "title" in body

    def test_get_text_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        agent.get_text.return_value = "some text"
        r = c.post("/page/text", json={"selector": "body"})
        assert r.status_code == 200

    def test_get_html_200(self, client_with_session) -> None:
        c, agent, *_ = client_with_session
        r = c.post("/page/html", json={"selector": "body"})
        assert r.status_code == 200

    def test_extract_links_200(self, client_with_session) -> None:
        c, *_ = client_with_session
        r = c.post("/page/extract_links", json={})
        assert r.status_code == 200

    def test_extract_table_200(self, client_with_session) -> None:
        c, *_ = client_with_session
        r = c.post("/page/extract_table", json={})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /session/status with active session
# ---------------------------------------------------------------------------

class TestSessionStatusActive:
    def test_active_true(self, client_with_session) -> None:
        c, *_ = client_with_session
        r = c.get("/session/status")
        assert r.status_code == 200
        assert r.json()["active"] is True


# ---------------------------------------------------------------------------
# System tools routes (real SystemTools, no browser needed)
# ---------------------------------------------------------------------------

class TestSystemToolsRoutes:
    def test_write_and_read_file(self, client) -> None:
        c, _ = client
        r = c.post("/system/write_file", json={"path": "hello.txt", "content": "world"})
        assert r.status_code == 200
        assert r.json()["bytes_written"] == len("world")

        r2 = c.post("/system/read_file", json={"path": "hello.txt"})
        assert r2.status_code == 200
        assert r2.json()["content"] == "world"

    def test_append_file(self, client) -> None:
        c, _ = client
        c.post("/system/write_file", json={"path": "log.txt", "content": "line1\n"})
        r = c.post("/system/append_file", json={"path": "log.txt", "content": "line2\n"})
        assert r.status_code == 200
        r2 = c.post("/system/read_file", json={"path": "log.txt"})
        assert "line2" in r2.json()["content"]

    def test_read_missing_file_returns_500(self, client) -> None:
        c, _ = client
        r = c.post("/system/read_file", json={"path": "does_not_exist.txt"})
        # FastAPI converts unhandled exceptions to 500
        assert r.status_code == 500

    def test_make_dir(self, client) -> None:
        c, _ = client
        r = c.post("/system/make_dir", json={"path": "mydir"})
        assert r.status_code == 200
        assert r.json()["created"] is True

    def test_list_dir(self, client) -> None:
        c, _ = client
        c.post("/system/write_file", json={"path": "a.txt", "content": "x"})
        r = c.post("/system/list_dir", json={"path": "."})
        assert r.status_code == 200
        names = [e["name"] for e in r.json()["entries"]]
        assert "a.txt" in names

    def test_delete_file(self, client) -> None:
        c, _ = client
        c.post("/system/write_file", json={"path": "del.txt", "content": "bye"})
        r = c.post("/system/delete_file", json={"path": "del.txt"})
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_system_info(self, client) -> None:
        c, _ = client
        r = c.get("/system/info")
        assert r.status_code == 200
        body = r.json()
        assert "workspace" in body
        assert "python" in body

    def test_run_python(self, client) -> None:
        c, _ = client
        r = c.post("/system/run_python", json={"code": "print('hi')"})
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert "hi" in body["stdout"]

    def test_run_shell(self, client) -> None:
        c, _ = client
        r = c.post("/system/run_shell", json={"command": "echo hello"})
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_traversal_write_returns_500(self, client) -> None:
        c, _ = client
        r = c.post("/system/write_file", json={"path": "../escape.txt", "content": "bad"})
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# /task/plan — session required, tests planning logic via mock planner
# ---------------------------------------------------------------------------

class TestTaskPlan:
    def test_plan_returns_steps(self, client_with_session) -> None:
        c, _, planner, _ = client_with_session
        planner.plan.return_value = [{"action": "navigate", "url": "https://example.com"}]
        r = c.post("/task/plan", json={"intent": "go to google and search cats"})
        assert r.status_code == 200
        body = r.json()
        assert "steps" in body
        assert "count" in body

    def test_plan_unknown_intent_422(self, client_with_session) -> None:
        c, _, planner, _ = client_with_session
        planner.plan.side_effect = ValueError("No template matched")
        r = c.post("/task/plan", json={"intent": "fly to mars"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /task/run — session required
# ---------------------------------------------------------------------------

class TestTaskRun:
    def test_run_success_returns_200(self, client_with_session) -> None:
        c, agent, planner, _ = client_with_session
        steps = [{"action": "navigate", "url": "https://example.com"}]
        planner.plan.return_value = steps
        planner.execute.return_value = [{"status": "ok", "result": {}}]
        r = c.post("/task/run", json={"intent": "go to google and search python"})
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["failed_count"] == 0

    def test_run_failure_returns_422(self, client_with_session) -> None:
        c, agent, planner, _ = client_with_session
        steps = [{"action": "click", "selector": "#missing"}]
        planner.plan.return_value = steps
        planner.execute.return_value = [{"status": "error", "error": "element not found"}]
        r = c.post("/task/run", json={"intent": "click something"})
        assert r.status_code == 422
        body = r.json()
        assert body["detail"]["success"] is False
        assert body["detail"]["failed_count"] == 1

    def test_run_bad_intent_returns_422(self, client_with_session) -> None:
        c, _, planner, _ = client_with_session
        planner.plan.side_effect = ValueError("No template matched")
        r = c.post("/task/run", json={"intent": "unknowable intent xyz"})
        assert r.status_code == 422

    def test_run_with_log_path_writes_file(self, client_with_session, tmp_path: Path) -> None:
        c, agent, planner, tools = client_with_session
        steps = [{"action": "navigate", "url": "https://example.com"}]
        planner.plan.return_value = steps
        planner.execute.return_value = [{"status": "ok", "result": {}}]
        r = c.post("/task/run", json={"intent": "open example.com", "log_path": "run.log"})
        assert r.status_code == 200
        # Log file should have been written to the workspace
        log_content = (tmp_path / "run.log").read_text()
        import json
        log = json.loads(log_content)
        assert "intent" in log
        assert "success" in log
        assert "step_count" in log


# ---------------------------------------------------------------------------
# /task/execute — pre-built step list
# ---------------------------------------------------------------------------

class TestTaskExecute:
    def test_execute_valid_steps(self, client_with_session) -> None:
        c, agent, planner, _ = client_with_session
        planner.execute.return_value = [{"status": "ok", "result": {}}]
        r = c.post(
            "/task/execute",
            json={
                "steps": [{"action": "navigate", "url": "https://example.com"}],
            },
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_execute_invalid_steps_422(self, client_with_session) -> None:
        c, *_ = client_with_session
        r = c.post(
            "/task/execute",
            json={"steps": [{"action": "fly_to_mars"}]},
        )
        assert r.status_code == 422
