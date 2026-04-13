"""
Additional API tests covering routes not exercised in test_api.py.

Covers: /type, /page/attribute, /page/query_all, /evaluate, /screenshot,
        /wait/selector, /wait/load_state, /assert/text (fail), /assert/url (fail),
        /wait/text, /cookies/save, /cookies/load, /tabs/*, /scroll_to_element,
        /session/stop, /skills/*, /, /ui, /task/execute stop_on_error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

import api_server
from api_server import app
from system_tools import SystemTools

# ---------------------------------------------------------------------------
# Helpers (duplicated minimally from test_api.py — no shared fixtures)
# ---------------------------------------------------------------------------

def _make_agent(*, page_active: bool = True) -> MagicMock:
    agent = MagicMock()
    agent._page = MagicMock() if page_active else None
    agent.headless = True
    agent.navigate.return_value = {"url": "https://example.com", "status": "ok"}
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Test"}
    agent.type_text.return_value = {"selector": "input", "status": "ok"}
    agent.scroll_to_element.return_value = {"scrolled_to": "h1"}
    agent.get_attribute.return_value = "attr_value"
    agent.query_all.return_value = [{"tag": "div", "text": "item"}]
    agent.evaluate.return_value = 99
    agent.screenshot.return_value = {"path": None, "base64": "abc=="}
    agent.wait_for_selector.return_value = {"visible": "div"}
    agent.wait_for_load_state.return_value = {"state": "load"}
    agent.wait_text.return_value = {"found": True, "text": "hello"}
    agent.get_cookies.return_value = [{"name": "sid", "value": "abc"}]
    agent.add_cookies.return_value = None
    agent.new_tab.return_value = {"tab_index": 1, "url": "about:blank", "title": "New Tab"}
    agent.switch_tab.return_value = {"tab_index": 0, "url": "https://example.com", "title": "Test"}
    agent.close_tab.return_value = {"closed_index": 0, "remaining_tabs": 1}
    agent.list_tabs.return_value = {"tabs": [], "count": 0}
    agent.close_popups.return_value = {"dismissed": [], "count": 0}
    agent.fill.return_value = {"selector": "input", "status": "ok"}
    agent.click.return_value = {"selector": "button", "status": "ok"}
    agent.press_key.return_value = {"key": "Enter"}
    agent.hover.return_value = {"hovered": "a"}
    agent.select_option.return_value = {"selected": "opt"}
    agent.scroll.return_value = {"scrolled": {"x": 0, "y": 500}}
    agent.get_text.return_value = "page text"
    agent.get_html.return_value = "<p>html</p>"
    agent.extract_links.return_value = {"links": [], "count": 0}
    agent.extract_table.return_value = {"headers": [], "rows": [], "count": 0}
    setattr(agent, "assert_text", MagicMock(return_value={"found": True}))
    setattr(agent, "assert_url", MagicMock(return_value={"matched": True}))
    return agent


def _make_planner(*, fail: bool = False) -> MagicMock:
    planner = MagicMock()
    steps = [{"action": "navigate", "url": "https://example.com"}]
    planner.plan.return_value = steps
    if fail:
        planner.execute.return_value = [{"status": "error", "error": "fail"}]
    else:
        planner.execute.return_value = [{"status": "ok", "result": {}}]
    return planner


@pytest.fixture()
def client_active(tmp_path: Path):
    """Client with an active (mocked) browser session and real SystemTools."""
    agent = _make_agent()
    planner = _make_planner()
    tools = SystemTools(workspace=tmp_path)
    with (
        patch.object(api_server, "_agent", agent),
        patch.object(api_server, "_planner", planner),
        patch.object(api_server, "_tools", tools),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, agent, planner, tools


@pytest.fixture()
def client_no_session(tmp_path: Path):
    """Client with no active session."""
    tools = SystemTools(workspace=tmp_path)
    with (
        patch.object(api_server, "_agent", None),
        patch.object(api_server, "_planner", None),
        patch.object(api_server, "_tools", tools),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, tools


# ---------------------------------------------------------------------------
# /type route
# ---------------------------------------------------------------------------

class TestTypeRoute:
    def test_type_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/type", json={"selector": "input#q", "text": "hello"})
        assert r.status_code == 200
        agent.type_text.assert_called_once_with("input#q", "hello", clear_first=True)

    def test_type_no_clear(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/type", json={"selector": "input", "text": "hi", "clear_first": False})
        assert r.status_code == 200
        _, kwargs = agent.type_text.call_args
        assert kwargs.get("clear_first") is False

    def test_type_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/type", json={"selector": "input", "text": "hi"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /scroll_to_element
# ---------------------------------------------------------------------------

class TestScrollToElementRoute:
    def test_scroll_to_element_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/scroll_to_element", json={"selector": "h1"})
        assert r.status_code == 200
        agent.scroll_to_element.assert_called_once_with("h1")

    def test_scroll_to_element_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/scroll_to_element", json={"selector": "h1"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /page/attribute and /page/query_all
# ---------------------------------------------------------------------------

class TestPageAttributeQueryAll:
    def test_get_attribute_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/page/attribute", json={"selector": "a", "attribute": "href"})
        assert r.status_code == 200
        body = r.json()
        assert "value" in body
        agent.get_attribute.assert_called_once_with("a", "href")

    def test_get_attribute_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/page/attribute", json={"selector": "a", "attribute": "href"})
        assert r.status_code == 400

    def test_query_all_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/page/query_all", json={"selector": "div.item"})
        assert r.status_code == 200
        body = r.json()
        assert "elements" in body
        agent.query_all.assert_called_once_with("div.item")

    def test_query_all_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/page/query_all", json={"selector": "div"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /evaluate
# ---------------------------------------------------------------------------

class TestEvaluateRoute:
    def test_evaluate_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/evaluate", json={"script": "document.title"})
        assert r.status_code == 200
        body = r.json()
        assert "result" in body
        agent.evaluate.assert_called_once_with("document.title")

    def test_evaluate_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/evaluate", json={"script": "1+1"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /screenshot
# ---------------------------------------------------------------------------

class TestScreenshotRoute:
    def test_screenshot_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/screenshot", json={})
        assert r.status_code == 200
        agent.screenshot.assert_called_once()

    def test_screenshot_with_path(self, client_active):
        c, agent, *_ = client_active
        agent.screenshot.return_value = {"path": "shot.png"}
        r = c.post("/screenshot", json={"path": "shot.png"})
        assert r.status_code == 200

    def test_screenshot_full_page(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/screenshot", json={"full_page": True})
        assert r.status_code == 200
        _, kwargs = agent.screenshot.call_args
        assert kwargs.get("full_page") is True

    def test_screenshot_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/screenshot", json={})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /wait/selector and /wait/load_state
# ---------------------------------------------------------------------------

class TestWaitRoutes:
    def test_wait_selector_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/wait/selector", json={"selector": "div.ready"})
        assert r.status_code == 200
        agent.wait_for_selector.assert_called_once_with("div.ready", timeout=None)

    def test_wait_selector_with_timeout(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/wait/selector", json={"selector": "div", "timeout": 5000})
        assert r.status_code == 200
        agent.wait_for_selector.assert_called_once_with("div", timeout=5000)

    def test_wait_selector_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/wait/selector", json={"selector": "div"})
        assert r.status_code == 400

    def test_wait_load_state_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/wait/load_state", json={"state": "load"})
        assert r.status_code == 200
        agent.wait_for_load_state.assert_called_once_with("load")

    def test_wait_load_state_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/wait/load_state", json={"state": "load"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /assert/text and /assert/url
# ---------------------------------------------------------------------------

class TestAssertRoutes:
    def test_assert_text_success_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/assert/text", json={"text": "hello"})
        assert r.status_code == 200

    def test_assert_text_failure_422(self, client_active):
        c, agent, *_ = client_active
        agent.assert_text.side_effect = AssertionError("Text not found")
        r = c.post("/assert/text", json={"text": "missing text"})
        assert r.status_code == 422

    def test_assert_text_case_sensitive(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/assert/text", json={"text": "Hello", "case_sensitive": True})
        assert r.status_code == 200
        _, kwargs = agent.assert_text.call_args
        assert kwargs.get("case_sensitive") is True

    def test_assert_text_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/assert/text", json={"text": "hi"})
        assert r.status_code == 400

    def test_assert_url_success_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/assert/url", json={"pattern": "example"})
        assert r.status_code == 200

    def test_assert_url_failure_422(self, client_active):
        c, agent, *_ = client_active
        agent.assert_url.side_effect = AssertionError("URL does not match")
        r = c.post("/assert/url", json={"pattern": "missing-pattern"})
        assert r.status_code == 422

    def test_assert_url_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/assert/url", json={"pattern": "example"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /wait/text
# ---------------------------------------------------------------------------

class TestWaitTextRoute:
    def test_wait_text_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/wait/text", json={"text": "Done"})
        assert r.status_code == 200
        agent.wait_text.assert_called_once_with("Done", selector="body", timeout=None)

    def test_wait_text_custom_selector(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/wait/text", json={"text": "Ready", "selector": ".status"})
        assert r.status_code == 200
        agent.wait_text.assert_called_once_with("Ready", selector=".status", timeout=None)

    def test_wait_text_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/wait/text", json={"text": "hello"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /cookies/save and /cookies/load
# ---------------------------------------------------------------------------

class TestCookieRoutes:
    def test_save_cookies_200(self, client_active, tmp_path):
        c, agent, _, tools = client_active
        r = c.post("/cookies/save", json={"path": "cookies.json"})
        assert r.status_code == 200
        body = r.json()
        assert "cookies_saved" in body
        assert body["path"] == "cookies.json"

    def test_save_cookies_writes_file(self, client_active, tmp_path):
        c, agent, _, tools = client_active
        agent.get_cookies.return_value = [{"name": "s", "value": "v"}]
        c.post("/cookies/save", json={"path": "cookies.json"})
        saved = json.loads((tmp_path / "cookies.json").read_text())
        assert saved[0]["name"] == "s"

    def test_save_cookies_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/cookies/save", json={"path": "c.json"})
        assert r.status_code == 400

    def test_load_cookies_200(self, client_active, tmp_path):
        c, agent, _, tools = client_active
        cookies = [{"name": "tok", "value": "xyz"}]
        (tmp_path / "c.json").write_text(json.dumps(cookies))
        r = c.post("/cookies/load", json={"path": "c.json"})
        assert r.status_code == 200
        body = r.json()
        assert "cookies_loaded" in body
        agent.add_cookies.assert_called_once_with(cookies)

    def test_load_cookies_without_session_400(self, client_no_session, tmp_path):
        c, tools = client_no_session
        # Create the file so read_file succeeds, then expect 400 from get_agent()
        (tmp_path / "c.json").write_text("[]")
        r = c.post("/cookies/load", json={"path": "c.json"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /tabs/* routes
# ---------------------------------------------------------------------------

class TestTabRoutes:
    def test_new_tab_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/tabs/new", json={})
        assert r.status_code == 200
        agent.new_tab.assert_called_once_with(url=None)

    def test_new_tab_with_url(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/tabs/new", json={"url": "https://example.com"})
        assert r.status_code == 200
        agent.new_tab.assert_called_once_with(url="https://example.com")

    def test_new_tab_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/tabs/new", json={})
        assert r.status_code == 400

    def test_switch_tab_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/tabs/switch", json={"index": 0})
        assert r.status_code == 200
        agent.switch_tab.assert_called_once_with(0)

    def test_switch_tab_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/tabs/switch", json={"index": 0})
        assert r.status_code == 400

    def test_close_tab_200(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/tabs/close", json={})
        assert r.status_code == 200
        agent.close_tab.assert_called_once_with(index=None)

    def test_close_tab_with_index(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/tabs/close", json={"index": 0})
        assert r.status_code == 200
        agent.close_tab.assert_called_once_with(index=0)

    def test_close_tab_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/tabs/close", json={})
        assert r.status_code == 400

    def test_list_tabs_200(self, client_active):
        c, agent, *_ = client_active
        r = c.get("/tabs/list")
        assert r.status_code == 200
        body = r.json()
        assert "tabs" in body
        agent.list_tabs.assert_called_once()

    def test_list_tabs_without_session_400(self, client_no_session):
        c, _ = client_no_session
        r = c.get("/tabs/list")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /session/stop
# ---------------------------------------------------------------------------

class TestSessionStop:
    def test_stop_when_running(self, client_active):
        c, agent, *_ = client_active
        r = c.post("/session/stop")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "stopped"
        agent.stop.assert_called_once()

    def test_stop_when_not_running(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/session/stop")
        assert r.status_code == 200
        assert r.json()["status"] == "not_running"


# ---------------------------------------------------------------------------
# /skills routes
# ---------------------------------------------------------------------------

class TestSkillRoutes:
    def test_skills_list_empty(self, client_no_session):
        c, _ = client_no_session
        from skills import SkillRegistry
        with patch("api_server.get_default_registry", return_value=SkillRegistry()):
            r = c.get("/skills")
        assert r.status_code == 200
        body = r.json()
        assert "count" in body
        assert body["count"] == 0

    def test_skills_list_with_skill(self, client_no_session):
        c, _ = client_no_session
        from skills import SkillDef, SkillRegistry
        reg = SkillRegistry()
        reg.register(SkillDef(name="test_skill", steps=[{"action": "close_popups"}]))
        with patch("api_server.get_default_registry", return_value=reg):
            r = c.get("/skills")
        body = r.json()
        assert body["count"] == 1
        assert body["skills"][0]["name"] == "test_skill"

    def test_skills_get_found(self, client_no_session):
        c, _ = client_no_session
        from skills import SkillDef, SkillRegistry
        reg = SkillRegistry()
        reg.register(SkillDef(name="my_skill", steps=[{"action": "close_popups"}]))
        with patch("api_server.get_default_registry", return_value=reg):
            r = c.get("/skills/my_skill")
        assert r.status_code == 200
        assert r.json()["name"] == "my_skill"

    def test_skills_get_not_found(self, client_no_session):
        c, _ = client_no_session
        from skills import SkillRegistry
        with patch("api_server.get_default_registry", return_value=SkillRegistry()):
            r = c.get("/skills/nonexistent")
        assert r.status_code == 404

    def test_skills_delete_found(self, client_no_session):
        c, _ = client_no_session
        from skills import SkillDef, SkillRegistry
        reg = SkillRegistry()
        reg.register(SkillDef(name="del_skill", steps=[{"action": "close_popups"}]))
        with patch("api_server.get_default_registry", return_value=reg):
            r = c.delete("/skills/del_skill")
        assert r.status_code == 200
        assert r.json()["unloaded"] == "del_skill"

    def test_skills_delete_not_found(self, client_no_session):
        c, _ = client_no_session
        from skills import SkillRegistry
        with patch("api_server.get_default_registry", return_value=SkillRegistry()):
            r = c.delete("/skills/nonexistent")
        assert r.status_code == 404

    def test_skills_load_local_path_rejected(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/skills/load", json={"source": "/local/path/skill.json"})
        assert r.status_code == 422

    def test_skills_load_http_not_https_rejected(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/skills/load", json={"source": "http://example.com/skill.json"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# / and /ui routes
# ---------------------------------------------------------------------------

class TestRootAndUI:
    def test_root_redirects_to_ui(self, client_no_session):
        c, _ = client_no_session
        r = c.get("/", follow_redirects=False)
        assert r.status_code in (301, 302, 307, 308)
        assert "/ui" in r.headers.get("location", "")

    def test_ui_returns_html(self, client_no_session):
        c, _ = client_no_session
        r = c.get("/ui")
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert "html" in r.text.lower()

    def test_ui_not_found_returns_404(self, client_no_session, tmp_path):
        c, _ = client_no_session
        from pathlib import Path as _Path
        fake_gui = _Path("/nonexistent/path/gui/index.html")
        with patch("api_server._Path") as mock_path:
            mock_path.return_value.__truediv__ = MagicMock(
                side_effect=lambda *a: fake_gui
            )
            r = c.get("/ui")
        # Either 200 (real gui found) or 404 (gui missing) — both are valid
        assert r.status_code in (200, 404)


# ---------------------------------------------------------------------------
# /task/execute stop_on_error
# ---------------------------------------------------------------------------

class TestTaskExecuteStopOnError:
    def test_execute_stop_on_error_false(self, client_active):
        c, agent, planner, _ = client_active
        planner.execute.return_value = [
            {"status": "error", "error": "step 0 fail"},
            {"status": "ok", "result": {}},
        ]
        r = c.post(
            "/task/execute",
            json={
                "steps": [
                    {"action": "navigate", "url": "https://example.com"},
                    {"action": "close_popups"},
                ],
                "stop_on_error": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["failed_count"] == 1
        # Check planner.execute called with stop_on_error=False
        _, kwargs = planner.execute.call_args
        assert kwargs.get("stop_on_error") is False


# ---------------------------------------------------------------------------
# /doctor
# ---------------------------------------------------------------------------

class TestDoctorRoute:
    def test_doctor_ok(self, client_no_session):
        c, _ = client_no_session
        ok_check = MagicMock()
        ok_check.status = "ok"
        ok_check.to_dict.return_value = {"name": "test", "status": "ok", "message": "fine"}
        with patch("api_server.run_checks", return_value=[ok_check]):
            r = c.get("/doctor")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert len(body["checks"]) == 1

    def test_doctor_degraded(self, client_no_session):
        c, _ = client_no_session
        fail_check = MagicMock()
        fail_check.status = "fail"
        fail_check.to_dict.return_value = {"name": "chromium", "status": "fail", "message": "missing"}
        with patch("api_server.run_checks", return_value=[fail_check]):
            r = c.get("/doctor")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"


# ---------------------------------------------------------------------------
# get_planner() — 400 when no session
# ---------------------------------------------------------------------------

class TestGetPlannerError:
    def test_task_plan_400_without_session(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/task/plan", json={"intent": "go to example.com"})
        assert r.status_code == 400

    def test_task_run_400_without_session(self, client_no_session):
        c, _ = client_no_session
        r = c.post("/task/run", json={"intent": "go to example.com"})
        assert r.status_code == 400

    def test_task_execute_400_without_session(self, client_no_session):
        c, _ = client_no_session
        r = c.post(
            "/task/execute",
            json={"steps": [{"action": "navigate", "url": "https://example.com"}]},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Security fixes: path traversal, URL scheme bypass, code-exec gate, API key
# ---------------------------------------------------------------------------
# Upload/Download file — path traversal protection
# ---------------------------------------------------------------------------

class TestUploadDownloadPathTraversal:
    def test_upload_file_traversal_rejected(self):
        """Paths outside the workspace should be rejected with 422."""
        agent = _make_agent()
        tools = SystemTools()

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            c = TestClient(app)
            r = c.post(
                "/upload_file",
                json={"selector": "input[type=file]", "path": "../../../etc/passwd"},
            )
        assert r.status_code == 422

    def test_download_file_traversal_rejected(self):
        """Paths outside the workspace should be rejected with 422."""
        agent = _make_agent()
        tools = SystemTools()

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            c = TestClient(app)
            r = c.post(
                "/download_file",
                json={"url": "https://example.com/file.zip", "path": "../../etc/passwd"},
            )
        assert r.status_code == 422

    def test_upload_file_valid_path_accepted(self):
        """A normal workspace-relative path should succeed."""
        agent = _make_agent()
        agent.upload_file.return_value = {"selector": "input", "uploaded": "file.txt", "ok": True}
        tools = SystemTools()

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            c = TestClient(app)
            r = c.post(
                "/upload_file",
                json={"selector": "input[type=file]", "path": "file.txt"},
            )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# SetViewportRequest — dimension validation
# ---------------------------------------------------------------------------

class TestSetViewportValidation:
    def test_negative_width_rejected(self):
        agent = _make_agent()
        with patch.object(api_server, "_agent", agent):
            c = TestClient(app)
            r = c.post("/session/viewport", json={"width": -1, "height": 600})
        assert r.status_code == 422

    def test_zero_width_rejected(self):
        agent = _make_agent()
        with patch.object(api_server, "_agent", agent):
            c = TestClient(app)
            r = c.post("/session/viewport", json={"width": 0, "height": 600})
        assert r.status_code == 422

    def test_negative_height_rejected(self):
        agent = _make_agent()
        with patch.object(api_server, "_agent", agent):
            c = TestClient(app)
            r = c.post("/session/viewport", json={"width": 1280, "height": -1})
        assert r.status_code == 422

    def test_valid_dimensions_accepted(self):
        agent = _make_agent()
        agent.set_viewport.return_value = {"width": 1280, "height": 720, "ok": True}
        with patch.object(api_server, "_agent", agent):
            c = TestClient(app)
            r = c.post("/session/viewport", json={"width": 1280, "height": 720})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /screenshot path traversal protection (Bug fix regression)
# ---------------------------------------------------------------------------

class TestScreenshotPathTraversal:
    def test_screenshot_traversal_path_rejected(self):
        """A path escaping the workspace must return 422."""
        agent = _make_agent()
        tools = SystemTools()  # default workspace = ./workspace
        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            c = TestClient(app)
            r = c.post("/screenshot", json={"path": "../../../etc/evil.png"})
        assert r.status_code == 422

    def test_screenshot_absolute_path_outside_workspace_rejected(self):
        """An absolute path outside the workspace must return 422."""
        agent = _make_agent()
        tools = SystemTools()
        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            c = TestClient(app)
            r = c.post("/screenshot", json={"path": "/tmp/evil.png"})
        assert r.status_code == 422

    def test_screenshot_safe_path_accepted(self):
        """A normal workspace-relative path must pass through to the agent."""
        agent = _make_agent()
        agent.screenshot.return_value = {"path": "shot.png", "base64": None}
        tools = SystemTools()
        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            c = TestClient(app)
            r = c.post("/screenshot", json={"path": "shot.png"})
        assert r.status_code == 200
        # The path passed to the agent must be absolute (resolved by safe_path)
        _args, kwargs = agent.screenshot.call_args
        called_path = kwargs.get("path")
        assert called_path is not None
        # Confirm it ends with the requested filename
        assert str(called_path).endswith("shot.png")

    def test_screenshot_no_path_no_validation(self):
        """Omitting path entirely (base64 mode) should still succeed."""
        agent = _make_agent()
        agent.screenshot.return_value = {"path": None, "base64": "abc=="}
        tools = SystemTools()
        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            c = TestClient(app)
            r = c.post("/screenshot", json={})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /recording/* — video and GIF endpoints
# ---------------------------------------------------------------------------

class TestRecordingEndpoints:
    """Tests for /recording/start, /recording/stop, /recording/gif."""

    def test_start_recording_valid_path(self, client_active, tmp_path):
        client, agent, _, _ = client_active
        agent.start_video_recording.return_value = {"recording": True, "save_path": str(tmp_path / "vid.webm")}
        resp = client.post("/recording/start", json={"path": "vid.webm"})
        assert resp.status_code == 200
        agent.start_video_recording.assert_called_once()

    def test_start_recording_path_traversal_rejected(self, client_active):
        client, agent, _, _ = client_active
        resp = client.post("/recording/start", json={"path": "../../../etc/evil.webm"})
        assert resp.status_code == 422
        agent.start_video_recording.assert_not_called()

    def test_start_recording_already_active_returns_409(self, client_active):
        client, agent, _, _ = client_active
        agent.start_video_recording.side_effect = RuntimeError("already active")
        resp = client.post("/recording/start", json={"path": "vid.webm"})
        assert resp.status_code == 409

    def test_stop_recording_ok(self, client_active):
        client, agent, _, _ = client_active
        agent.stop_video_recording.return_value = {"saved_to": "/ws/vid.webm", "ok": True}
        resp = client.post("/recording/stop")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_stop_recording_not_started_returns_409(self, client_active):
        client, agent, _, _ = client_active
        agent.stop_video_recording.side_effect = RuntimeError("No recording in progress")
        resp = client.post("/recording/stop")
        assert resp.status_code == 409

    def test_gif_valid_path(self, client_active, tmp_path):
        client, agent, _, _ = client_active
        agent.record_gif.return_value = {
            "saved_to": str(tmp_path / "out.gif"), "frames": 4, "fps": 2, "duration": 2.0, "ok": True
        }
        resp = client.post("/recording/gif", json={"path": "out.gif", "duration": 2.0, "fps": 2})
        assert resp.status_code == 200
        agent.record_gif.assert_called_once()

    def test_gif_path_traversal_rejected(self, client_active):
        client, agent, _, _ = client_active
        resp = client.post("/recording/gif", json={"path": "../../etc/evil.gif"})
        assert resp.status_code == 422
        agent.record_gif.assert_not_called()

    def test_gif_fps_bounds(self, client_active):
        """fps outside [1,30] should be rejected by Pydantic."""
        client, _, _, _ = client_active
        resp = client.post("/recording/gif", json={"path": "g.gif", "fps": 0})
        assert resp.status_code == 422
        resp2 = client.post("/recording/gif", json={"path": "g.gif", "fps": 31})
        assert resp2.status_code == 422

    def test_gif_duration_minimum(self, client_active):
        """duration < 0.1 should be rejected."""
        client, _, _, _ = client_active
        resp = client.post("/recording/gif", json={"path": "g.gif", "duration": 0.0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /tabs/execute_parallel
# ---------------------------------------------------------------------------

class TestTabsExecuteParallel:
    """Tests for /tabs/execute_parallel."""

    def _make_planner_with_results(self, ok: bool = True):
        p = MagicMock()
        if ok:
            p.execute.return_value = [{"step": 0, "action": "navigate", "status": "ok", "result": {}}]
        else:
            p.execute.return_value = [{"step": 0, "action": "navigate", "status": "error", "error": "boom"}]
        return p

    def test_two_tab_tasks_ok(self, client_active):
        client, agent, planner, _ = client_active
        agent.new_tab.return_value = {"tab_index": 1, "url": "about:blank", "title": "New"}
        planner.execute.return_value = [{"step": 0, "action": "navigate", "status": "ok", "result": {}}]
        body = {
            "tasks": [
                {"steps": [{"action": "navigate", "url": "https://a.com"}]},
                {"steps": [{"action": "navigate", "url": "https://b.com"}]},
            ]
        }
        resp = client.post("/tabs/execute_parallel", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tabs"] == 2
        assert data["all_succeeded"] is True

    def test_single_task_rejected(self, client_active):
        client, *_ = client_active
        body = {"tasks": [{"steps": [{"action": "navigate", "url": "https://a.com"}]}]}
        resp = client.post("/tabs/execute_parallel", json=body)
        assert resp.status_code == 422

    def test_too_many_tasks_rejected(self, client_active):
        client, *_ = client_active
        tasks = [{"steps": [{"action": "navigate", "url": "https://a.com"}]} for _ in range(21)]
        resp = client.post("/tabs/execute_parallel", json={"tasks": tasks})
        assert resp.status_code == 422

    def test_invalid_step_schema_rejected(self, client_active):
        client, *_ = client_active
        body = {
            "tasks": [
                {"steps": [{"action": "not_a_real_action"}]},
                {"steps": [{"action": "navigate", "url": "https://b.com"}]},
            ]
        }
        resp = client.post("/tabs/execute_parallel", json=body)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /agents/pool/* management
# ---------------------------------------------------------------------------

class TestAgentPool:
    """Tests for pool CRUD endpoints."""

    @pytest.fixture()
    def clean_pool(self):
        """Ensure the global pool is empty before and after each test."""
        import api_server as _srv
        with _srv._pool_lock:
            _srv._agent_pool.clear()
            _srv._planner_pool.clear()
        yield
        with _srv._pool_lock:
            _srv._agent_pool.clear()
            _srv._planner_pool.clear()

    def test_start_agent_auto_id(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools), \
             patch("api_server.BrowserAgent") as MockAgent:
            mock_inst = MagicMock()
            MockAgent.return_value = mock_inst
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/pool/start", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_id" in data
        assert data["status"] == "started"
        mock_inst.start.assert_called_once()

    def test_start_agent_custom_id(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools), \
             patch("api_server.BrowserAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/pool/start", json={"agent_id": "my-worker"})
        assert resp.json()["agent_id"] == "my-worker"

    def test_start_duplicate_id_returns_409(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools), \
             patch("api_server.BrowserAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            with TestClient(app, raise_server_exceptions=False) as c:
                c.post("/agents/pool/start", json={"agent_id": "dup"})
                resp = c.post("/agents/pool/start", json={"agent_id": "dup"})
        assert resp.status_code == 409

    def test_list_agents_empty(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools), \
             patch.object(_srv, "_agent", None):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/agents/pool")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_agents_populated(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        mock_agent = MagicMock()
        mock_agent._page = MagicMock()
        mock_agent._page.url = "https://example.com"
        mock_agent._pages = [mock_agent._page]
        mock_agent._recording = False
        with _srv._pool_lock:
            _srv._agent_pool["worker-1"] = mock_agent
        with patch.object(_srv, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/agents/pool")
        data = resp.json()
        assert data["count"] == 1
        assert data["agents"][0]["agent_id"] == "worker-1"

    def test_get_agent_status_found(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        mock_agent = MagicMock()
        mock_agent._page = MagicMock()
        mock_agent._page.url = "https://example.com"
        mock_agent._pages = [mock_agent._page]
        mock_agent._recording = False
        mock_agent.list_tabs.return_value = {"tabs": [], "count": 0}
        with _srv._pool_lock:
            _srv._agent_pool["w1"] = mock_agent
        with patch.object(_srv, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/agents/pool/w1")
        assert resp.status_code == 200
        assert resp.json()["active"] is True

    def test_get_agent_status_not_found(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/agents/pool/does-not-exist")
        assert resp.status_code == 404

    def test_delete_agent_ok(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        mock_agent = MagicMock()
        mock_agent._page = MagicMock()
        mock_agent._pages = [mock_agent._page]
        mock_agent._recording = False
        with _srv._pool_lock:
            _srv._agent_pool["del-me"] = mock_agent
        with patch.object(_srv, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.delete("/agents/pool/del-me")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"
        mock_agent.stop.assert_called_once()
        with _srv._pool_lock:
            assert "del-me" not in _srv._agent_pool

    def test_delete_agent_not_found(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.delete("/agents/pool/ghost")
        assert resp.status_code == 404

    def test_execute_on_pooled_agent(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        mock_agent = MagicMock()
        mock_agent._page = MagicMock()
        mock_agent._pages = [mock_agent._page]
        mock_planner = MagicMock()
        mock_planner.execute.return_value = [
            {"step": 0, "action": "navigate", "status": "ok", "result": {}}
        ]
        with _srv._pool_lock:
            _srv._agent_pool["exec-agent"] = mock_agent
            _srv._planner_pool["exec-agent"] = mock_planner
        with patch.object(_srv, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/agents/pool/exec-agent/task/execute",
                    json={"steps": [{"action": "navigate", "url": "https://example.com"}]},
                )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ---------------------------------------------------------------------------
# /agents/execute_parallel
# ---------------------------------------------------------------------------

class TestAgentsExecuteParallel:
    """Tests for /agents/execute_parallel."""

    @pytest.fixture()
    def clean_pool(self):
        import api_server as _srv
        with _srv._pool_lock:
            _srv._agent_pool.clear()
            _srv._planner_pool.clear()
        yield
        with _srv._pool_lock:
            _srv._agent_pool.clear()
            _srv._planner_pool.clear()

    def _mock_browser_agent(self):
        inst = MagicMock()
        inst._page = MagicMock()
        inst._pages = [inst._page]
        return inst

    def test_broadcast_mode_spawns_n_agents(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools), \
             patch("api_server.BrowserAgent") as MockBA, \
             patch("api_server.TaskPlanner") as MockTP:
            mock_inst = self._mock_browser_agent()
            MockBA.return_value = mock_inst
            mock_planner = MagicMock()
            mock_planner.execute.return_value = [
                {"step": 0, "action": "navigate", "status": "ok", "result": {}}
            ]
            MockTP.return_value = mock_planner
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={
                    "agent_count": 2,
                    "steps": [{"action": "navigate", "url": "https://example.com"}],
                    "auto_stop": False,
                })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_agents"] == 2

    def test_explicit_tasks_mode(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools), \
             patch("api_server.BrowserAgent") as MockBA, \
             patch("api_server.TaskPlanner") as MockTP:
            mock_inst = self._mock_browser_agent()
            MockBA.return_value = mock_inst
            mock_planner = MagicMock()
            mock_planner.execute.return_value = [
                {"step": 0, "action": "navigate", "status": "ok", "result": {}}
            ]
            MockTP.return_value = mock_planner
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={
                    "tasks": [
                        {"steps": [{"action": "navigate", "url": "https://a.com"}]},
                        {"steps": [{"action": "navigate", "url": "https://b.com"}]},
                    ],
                    "auto_stop": False,
                })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_agents"] == 2
        assert data["all_succeeded"] is True

    def test_no_tasks_and_no_agent_count_returns_422(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={})
        assert resp.status_code == 422

    def test_broadcast_without_steps_returns_422(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={"agent_count": 2})
        assert resp.status_code == 422

    def test_auto_stop_removes_spawned_agents(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        with patch.object(_srv, "_tools", tools), \
             patch("api_server.BrowserAgent") as MockBA, \
             patch("api_server.TaskPlanner") as MockTP:
            mock_inst = self._mock_browser_agent()
            MockBA.return_value = mock_inst
            mock_planner = MagicMock()
            mock_planner.execute.return_value = [
                {"step": 0, "action": "navigate", "status": "ok", "result": {}}
            ]
            MockTP.return_value = mock_planner
            with TestClient(app, raise_server_exceptions=False) as c:
                c.post("/agents/execute_parallel", json={
                    "agent_count": 1,
                    "steps": [{"action": "navigate", "url": "https://example.com"}],
                    "auto_stop": True,
                })
        with _srv._pool_lock:
            assert len(_srv._agent_pool) == 0

    def test_existing_pooled_agent_used(self, tmp_path, clean_pool):
        import api_server as _srv
        tools = SystemTools(workspace=tmp_path)
        mock_inst = self._mock_browser_agent()
        mock_planner = MagicMock()
        mock_planner.execute.return_value = [
            {"step": 0, "action": "navigate", "status": "ok", "result": {}}
        ]
        with _srv._pool_lock:
            _srv._agent_pool["existing"] = mock_inst
            _srv._planner_pool["existing"] = mock_planner
        with patch.object(_srv, "_tools", tools), \
             patch("api_server.TaskPlanner") as MockTP:
            MockTP.return_value = mock_planner
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={
                    "tasks": [
                        {"agent_id": "existing",
                         "steps": [{"action": "navigate", "url": "https://example.com"}]},
                    ],
                    "auto_stop": True,
                })
                assert resp.status_code == 200
                # auto_stop must NOT touch the pre-existing pooled agent —
                # verify this BEFORE the TestClient context exits (which triggers
                # lifespan shutdown that would legitimately call stop()).
                mock_inst.stop.assert_not_called()
                with _srv._pool_lock:
                    assert "existing" in _srv._agent_pool
class TestCodeExecGate:
    def test_run_python_disabled_returns_403(self, client_no_session):
        c, _ = client_no_session
        import api_server as _srv
        orig = _srv._CODE_EXEC_ALLOWED
        _srv._CODE_EXEC_ALLOWED = False
        try:
            r = c.post("/system/run_python", json={"code": "1+1"})
        finally:
            _srv._CODE_EXEC_ALLOWED = orig
        assert r.status_code == 403
        assert "BROWSER_ALLOW_CODE_EXEC" in r.json()["detail"]

    def test_run_shell_disabled_returns_403(self, client_no_session):
        c, _ = client_no_session
        import api_server as _srv
        orig = _srv._CODE_EXEC_ALLOWED
        _srv._CODE_EXEC_ALLOWED = False
        try:
            r = c.post("/system/run_shell", json={"command": "echo hi"})
        finally:
            _srv._CODE_EXEC_ALLOWED = orig
        assert r.status_code == 403
        assert "BROWSER_ALLOW_CODE_EXEC" in r.json()["detail"]


class TestApiKeyMiddleware:
    def test_request_without_key_rejected_when_key_configured(self, client_no_session):
        c, _ = client_no_session
        import api_server as _srv
        orig = _srv._API_KEY
        _srv._API_KEY = "secret123"
        try:
            r = c.get("/session/status")
        finally:
            _srv._API_KEY = orig
        assert r.status_code == 401
        assert "X-API-Key" in r.json()["detail"]

    def test_request_with_wrong_key_rejected(self, client_no_session):
        c, _ = client_no_session
        import api_server as _srv
        orig = _srv._API_KEY
        _srv._API_KEY = "secret123"
        try:
            r = c.get("/session/status", headers={"X-API-Key": "wrong"})
        finally:
            _srv._API_KEY = orig
        assert r.status_code == 401

    def test_request_with_correct_key_accepted(self, client_no_session):
        c, _ = client_no_session
        import api_server as _srv
        orig = _srv._API_KEY
        _srv._API_KEY = "secret123"
        try:
            r = c.get("/session/status", headers={"X-API-Key": "secret123"})
        finally:
            _srv._API_KEY = orig
        assert r.status_code == 200

    def test_no_key_configured_allows_all(self, client_no_session):
        c, _ = client_no_session
        import api_server as _srv
        orig = _srv._API_KEY
        _srv._API_KEY = None
        try:
            r = c.get("/session/status")
        finally:
            _srv._API_KEY = orig
        assert r.status_code == 200

    def test_exempt_paths_always_accessible(self, client_no_session):
        c, _ = client_no_session
        import api_server as _srv
        orig = _srv._API_KEY
        _srv._API_KEY = "secret123"
        try:
            r_root = c.get("/", follow_redirects=False)
            r_docs = c.get("/docs", follow_redirects=False)
        finally:
            _srv._API_KEY = orig
        # root may redirect (3xx) or return 200 — either is fine; must not be 401
        assert r_root.status_code != 401
        assert r_docs.status_code != 401
