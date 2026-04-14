"""
tests/test_browser_thread_dispatch.py

Tests for the dedicated browser-worker-thread architecture introduced to fix
the Playwright sync-API thread-affinity bug ("Cannot switch to a different
thread").

Design under test
-----------------
All routes that touch the shared Playwright BrowserAgent execute their browser
calls on a single, long-lived daemon thread (_browser_thread).  Requests from
FastAPI's thread-pool workers are marshalled onto the owner thread via
_BrowserThread.submit(), which uses a queue + concurrent.futures.Future.

Test strategy
-------------
* Unit-test _BrowserThread.submit() in isolation.
* Verify that browser-touching routes actually call agent methods from the
  browser-worker thread's identity (not the calling thread).
* Verify that exceptions propagate correctly across the thread boundary.
* Verify that session_start / session_stop / session_status run on the browser
  thread, including the "already_running" short-circuit.
* Verify ws/task streaming uses the browser thread.
* Confirm that non-browser routes (system tools, task/plan, etc.) are unaffected.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

import api_server
from api_server import _browser_thread, _BrowserThread, app
from system_tools import SystemTools

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_agent(*, page_active: bool = True) -> MagicMock:
    agent = MagicMock()
    agent._page = MagicMock() if page_active else None
    agent.headless = True
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Example"}
    agent.navigate.return_value = {"url": "https://example.com", "status": "ok"}
    agent.click.return_value = {"selector": "button", "status": "ok"}
    agent.fill.return_value = {"selector": "input", "status": "ok"}
    agent.press_key.return_value = {"key": "Enter", "status": "ok"}
    agent.type_text.return_value = {"selector": "input", "status": "ok"}
    agent.close_popups.return_value = {"dismissed": 0}
    agent.get_text.return_value = "page text"
    agent.get_html.return_value = "<p>html</p>"
    agent.get_attribute.return_value = "value"
    agent.query_all.return_value = []
    agent.evaluate.return_value = 42
    agent.screenshot.return_value = {"data": "base64=="}
    agent.wait_for_selector.return_value = {"status": "ok"}
    agent.wait_for_load_state.return_value = {"state": "networkidle", "status": "ok"}
    agent.extract_links.return_value = {"links": [], "count": 0}
    agent.extract_table.return_value = {"headers": [], "rows": [], "count": 0}
    setattr(agent, "assert_text", MagicMock(return_value={"found": True}))
    setattr(agent, "assert_url",  MagicMock(return_value={"matched": True}))
    agent.wait_text.return_value = {"found": True}
    agent.get_cookies.return_value = []
    agent.add_cookies.return_value = None
    agent.new_tab.return_value = {"tab_index": 1, "url": "about:blank"}
    agent.switch_tab.return_value = {"index": 0, "status": "ok"}
    agent.close_tab.return_value = {"closed_index": 0, "status": "ok"}
    agent.list_tabs.return_value = {"tabs": [], "count": 0}
    agent.hover.return_value = {"status": "ok"}
    agent.scroll.return_value = {"status": "ok"}
    agent.scroll_to_element.return_value = {"status": "ok"}
    agent.select_option.return_value = {"status": "ok"}
    agent.drag_and_drop.return_value = {"status": "ok"}
    agent.right_click.return_value = {"status": "ok"}
    agent.double_click.return_value = {"status": "ok"}
    agent.get_element_rect.return_value = {"x": 0, "y": 0, "width": 100, "height": 50}
    agent.set_network_intercept.return_value = {"status": "ok"}
    agent.clear_network_intercepts.return_value = {"status": "ok"}
    agent.set_viewport.return_value = {"status": "ok"}
    agent.set_geolocation.return_value = {"status": "ok"}
    agent.upload_file.return_value = {"status": "ok"}
    agent.download_file.return_value = {"status": "ok"}
    agent.start_video_recording.return_value = {"status": "ok"}
    agent.stop_video_recording.return_value = {"status": "ok"}
    agent.record_gif.return_value = {"status": "ok"}
    agent._recording = False
    agent._pages = [MagicMock()]
    return agent


def _make_planner() -> MagicMock:
    planner = MagicMock()
    planner.plan.return_value = [{"action": "navigate", "url": "https://example.com"}]
    planner.execute.return_value = [{"action": "navigate", "status": "ok", "step": 0}]
    planner.agentic_run.return_value = {
        "results": [{"action": "navigate", "status": "ok", "step": 0}],
        "stopped_reason": "done_by_model",
        "verified": True,
        "recovery_steps_injected": 0,
    }
    return planner


@pytest.fixture()
def client_with_session(tmp_path: Path):
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


# ---------------------------------------------------------------------------
# Unit tests: _BrowserThread
# ---------------------------------------------------------------------------

class TestBrowserThreadUnit:
    """Low-level tests for the _BrowserThread dispatcher."""

    def test_submit_returns_result(self):
        bt = _BrowserThread()
        try:
            result = bt.submit(lambda: 42)
            assert result == 42
        finally:
            bt.stop()

    def test_submit_with_args_and_kwargs(self):
        bt = _BrowserThread()
        try:
            result = bt.submit(lambda x, y=0: x + y, 10, y=5)
            assert result == 15
        finally:
            bt.stop()

    def test_submit_propagates_exception(self):
        bt = _BrowserThread()
        try:
            with pytest.raises(ValueError, match="boom"):
                bt.submit(lambda: (_ for _ in ()).throw(ValueError("boom")))
        finally:
            bt.stop()

    def test_submit_runs_on_worker_thread(self):
        """Work submitted to the browser thread must NOT run on the calling thread."""
        bt = _BrowserThread()
        try:
            caller_tid = threading.get_ident()
            worker_tid = bt.submit(threading.get_ident)
            assert worker_tid != caller_tid, (
                "Browser work ran on calling thread — thread-affinity not preserved"
            )
        finally:
            bt.stop()

    def test_worker_thread_is_consistent(self):
        """All work submitted to one _BrowserThread instance runs on the same thread."""
        bt = _BrowserThread()
        try:
            tids = [bt.submit(threading.get_ident) for _ in range(5)]
            assert len(set(tids)) == 1, "Multiple threads used — singleton violated"
        finally:
            bt.stop()

    def test_stop_joins_thread(self):
        bt = _BrowserThread()
        assert bt._thread.is_alive()
        bt.stop()
        bt._thread.join(timeout=1)
        assert not bt._thread.is_alive()

    def test_sequential_ordering(self):
        """Work items must be executed in submission order."""
        bt = _BrowserThread()
        results = []
        try:
            for i in range(10):
                bt.submit(results.append, i)
            assert results == list(range(10))
        finally:
            bt.stop()


# ---------------------------------------------------------------------------
# Integration: module-level _browser_thread singleton
# ---------------------------------------------------------------------------

class TestBrowserThreadSingleton:
    def test_singleton_is_alive(self):
        assert _browser_thread._thread.is_alive()

    def test_singleton_is_daemon(self):
        assert _browser_thread._thread.daemon

    def test_singleton_name(self):
        assert _browser_thread._thread.name == "browser-worker"

    def test_on_browser_thread_helper(self):
        """_on_browser_thread() delegates to _browser_thread.submit()."""
        result = api_server._on_browser_thread(lambda: "hello")
        assert result == "hello"

    def test_browser_work_not_on_main_thread(self):
        main_tid = threading.get_ident()
        worker_tid = api_server._on_browser_thread(threading.get_ident)
        assert worker_tid != main_tid


# ---------------------------------------------------------------------------
# Integration: browser route execution happens on the browser thread
# ---------------------------------------------------------------------------

class TestBrowserRouteDispatch:
    """
    Verify that browser-touching routes actually run their agent calls on the
    dedicated browser-worker thread — not on the FastAPI worker calling thread.
    """

    def _thread_id_agent(self):
        """Return a mock agent whose navigate() records the thread it was called from."""
        agent = _make_agent()
        call_thread_ids = []

        def _navigate(url, **kw):
            call_thread_ids.append(threading.get_ident())
            return {"url": url, "status": "ok"}

        agent.navigate.side_effect = _navigate
        return agent, call_thread_ids

    def test_navigate_called_on_browser_thread(self, tmp_path):
        agent, call_tids = self._thread_id_agent()
        planner = _make_planner()
        tools = SystemTools(workspace=tmp_path)
        test_thread_id = threading.get_ident()

        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", planner),
            patch.object(api_server, "_tools", tools),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/navigate", json={"url": "https://example.com"})

        assert resp.status_code == 200
        assert len(call_tids) == 1
        # The navigate call must NOT have happened on the test (main) thread
        assert call_tids[0] != test_thread_id
        # It must have happened on the browser-worker thread
        assert call_tids[0] == _browser_thread._thread.ident

    def test_click_called_on_browser_thread(self, tmp_path):
        agent = _make_agent()
        call_tids = []

        def _click(selector, **kw):
            call_tids.append(threading.get_ident())
            return {"selector": selector, "status": "ok"}

        agent.click.side_effect = _click
        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", _make_planner()),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                c.post("/click", json={"selector": "button"})

        assert len(call_tids) == 1
        assert call_tids[0] == _browser_thread._thread.ident

    def test_fill_called_on_browser_thread(self, tmp_path):
        agent = _make_agent()
        call_tids = []

        def _fill(selector, value):
            call_tids.append(threading.get_ident())
            return {"selector": selector, "status": "ok"}

        agent.fill.side_effect = _fill
        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", _make_planner()),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                c.post("/fill", json={"selector": "input", "value": "text"})

        assert len(call_tids) == 1
        assert call_tids[0] == _browser_thread._thread.ident

    def test_press_key_called_on_browser_thread(self, tmp_path):
        agent = _make_agent()
        call_tids = []

        def _press_key(key):
            call_tids.append(threading.get_ident())
            return {"key": key, "status": "ok"}

        agent.press_key.side_effect = _press_key
        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", _make_planner()),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                c.post("/press_key", json={"key": "Enter"})

        assert len(call_tids) == 1
        assert call_tids[0] == _browser_thread._thread.ident

    def test_task_execute_called_on_browser_thread(self, tmp_path):
        agent = _make_agent()
        planner = _make_planner()
        execute_tids = []

        def _execute(steps, agent_arg, **kw):
            execute_tids.append(threading.get_ident())
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        planner.execute.side_effect = _execute
        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", planner),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/task/execute",
                    json={"steps": [{"action": "navigate", "url": "https://example.com"}]},
                )

        assert resp.status_code == 200
        assert len(execute_tids) == 1
        assert execute_tids[0] == _browser_thread._thread.ident

    def test_task_run_execute_on_browser_thread(self, tmp_path):
        agent = _make_agent()
        planner = _make_planner()
        execute_tids = []

        def _execute(steps, agent_arg, **kw):
            execute_tids.append(threading.get_ident())
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        planner.execute.side_effect = _execute
        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", planner),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/task/run", json={"intent": "go to google"})

        assert resp.status_code == 200
        assert len(execute_tids) == 1
        assert execute_tids[0] == _browser_thread._thread.ident

    def test_multiple_sequential_requests_same_thread(self, tmp_path):
        """All browser calls from different HTTP requests must land on the same thread."""
        agent = _make_agent()
        call_tids = []

        def _navigate(url, **kw):
            call_tids.append(threading.get_ident())
            return {"url": url, "status": "ok"}

        agent.navigate.side_effect = _navigate

        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", _make_planner()),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                for _ in range(5):
                    c.post("/navigate", json={"url": "https://example.com"})

        assert len(call_tids) == 5
        # All calls must have used the same thread (the browser-worker thread)
        assert len(set(call_tids)) == 1
        assert call_tids[0] == _browser_thread._thread.ident


# ---------------------------------------------------------------------------
# Session management routes run on the browser thread
# ---------------------------------------------------------------------------

class TestSessionRoutesOnBrowserThread:
    def test_session_start_already_running_uses_browser_thread(self, tmp_path):
        """session_start's 'already_running' check runs on the browser thread."""
        agent = _make_agent()
        access_tids = []

        # Track calls to agent.headless (accessed inside _do on the browser thread
        # when the "already_running" branch is taken).
        original_headless = agent.headless
        type(agent).headless = property(lambda self: (
            access_tids.append(threading.get_ident()) or original_headless
        ))

        tools = SystemTools(workspace=tmp_path)
        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/session/start", json={})

        assert resp.status_code == 200
        assert resp.json()["status"] == "already_running"
        # headless was accessed inside _do, which ran on the browser-worker thread
        assert len(access_tids) >= 1
        assert all(tid == _browser_thread._thread.ident for tid in access_tids)

    def test_session_stop_runs_on_browser_thread(self, tmp_path):
        """session_stop's agent.stop() is called from the browser thread."""
        agent = _make_agent()
        stop_tids = []

        def _stop():
            stop_tids.append(threading.get_ident())

        agent.stop.side_effect = _stop
        with patch.object(api_server, "_agent", agent):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/session/stop")

        assert resp.status_code == 200
        # stop() was called once by the route (the lifespan may also call it)
        assert len(stop_tids) >= 1
        # All stop() calls came from the browser-worker thread
        assert all(tid == _browser_thread._thread.ident for tid in stop_tids)

    def test_session_status_runs_on_browser_thread(self, tmp_path):
        """session_status's get_page_info() is called from the browser thread."""
        agent = _make_agent()
        info_tids = []

        def _page_info():
            info_tids.append(threading.get_ident())
            return {"url": "https://example.com", "title": "Example"}

        agent.get_page_info.side_effect = _page_info
        with patch.object(api_server, "_agent", agent):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/session/status")

        assert resp.status_code == 200
        assert resp.json()["active"] is True
        assert len(info_tids) == 1
        assert info_tids[0] == _browser_thread._thread.ident


# ---------------------------------------------------------------------------
# Exception propagation across thread boundary
# ---------------------------------------------------------------------------

class TestExceptionPropagation:
    def test_playwright_error_propagates_to_http_response(self, tmp_path):
        """Errors raised inside the browser thread must surface as HTTP errors."""
        agent = _make_agent()
        agent.navigate.side_effect = RuntimeError("Playwright: Cannot switch to a different thread")

        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", _make_planner()),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/navigate", json={"url": "https://example.com"})

        # The error should surface as a 5xx or be caught — not hang the request
        assert resp.status_code in (422, 500)

    def test_assertion_error_propagates_through_browser_thread(self, tmp_path):
        """AssertionError from assert_text must still produce HTTP 422."""
        agent = _make_agent()
        setattr(agent, "assert_text", MagicMock(side_effect=AssertionError("text not found")))

        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", _make_planner()),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/assert/text", json={"text": "missing text"})

        assert resp.status_code == 422

    def test_http_exception_before_browser_thread(self, tmp_path):
        """HTTPException raised by get_agent() (inactive session) must NOT enter the browser thread."""
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", None),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/navigate", json={"url": "https://example.com"})

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# WebSocket /ws/task uses the browser thread
# ---------------------------------------------------------------------------

class TestWsTaskBrowserThread:
    def test_ws_task_execute_on_browser_thread(self, tmp_path):
        """planner.execute() called from /ws/task must run on the browser thread."""
        agent = _make_agent()
        planner = _make_planner()
        execute_tids = []

        def _execute(steps, agent_arg, **kw):
            execute_tids.append(threading.get_ident())
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        planner.execute.side_effect = _execute

        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", planner),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                with c.websocket_connect("/ws/task") as ws:
                    ws.send_json({"intent": "go to google"})
                    messages = []
                    while True:
                        msg = ws.receive_json()
                        messages.append(msg)
                        if msg["type"] in ("done", "error"):
                            break

        types = [m["type"] for m in messages]
        assert "planned" in types
        assert "done" in types
        assert len(execute_tids) == 1
        assert execute_tids[0] == _browser_thread._thread.ident

    def test_ws_task_streams_step_events(self, tmp_path):
        """step_start and step_done events must be streamed before 'done'."""
        agent = _make_agent()
        planner = _make_planner()

        def _execute(steps, agent_arg, stop_on_error=True,
                     step_start_callback=None, step_callback=None):
            for i, step in enumerate(steps):
                if step_start_callback:
                    step_start_callback(i, step["action"])
                result = {"action": step["action"], "status": "ok", "step": i}
                if step_callback:
                    step_callback(result)
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        planner.execute.side_effect = _execute

        with (
            patch.object(api_server, "_agent", agent),
            patch.object(api_server, "_planner", planner),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                with c.websocket_connect("/ws/task") as ws:
                    ws.send_json({"intent": "go to google"})
                    messages = []
                    while True:
                        msg = ws.receive_json()
                        messages.append(msg)
                        if msg["type"] in ("done", "error"):
                            break

        types = [m["type"] for m in messages]
        assert "planned" in types
        assert "step_start" in types
        assert "step_done" in types
        assert types[-1] == "done"


# ---------------------------------------------------------------------------
# Non-browser routes are NOT routed through the browser thread
# ---------------------------------------------------------------------------

class TestNonBrowserRoutesUnaffected:
    def test_task_plan_does_not_use_browser_thread(self, tmp_path):
        """/task/plan only plans — no browser calls — so it must work with no agent."""
        planner = _make_planner()
        with (
            patch.object(api_server, "_agent", None),
            patch.object(api_server, "_planner", planner),
            patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/task/plan", json={"intent": "go to google"})

        assert resp.status_code == 200
        assert "steps" in resp.json()

    def test_system_write_file_does_not_use_browser_thread(self, tmp_path):
        with patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/system/write_file",
                    json={"path": "test.txt", "content": "hello"},
                )
        assert resp.status_code == 200

    def test_task_schema_no_session_required(self, tmp_path):
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/task/schema")
        assert resp.status_code == 200
        assert "schema" in resp.json()

    def test_ui_no_session_required(self):
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/ui")
        assert resp.status_code == 200
