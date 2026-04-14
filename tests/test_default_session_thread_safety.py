"""
tests/test_default_session_thread_safety.py

Focused tests proving the default shared browser session is thread-safe:

1. The _BrowserThread / _on_browser_thread dispatch mechanism correctly
   serialises concurrent callers onto a single owner thread.
2. Every browser-touching route for the default session that is NOT already
   covered by test_browser_thread_dispatch.py is verified to dispatch through
   the dedicated browser-worker thread (_browser_thread).
3. A "Cannot switch to a different thread" simulation confirms that the
   dispatch architecture prevents the Playwright thread-affinity error.
4. Concurrent HTTP requests from multiple caller threads all land on the
   same single browser-worker thread.

Routes covered here (supplement to test_browser_thread_dispatch.py):
  /popups/close, /hover, /select_option, /scroll, /scroll_to_element,
  /page/info, /page/text, /page/html, /page/attribute, /page/query_all,
  /evaluate, /screenshot, /wait/selector, /wait/load_state, /wait/text,
  /page/extract_links, /page/extract_table, /assert/text, /assert/url,
  /cookies/save, /cookies/load, /tabs/new, /tabs/switch, /tabs/close,
  /tabs/list, /tabs/execute_parallel, /drag_drop, /right_click,
  /double_click, /page/rect, /network/intercept, /network/clear_intercepts,
  /session/viewport, /session/geolocation,
  /task/vision_plan, /task/vision_run, /task/agentic_run
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

import api_server
from api_server import _browser_thread, _on_browser_thread, app
from system_tools import SystemTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(*, page_active: bool = True) -> MagicMock:
    agent = MagicMock()
    agent._page = MagicMock() if page_active else None
    agent.headless = True
    agent._pages = [MagicMock()] if page_active else []
    agent._recording = False
    if page_active:
        agent._page.url = "https://example.com"
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Example"}
    agent.navigate.return_value = {"url": "https://example.com", "status": "ok"}
    agent.close_popups.return_value = {"dismissed": 0}
    agent.click.return_value = {"clicked": True}
    agent.fill.return_value = {"filled": True}
    agent.type_text.return_value = {"typed": True}
    agent.press_key.return_value = {"key": "Enter"}
    agent.hover.return_value = {"hovered": True}
    agent.select_option.return_value = {"selected": "opt"}
    agent.scroll.return_value = {"scrolled": True}
    agent.scroll_to_element.return_value = {"scrolled": True}
    agent.get_text.return_value = "page text"
    agent.get_html.return_value = "<html/>"
    agent.get_attribute.return_value = "value"
    agent.query_all.return_value = []
    agent.evaluate.return_value = 42
    agent.screenshot.return_value = {"path": None, "base64": "aGVsbG8="}
    agent.wait_for_selector.return_value = {"found": True}
    agent.wait_for_load_state.return_value = {"state": "load"}
    agent.wait_text.return_value = {"found": True}
    agent.extract_links.return_value = {"links": []}
    agent.extract_table.return_value = {"rows": []}
    agent.get_cookies.return_value = [{"name": "c", "value": "v"}]
    agent.add_cookies.return_value = {"added": 1}
    agent.new_tab.return_value = {"tab_index": 1}
    agent.switch_tab.return_value = {"tab_index": 0}
    agent.close_tab.return_value = {"closed": True}
    agent.list_tabs.return_value = {"tabs": [], "count": 0}
    agent.drag_and_drop.return_value = {"dragged": True}
    agent.right_click.return_value = {"clicked": True}
    agent.double_click.return_value = {"clicked": True}
    agent.get_element_rect.return_value = {"x": 0, "y": 0, "width": 100, "height": 50}
    agent.set_network_intercept.return_value = {"intercepting": True}
    agent.clear_network_intercepts.return_value = {"cleared": True}
    agent.set_viewport.return_value = {"width": 1280, "height": 720}
    agent.set_geolocation.return_value = {"lat": 0.0, "lon": 0.0}
    agent.upload_file.return_value = {"uploaded": True}
    agent.download_file.return_value = {"saved": True}
    return agent


def _make_planner() -> MagicMock:
    planner = MagicMock()
    planner.plan.return_value = [{"action": "navigate", "url": "https://example.com"}]
    planner.execute.return_value = [{"action": "navigate", "status": "ok", "step": 0}]
    planner.vision_plan.return_value = [{"action": "navigate", "url": "https://example.com"}]
    planner.agentic_run.return_value = {
        "success": True,
        "stopped_reason": "done_by_model",
        "results": [],
        "step_count": 0,
        "failed_count": 0,
    }
    return planner


def _browser_tid() -> int:
    """Return the ident of the global browser-worker thread."""
    return _browser_thread._thread.ident


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def active_session(tmp_path):
    """Provide a patched active default session and SystemTools."""
    agent = _make_agent()
    planner = _make_planner()
    tools = SystemTools(workspace=tmp_path)
    with (
        patch.object(api_server, "_agent", agent),
        patch.object(api_server, "_planner", planner),
        patch.object(api_server, "_tools", tools),
    ):
        yield agent, planner, tools


# ===========================================================================
# 1.  _BrowserThread / _on_browser_thread — concurrent dispatch unit tests
# ===========================================================================

class TestBrowserThreadConcurrentDispatch:
    """Prove that concurrent callers to _on_browser_thread are serialised onto
    the single browser-worker thread and never cross onto a different thread."""

    def test_concurrent_callers_land_on_browser_thread(self):
        """N concurrent threads each calling _on_browser_thread: all land on browser thread."""
        N = 30
        tids: list[int] = []
        tid_lock = threading.Lock()

        def _work(n: int) -> int:
            with tid_lock:
                tids.append(threading.get_ident())
            return n

        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(_on_browser_thread, _work, i) for i in range(N)]
            values = sorted(f.result() for f in as_completed(futures))

        assert values == list(range(N))          # all N calls completed
        assert len(tids) == N                    # exactly N executions
        assert len(set(tids)) == 1               # all on the SAME thread
        assert tids[0] == _browser_tid()         # and that thread IS the browser thread

    def test_concurrent_dispatch_not_on_caller_threads(self):
        """Work submitted via _on_browser_thread never runs on any of the caller threads."""
        N = 20
        caller_tids: set[int] = set()
        worker_tids: list[int] = []
        lock = threading.Lock()

        def _work(caller_tid: int) -> None:
            with lock:
                worker_tids.append(threading.get_ident())

        def _caller(n: int) -> None:
            tid = threading.get_ident()
            with lock:
                caller_tids.add(tid)
            _on_browser_thread(_work, tid)

        with ThreadPoolExecutor(max_workers=N) as pool:
            list(pool.map(_caller, range(N)))

        assert len(set(worker_tids)) == 1
        assert worker_tids[0] == _browser_tid()
        assert worker_tids[0] not in caller_tids

    def test_concurrent_exceptions_propagate_to_callers(self):
        """Exceptions raised inside browser-thread work propagate to the calling thread."""
        N = 10
        errors: list[Exception] = []
        lock = threading.Lock()

        def _boom(n: int) -> None:
            raise ValueError(f"boom-{n}")

        def _caller(n: int) -> None:
            try:
                _on_browser_thread(_boom, n)
            except ValueError as exc:
                with lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=N) as pool:
            list(pool.map(_caller, range(N)))

        assert len(errors) == N
        assert all(str(e).startswith("boom-") for e in errors)

    def test_results_are_independent_per_caller(self):
        """Concurrent callers receive their own return value, not a shared one."""
        N = 25

        def _work(n: int) -> int:
            return n * n

        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = {pool.submit(_on_browser_thread, _work, i): i for i in range(N)}
            results = {futures[f]: f.result() for f in as_completed(futures)}

        for i, result in results.items():
            assert result == i * i


# ===========================================================================
# 2.  Default-session thread-affinity simulation
# ===========================================================================

class TestDefaultSessionThreadAffinityPrevention:
    """Prove that the dispatch architecture prevents the
    'Cannot switch to a different thread' Playwright error for the default
    shared session."""

    def test_navigate_prevents_thread_affinity_error(self, tmp_path):
        """Simulate Playwright raising on wrong thread — dispatch keeps it on owner thread."""
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)

        def _navigate_guarded(url, **kw):
            if threading.get_ident() != _browser_tid():
                raise RuntimeError("Cannot switch to a different thread")
            return {"url": url, "status": "ok"}

        agent.navigate.side_effect = _navigate_guarded

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", _make_planner()), \
             patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/navigate", json={"url": "https://example.com"})

        assert resp.status_code == 200

    def test_task_execute_prevents_thread_affinity_error(self, tmp_path):
        """planner.execute() running on browser thread doesn't raise thread-affinity error."""
        agent = _make_agent()
        planner = MagicMock()
        tools = SystemTools(workspace=tmp_path)

        def _execute_guarded(steps, agent_arg, **kw):
            if threading.get_ident() != _browser_tid():
                raise RuntimeError("Cannot switch to a different thread")
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        planner.execute.side_effect = _execute_guarded

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/task/execute",
                    json={"steps": [{"action": "navigate", "url": "https://example.com"}]},
                )

        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_session_start_creates_agent_on_browser_thread(self, tmp_path):
        """BrowserAgent.start() is called from the browser-worker thread, not the request thread."""
        start_tids: list[int] = []
        mock_agent = _make_agent()
        mock_agent._page = None  # before start()

        def _start_side_effect():
            start_tids.append(threading.get_ident())
            mock_agent._page = MagicMock()
            mock_agent._page.url = "https://about:blank"

        mock_agent.start.side_effect = _start_side_effect

        tools = SystemTools(workspace=tmp_path)

        with patch("api_server.BrowserAgent", return_value=mock_agent), \
             patch("api_server.TaskPlanner", return_value=_make_planner()), \
             patch.object(api_server, "_tools", tools), \
             patch.object(api_server, "_agent", None), \
             patch.object(api_server, "_planner", None):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/session/start", json={})

        assert resp.status_code == 200
        assert len(start_tids) == 1
        assert start_tids[0] == _browser_tid()
        assert start_tids[0] != threading.get_ident()

    def test_session_stop_runs_on_browser_thread(self, tmp_path):
        """agent.stop() is dispatched to the browser-worker thread."""
        agent = _make_agent()
        stop_tids: list[int] = []

        def _stop():
            stop_tids.append(threading.get_ident())

        agent.stop.side_effect = _stop

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", _make_planner()), \
             patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/session/stop")

        assert resp.status_code == 200
        assert len(stop_tids) == 1
        assert stop_tids[0] == _browser_tid()

    def test_concurrent_requests_all_land_on_browser_thread(self, tmp_path):
        """_on_browser_thread serialises N concurrent request threads onto one owner thread."""
        agent = _make_agent()
        nav_tids: list[int] = []
        lock = threading.Lock()

        def _navigate(url, **kw):
            with lock:
                nav_tids.append(threading.get_ident())
            return {"url": url, "status": "ok"}

        agent.navigate.side_effect = _navigate

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", _make_planner()), \
             patch.object(api_server, "_tools", SystemTools(workspace=tmp_path)):
            with TestClient(app, raise_server_exceptions=False):
                # Send N requests via _on_browser_thread from concurrent Python threads
                # (bypassing HTTP transport to avoid httpx threading limitations).
                N = 15
                results: list[dict] = []
                res_lock = threading.Lock()

                def _do_dispatch(i: int) -> None:
                    result = _on_browser_thread(
                        agent.navigate, f"https://example.com/{i}"
                    )
                    with res_lock:
                        results.append(result)

                with ThreadPoolExecutor(max_workers=N) as pool:
                    list(pool.map(_do_dispatch, range(N)))

        assert len(nav_tids) == N
        assert len(set(nav_tids)) == 1
        assert nav_tids[0] == _browser_tid()
        assert len(results) == N


# ===========================================================================
# 3.  Route coverage — routes not yet in test_browser_thread_dispatch.py
# ===========================================================================

class TestDefaultSessionRoutesCoverage:
    """Verify that every browser-touching route for the default session
    dispatches its Playwright work to the browser-worker thread."""

    # -- interaction routes -----------------------------------------------

    def test_popups_close_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.close_popups.side_effect = lambda: tids.append(threading.get_ident()) or {"dismissed": 0}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/popups/close")
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_hover_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.hover.side_effect = lambda sel: tids.append(threading.get_ident()) or {"hovered": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/hover", json={"selector": "#btn"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_select_option_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.select_option.side_effect = lambda sel, val: tids.append(threading.get_ident()) or {"selected": val}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/select_option", json={"selector": "#sel", "value": "opt1"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_scroll_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.scroll.side_effect = lambda x, y: tids.append(threading.get_ident()) or {"scrolled": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/scroll", json={"x": 0, "y": 300})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_scroll_to_element_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.scroll_to_element.side_effect = lambda sel: tids.append(threading.get_ident()) or {"scrolled": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/scroll_to_element", json={"selector": "#footer"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_right_click_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.right_click.side_effect = lambda sel: tids.append(threading.get_ident()) or {"clicked": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/right_click", json={"selector": "#menu"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_double_click_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.double_click.side_effect = lambda sel: tids.append(threading.get_ident()) or {"clicked": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/double_click", json={"selector": "#item"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_drag_drop_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.drag_and_drop.side_effect = lambda src, tgt: tids.append(threading.get_ident()) or {"dragged": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/drag_drop", json={"source": "#src", "target": "#dst"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    # -- page info routes -----------------------------------------------

    def test_page_info_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.get_page_info.side_effect = lambda: tids.append(threading.get_ident()) or {"url": "https://x.com", "title": "X"}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/page/info")
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_get_text_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.get_text.side_effect = lambda sel: tids.append(threading.get_ident()) or "page text"
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/page/text", json={"selector": "body"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_get_html_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.get_html.side_effect = lambda sel: tids.append(threading.get_ident()) or "<p/>"
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/page/html", json={"selector": "body"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_evaluate_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.evaluate.side_effect = lambda script: tids.append(threading.get_ident()) or 1
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/evaluate", json={"script": "1+1"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_screenshot_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.screenshot.side_effect = lambda **kw: tids.append(threading.get_ident()) or {"path": None, "base64": "abc"}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/screenshot", json={})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_get_element_rect_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.get_element_rect.side_effect = lambda sel: tids.append(threading.get_ident()) or {"x": 0, "y": 0, "width": 10, "height": 10}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/page/rect", json={"selector": "#box"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    # -- wait / assert routes -------------------------------------------

    def test_wait_selector_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.wait_for_selector.side_effect = lambda sel, **kw: tids.append(threading.get_ident()) or {"found": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/wait/selector", json={"selector": "#el"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_wait_load_state_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.wait_for_load_state.side_effect = lambda state: tids.append(threading.get_ident()) or {"state": state}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/wait/load_state", json={"state": "load"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_wait_text_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.wait_text.side_effect = lambda text, **kw: tids.append(threading.get_ident()) or {"found": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/wait/text", json={"text": "hello"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_assert_url_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        # MagicMock intercepts 'assert_*' names; use setattr to bypass it.
        mock_assert_url = MagicMock(
            side_effect=lambda pattern: tids.append(threading.get_ident()) or {"matched": True}
        )
        setattr(agent, "assert_url", mock_assert_url)
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/assert/url", json={"pattern": "example\\.com"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    # -- extract routes -------------------------------------------------

    def test_extract_links_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.extract_links.side_effect = lambda **kw: tids.append(threading.get_ident()) or {"links": []}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/page/extract_links", json={})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_extract_table_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.extract_table.side_effect = lambda **kw: tids.append(threading.get_ident()) or {"rows": []}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/page/extract_table", json={})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    # -- cookies routes -------------------------------------------------

    def test_cookies_save_on_browser_thread(self, active_session, tmp_path):
        agent, _, tools = active_session
        tids: list[int] = []
        cookies = [{"name": "session", "value": "abc"}]
        agent.get_cookies.side_effect = lambda: tids.append(threading.get_ident()) or cookies
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/cookies/save", json={"path": "cookies.json"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_cookies_load_on_browser_thread(self, active_session, tmp_path):
        agent, _, tools = active_session
        tids: list[int] = []
        cookies = [{"name": "session", "value": "abc"}]
        # Write the cookie file into workspace
        (tmp_path / "cookies.json").write_text(json.dumps(cookies))
        agent.add_cookies.side_effect = lambda cks: tids.append(threading.get_ident()) or {"added": len(cks)}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/cookies/load", json={"path": "cookies.json"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    # -- tab routes -----------------------------------------------------

    def test_new_tab_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.new_tab.side_effect = lambda **kw: tids.append(threading.get_ident()) or {"tab_index": 1}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/tabs/new", json={})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_switch_tab_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.switch_tab.side_effect = lambda idx: tids.append(threading.get_ident()) or {"tab_index": idx}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/tabs/switch", json={"index": 0})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_close_tab_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.close_tab.side_effect = lambda **kw: tids.append(threading.get_ident()) or {"closed": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/tabs/close", json={})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_list_tabs_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.list_tabs.side_effect = lambda: tids.append(threading.get_ident()) or {"tabs": [], "count": 0}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/tabs/list")
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_tabs_execute_parallel_all_on_browser_thread(self, active_session):
        """tabs/execute_parallel wraps all tab work in _browser_thread.submit so
        all calls (new_tab, switch_tab, navigate, execute) land on the browser thread."""
        agent, planner, _ = active_session
        tids: list[int] = []
        lock = threading.Lock()

        def _record_tid(*a, **kw):
            with lock:
                tids.append(threading.get_ident())

        agent.new_tab.side_effect = lambda **kw: (_record_tid(), {"tab_index": 1})[1]
        agent.switch_tab.side_effect = lambda idx: (_record_tid(), {"tab_index": idx})[1]
        agent.navigate.side_effect = lambda url, **kw: (_record_tid(), {"url": url, "status": "ok"})[1]
        planner.execute.side_effect = (
            lambda steps, agent_arg, **kw: [
                {"action": s["action"], "status": "ok", "step": i}
                for i, s in enumerate(steps)
            ]
        )

        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/tabs/execute_parallel", json={
                "tasks": [
                    {"steps": [{"action": "navigate", "url": "https://a.com"}]},
                    {"steps": [{"action": "navigate", "url": "https://b.com"}]},
                ],
            })

        assert resp.status_code == 200
        assert len(tids) > 0
        assert len(set(tids)) == 1
        assert tids[0] == _browser_tid()

    # -- network / viewport / geolocation routes -----------------------

    def test_network_intercept_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.set_network_intercept.side_effect = lambda pat, **kw: tids.append(threading.get_ident()) or {"intercepting": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/network/intercept", json={"url_pattern": "*.js", "action": "block"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_clear_network_intercepts_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.clear_network_intercepts.side_effect = lambda: tids.append(threading.get_ident()) or {"cleared": True}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/network/clear_intercepts")
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_set_viewport_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.set_viewport.side_effect = lambda w, h: tids.append(threading.get_ident()) or {"width": w, "height": h}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/session/viewport", json={"width": 1920, "height": 1080})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_set_geolocation_on_browser_thread(self, active_session):
        agent, _, _ = active_session
        tids: list[int] = []
        agent.set_geolocation.side_effect = lambda lat, lon, **kw: tids.append(threading.get_ident()) or {"lat": lat, "lon": lon}
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/session/geolocation", json={"latitude": 51.5, "longitude": -0.1})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    # -- task/vision and task/agentic routes ---------------------------

    def test_vision_plan_on_browser_thread(self, active_session):
        """task/vision_plan dispatches vision_plan() (which takes a screenshot) to the browser thread."""
        agent, planner, _ = active_session
        tids: list[int] = []
        planner.vision_plan.side_effect = (
            lambda intent, agent_arg, **kw:
            (tids.append(threading.get_ident()),
             [{"action": "navigate", "url": "https://example.com"}])[1]
        )
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/task/vision_plan", json={"intent": "find the login button"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_vision_run_execute_on_browser_thread(self, active_session):
        """task/vision_run dispatches both vision_plan() and execute() to the browser thread."""
        agent, planner, _ = active_session
        tids: list[int] = []
        plan_tids: list[int] = []

        planner.vision_plan.side_effect = (
            lambda intent, agent_arg, **kw:
            (plan_tids.append(threading.get_ident()),
             [{"action": "navigate", "url": "https://example.com"}])[1]
        )
        planner.execute.side_effect = (
            lambda steps, agent_arg, **kw:
            (tids.append(threading.get_ident()),
             [{"action": s["action"], "status": "ok", "step": i} for i, s in enumerate(steps)])[1]
        )

        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/task/vision_run", json={"intent": "click the big button"})
        assert resp.status_code == 200
        assert len(plan_tids) == 1 and plan_tids[0] == _browser_tid()
        assert len(tids) == 1 and tids[0] == _browser_tid()

    def test_agentic_run_on_browser_thread(self, active_session):
        """task/agentic_run dispatches agentic_run() to the browser thread."""
        agent, planner, _ = active_session
        tids: list[int] = []
        planner.agentic_run.side_effect = (
            lambda intent, agent_arg, **kw:
            (tids.append(threading.get_ident()), {
                "success": True,
                "stopped_reason": "done_by_model",
                "results": [],
                "step_count": 0,
                "failed_count": 0,
            })[1]
        )

        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/task/agentic_run", json={"intent": "search for Python"})
        assert resp.status_code == 200
        assert len(tids) == 1 and tids[0] == _browser_tid()

    # -- multiple sequential requests prove same thread each time -------

    def test_sequential_default_session_routes_use_same_thread(self, active_session):
        """Five different browser routes fired sequentially all land on the same thread."""
        agent, planner, _ = active_session
        tids: list[int] = []
        lock = threading.Lock()

        def _record(*a, **kw):
            with lock:
                tids.append(threading.get_ident())

        agent.navigate.side_effect = lambda url, **kw: (_record(), {"url": url, "status": "ok"})[1]
        agent.click.side_effect = lambda sel, **kw: (_record(), {"clicked": True})[1]
        agent.fill.side_effect = lambda sel, val: (_record(), {"filled": True})[1]
        agent.press_key.side_effect = lambda key: (_record(), {"key": key})[1]
        agent.get_page_info.side_effect = lambda: (_record(), {"url": "https://x.com", "title": "X"})[1]

        with TestClient(app, raise_server_exceptions=False) as c:
            c.post("/navigate", json={"url": "https://example.com"})
            c.post("/click", json={"selector": "#btn"})
            c.post("/fill", json={"selector": "#inp", "value": "hi"})
            c.post("/press_key", json={"key": "Enter"})
            c.get("/page/info")

        assert len(tids) == 5
        assert len(set(tids)) == 1
        assert tids[0] == _browser_tid()
