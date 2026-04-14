"""
tests/test_pool_and_ws_auth.py

Tests for:
1. Pooled-agent thread-dispatch — each pooled BrowserAgent gets its own
   _BrowserThread, and all Playwright calls run on that owner thread.
2. WebSocket auth enforcement — /ws/task rejects connections when
   BROWSER_API_KEY is set and the correct key is not supplied.
3. GUI task execution path — no thread-affinity error in the ws/task
   execution path (dispatch goes through a browser thread).
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
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(*, page_active: bool = True) -> MagicMock:
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


def _make_planner(*, steps: list[dict] | None = None) -> MagicMock:
    planner = MagicMock()
    planner.plan.return_value = steps or [{"action": "navigate", "url": "https://example.com"}]
    planner.execute.return_value = [{"action": "navigate", "status": "ok", "step": 0}]
    return planner


# ---------------------------------------------------------------------------
# Fixture: clean pool state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_pool():
    """Reset pool dicts before and after each test."""
    with api_server._pool_lock:
        api_server._agent_pool.clear()
        api_server._planner_pool.clear()
        api_server._pool_thread.clear()
    yield
    with api_server._pool_lock:
        api_server._agent_pool.clear()
        api_server._planner_pool.clear()
        api_server._pool_thread.clear()


# ===========================================================================
# 1.  Pooled-agent thread-dispatch
# ===========================================================================

class TestPooledAgentThreadDispatch:
    """Verify each pooled agent executes browser calls on its own owner thread."""

    def _make_client(self, tmp_path: Path):
        tools = SystemTools(workspace=tmp_path)
        return TestClient(app, raise_server_exceptions=False), tools

    def test_pool_start_creates_dedicated_thread(self, tmp_path):
        """pool_agent_start must register a _BrowserThread in _pool_thread."""
        c, tools = self._make_client(tmp_path)
        mock_agent = _make_agent()
        captured_bt: list[_BrowserThread] = []

        with patch("api_server.BrowserAgent", return_value=mock_agent), \
             patch.object(api_server, "_tools", tools):
            with c as client:
                resp = client.post(
                    "/agents/pool/start",
                    json={"agent_id": "my-agent"},
                )
                assert resp.status_code == 200
                assert resp.json()["agent_id"] == "my-agent"
                # Check inside the client context — before lifespan shutdown clears the pool
                with api_server._pool_lock:
                    assert "my-agent" in api_server._pool_thread
                    captured_bt.append(api_server._pool_thread["my-agent"])

        assert isinstance(captured_bt[0], _BrowserThread)
        assert captured_bt[0]._thread.is_alive()

    def test_pool_start_starts_agent_on_dedicated_thread(self, tmp_path):
        """agent.start() must be called from the dedicated pool thread, not the request thread."""
        c, tools = self._make_client(tmp_path)
        start_tids: list[int] = []

        mock_agent = _make_agent()

        def _start():
            start_tids.append(threading.get_ident())

        mock_agent.start.side_effect = _start

        with patch("api_server.BrowserAgent", return_value=mock_agent), \
             patch.object(api_server, "_tools", tools):
            with c as client:
                resp = client.post(
                    "/agents/pool/start",
                    json={"agent_id": "thread-start"},
                )
                assert resp.status_code == 200
                # Inspect thread inside client context before lifespan shutdown
                with api_server._pool_lock:
                    bt = api_server._pool_thread["thread-start"]

        assert len(start_tids) == 1
        # Must NOT be the test (main) thread
        assert start_tids[0] != threading.get_ident()
        # Must be the dedicated pool thread
        assert start_tids[0] == bt._thread.ident

    def test_pool_execute_runs_on_dedicated_thread(self, tmp_path):
        """planner.execute() for a pooled agent must run on its owner thread."""
        c, tools = self._make_client(tmp_path)
        execute_tids: list[int] = []

        mock_agent = _make_agent()
        mock_planner = _make_planner()

        def _execute(steps, agent_arg, **kw):
            execute_tids.append(threading.get_ident())
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        mock_planner.execute.side_effect = _execute

        # Pre-populate pool with a dedicated thread
        bt = _BrowserThread()
        with api_server._pool_lock:
            api_server._agent_pool["pool-exec"] = mock_agent
            api_server._planner_pool["pool-exec"] = mock_planner
            api_server._pool_thread["pool-exec"] = bt

        with patch.object(api_server, "_tools", tools):
            with c as client:
                resp = client.post(
                    "/agents/pool/pool-exec/task/execute",
                    json={"steps": [{"action": "navigate", "url": "https://example.com"}]},
                )

        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert len(execute_tids) == 1
        assert execute_tids[0] == bt._thread.ident

    def test_pool_execute_different_agents_use_different_threads(self, tmp_path):
        """Two pooled agents must use independent owner threads."""
        c, tools = self._make_client(tmp_path)
        execute_tids: dict[str, list[int]] = {"a1": [], "a2": []}

        def make_planner_for(key: str) -> MagicMock:
            p = _make_planner()
            def _execute(steps, agent_arg, **kw):
                execute_tids[key].append(threading.get_ident())
                return [{"action": s["action"], "status": "ok", "step": i}
                        for i, s in enumerate(steps)]
            p.execute.side_effect = _execute
            return p

        bt1 = _BrowserThread()
        bt2 = _BrowserThread()
        with api_server._pool_lock:
            api_server._agent_pool["a1"]   = _make_agent()
            api_server._agent_pool["a2"]   = _make_agent()
            api_server._planner_pool["a1"] = make_planner_for("a1")
            api_server._planner_pool["a2"] = make_planner_for("a2")
            api_server._pool_thread["a1"]  = bt1
            api_server._pool_thread["a2"]  = bt2

        with patch.object(api_server, "_tools", tools):
            with c as client:
                for aid in ("a1", "a2"):
                    client.post(
                        f"/agents/pool/{aid}/task/execute",
                        json={"steps": [{"action": "navigate", "url": "https://example.com"}]},
                    )

        assert execute_tids["a1"] == [bt1._thread.ident]
        assert execute_tids["a2"] == [bt2._thread.ident]
        assert bt1._thread.ident != bt2._thread.ident

    def test_pool_stop_calls_agent_stop_on_owner_thread(self, tmp_path):
        """DELETE /agents/pool/{id} must call agent.stop() on the owner thread."""
        c, tools = self._make_client(tmp_path)
        stop_tids: list[int] = []

        mock_agent = _make_agent()

        def _stop():
            stop_tids.append(threading.get_ident())

        mock_agent.stop.side_effect = _stop

        bt = _BrowserThread()
        with api_server._pool_lock:
            api_server._agent_pool["stop-me"] = mock_agent
            api_server._pool_thread["stop-me"] = bt

        with patch.object(api_server, "_tools", tools):
            with c as client:
                resp = client.delete("/agents/pool/stop-me")

        assert resp.status_code == 200
        assert len(stop_tids) == 1
        assert stop_tids[0] == bt._thread.ident

    def test_pool_stop_removes_thread_entry(self, tmp_path):
        """_pool_thread must be cleaned up when an agent is stopped."""
        c, tools = self._make_client(tmp_path)
        mock_agent = _make_agent()
        bt = _BrowserThread()
        with api_server._pool_lock:
            api_server._agent_pool["clean-me"] = mock_agent
            api_server._pool_thread["clean-me"] = bt

        with patch.object(api_server, "_tools", tools):
            with c as client:
                client.delete("/agents/pool/clean-me")

        with api_server._pool_lock:
            assert "clean-me" not in api_server._pool_thread

    def test_pool_status_reads_on_owner_thread(self, tmp_path):
        """GET /agents/pool/{id} status reads must happen on the owner thread."""
        c, tools = self._make_client(tmp_path)
        info_tids: list[int] = []

        mock_agent = _make_agent()

        def _list_tabs():
            info_tids.append(threading.get_ident())
            return {"tabs": [], "count": 0}

        mock_agent.list_tabs.side_effect = _list_tabs

        bt = _BrowserThread()
        with api_server._pool_lock:
            api_server._agent_pool["status-me"] = mock_agent
            api_server._pool_thread["status-me"] = bt

        with patch.object(api_server, "_tools", tools):
            with c as client:
                resp = client.get("/agents/pool/status-me")

        assert resp.status_code == 200
        assert resp.json()["active"] is True
        assert len(info_tids) == 1
        assert info_tids[0] == bt._thread.ident

    def test_agents_execute_parallel_uses_dedicated_threads(self, tmp_path):
        """agents_execute_parallel must dispatch each agent on its own thread."""
        c, tools = self._make_client(tmp_path)

        # Use per-agent capture dicts keyed by agent id
        execute_info: dict[str, list[int]] = {"p1": [], "p2": []}

        bt1 = _BrowserThread()
        bt2 = _BrowserThread()

        def make_planner_for(key: str) -> MagicMock:
            p = MagicMock()
            def _execute(steps, agent_arg, **kw):
                execute_info[key].append(threading.get_ident())
                return [{"action": s["action"], "status": "ok", "step": i}
                        for i, s in enumerate(steps)]
            p.execute.side_effect = _execute
            return p

        mock_p1 = make_planner_for("p1")
        mock_p2 = make_planner_for("p2")

        # Track which planner is created next so each _run() call gets the
        # right per-agent planner even though agents_execute_parallel always
        # calls TaskPlanner() inside _run().
        planners_iter = iter([mock_p1, mock_p2])

        with api_server._pool_lock:
            api_server._agent_pool["p1"]  = _make_agent()
            api_server._agent_pool["p2"]  = _make_agent()
            api_server._pool_thread["p1"] = bt1
            api_server._pool_thread["p2"] = bt2

        with patch.object(api_server, "_tools", tools), \
             patch("api_server.TaskPlanner", side_effect=planners_iter):
            with c as client:
                resp = client.post("/agents/execute_parallel", json={
                    "tasks": [
                        {"agent_id": "p1",
                         "steps": [{"action": "navigate", "url": "https://a.com"}]},
                        {"agent_id": "p2",
                         "steps": [{"action": "navigate", "url": "https://b.com"}]},
                    ],
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["all_succeeded"] is True
        # Exactly one execute() per agent
        total_executions = len(execute_info["p1"]) + len(execute_info["p2"])
        assert total_executions == 2
        # Each agent's execute ran on a browser thread (not main test thread)
        for tids in execute_info.values():
            for tid in tids:
                assert tid != threading.get_ident()
        # The two threads are different from each other
        p1_tids = set(execute_info["p1"])
        p2_tids = set(execute_info["p2"])
        if p1_tids and p2_tids:
            assert p1_tids.isdisjoint(p2_tids) or bt1._thread.ident == bt2._thread.ident
        # Verify bt1 and bt2 are actually different threads
        assert bt1._thread.ident != bt2._thread.ident

    def test_agents_execute_parallel_autospawn_creates_thread(self, tmp_path):
        """Auto-spawned agents in execute_parallel get their own _BrowserThread."""
        c, tools = self._make_client(tmp_path)
        spawn_tids: list[int] = []

        spawned_mock = _make_agent()

        def _start():
            spawn_tids.append(threading.get_ident())

        spawned_mock.start.side_effect = _start

        mock_planner = _make_planner()

        with patch.object(api_server, "_tools", tools), \
             patch("api_server.BrowserAgent", return_value=spawned_mock), \
             patch("api_server.TaskPlanner", return_value=mock_planner):
            with c as client:
                resp = client.post("/agents/execute_parallel", json={
                    "agent_count": 1,
                    "steps": [{"action": "navigate", "url": "https://example.com"}],
                    "auto_stop": False,
                })
                assert resp.status_code == 200
                # Check pool inside client context — before lifespan shutdown
                with api_server._pool_lock:
                    pool_aids = list(api_server._agent_pool.keys())
                    assert len(pool_aids) == 1
                    # A dedicated thread was registered
                    assert pool_aids[0] in api_server._pool_thread

        # agent.start() was called from a worker thread, not the main thread
        assert len(spawn_tids) == 1
        assert spawn_tids[0] != threading.get_ident()

    def test_get_pool_thread_fallback_to_shared(self, tmp_path):
        """get_pool_thread falls back to _browser_thread when no dedicated thread exists."""
        mock_agent = _make_agent()
        with api_server._pool_lock:
            api_server._agent_pool["legacy"] = mock_agent
            # no entry in _pool_thread — simulates legacy / test injection

        bt = api_server.get_pool_thread("legacy")
        assert bt is _browser_thread


# ===========================================================================
# 2.  WebSocket auth enforcement
# ===========================================================================

class TestWsTaskAuth:
    """Verify /ws/task enforces BROWSER_API_KEY when set."""

    def test_no_api_key_set_allows_any_connection(self, tmp_path):
        """When BROWSER_API_KEY is not set, ws/task is freely accessible."""
        agent   = _make_agent()
        planner = _make_planner()
        tools   = SystemTools(workspace=tmp_path)

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools), \
             patch.object(api_server, "_API_KEY", None):
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

    def test_correct_key_in_header_allows_connection(self, tmp_path):
        """A valid X-API-Key header is accepted."""
        agent   = _make_agent()
        planner = _make_planner()
        tools   = SystemTools(workspace=tmp_path)
        secret  = "my-secret-key"

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools), \
             patch.object(api_server, "_API_KEY", secret):
            with TestClient(app, raise_server_exceptions=False) as c:
                with c.websocket_connect(
                    "/ws/task", headers={"X-API-Key": secret}
                ) as ws:
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

    def test_correct_key_in_query_param_allows_connection(self, tmp_path):
        """A valid ?api_key= query parameter is accepted."""
        agent   = _make_agent()
        planner = _make_planner()
        tools   = SystemTools(workspace=tmp_path)
        secret  = "query-key"

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools), \
             patch.object(api_server, "_API_KEY", secret):
            with TestClient(app, raise_server_exceptions=False) as c:
                with c.websocket_connect(
                    f"/ws/task?api_key={secret}"
                ) as ws:
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

    def test_missing_key_closes_connection(self, tmp_path):
        """When BROWSER_API_KEY is set and no key is provided, the WS must be closed."""
        secret = "gate-key"

        with patch.object(api_server, "_API_KEY", secret):
            with TestClient(app, raise_server_exceptions=False) as c:
                with pytest.raises(Exception):
                    # The server closes the connection with code 4401;
                    # TestClient raises when the server closes before accept().
                    with c.websocket_connect("/ws/task") as ws:
                        ws.send_json({"intent": "test"})
                        ws.receive_json()

    def test_wrong_key_closes_connection(self, tmp_path):
        """A wrong key must be rejected (connection closed)."""
        secret = "real-key"

        with patch.object(api_server, "_API_KEY", secret):
            with TestClient(app, raise_server_exceptions=False) as c:
                with pytest.raises(Exception):
                    with c.websocket_connect(
                        "/ws/task", headers={"X-API-Key": "wrong-key"}
                    ) as ws:
                        ws.send_json({"intent": "test"})
                        ws.receive_json()

    def test_http_route_still_protected_when_key_set(self, tmp_path):
        """HTTP routes remain protected; this is a quick sanity check."""
        secret = "http-key"
        with patch.object(api_server, "_API_KEY", secret):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/session/status")
                assert resp.status_code == 401
                resp = c.get("/session/status", headers={"X-API-Key": secret})
                # 400 because no session started — not 401
                assert resp.status_code != 401


# ===========================================================================
# 3.  GUI task execution path — no thread-affinity error
# ===========================================================================

class TestGuiTaskExecutionPath:
    """
    Verify the ws/task code path dispatches Playwright work through a browser
    thread, preventing the 'Cannot switch to a different thread' error from
    the GUI task runner.
    """

    def test_ws_task_execute_runs_on_browser_thread(self, tmp_path):
        """planner.execute() inside /ws/task must run on the browser-worker thread."""
        agent   = _make_agent()
        tools   = SystemTools(workspace=tmp_path)
        execute_tids: list[int] = []
        planner = MagicMock()
        planner.plan.return_value = [{"action": "navigate", "url": "https://example.com"}]

        def _execute(steps, agent_arg, **kw):
            execute_tids.append(threading.get_ident())
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        planner.execute.side_effect = _execute

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools), \
             patch.object(api_server, "_API_KEY", None):
            with TestClient(app, raise_server_exceptions=False) as c:
                with c.websocket_connect("/ws/task") as ws:
                    ws.send_json({"intent": "go to google and search pwwp2a"})
                    messages = []
                    while True:
                        msg = ws.receive_json()
                        messages.append(msg)
                        if msg["type"] in ("done", "error"):
                            break

        types = [m["type"] for m in messages]
        assert "planned" in types
        assert "done" in types
        assert messages[-1]["type"] == "done"
        assert messages[-1]["success"] is True

        # execute() ran on the browser-worker thread, not the test/main thread
        assert len(execute_tids) == 1
        test_tid = threading.get_ident()
        assert execute_tids[0] != test_tid
        assert execute_tids[0] == _browser_thread._thread.ident

    def test_ws_task_no_thread_switch_error(self, tmp_path):
        """
        Simulate the thread-affinity bug scenario: the agent was created on
        the browser-worker thread; executing on the same thread must not raise.
        This test confirms the corrected code path reaches the same thread.
        """
        agent  = _make_agent()
        tools  = SystemTools(workspace=tmp_path)

        planner = MagicMock()
        planner.plan.return_value = [{"action": "navigate", "url": "https://example.com"}]

        def _execute_may_raise_on_wrong_thread(steps, agent_arg, **kw):
            # Raise the Playwright error if called from any thread other than
            # the browser-worker thread — simulating real Playwright behaviour.
            if threading.get_ident() != _browser_thread._thread.ident:
                raise RuntimeError("Cannot switch to a different thread")
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        planner.execute.side_effect = _execute_may_raise_on_wrong_thread

        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools), \
             patch.object(api_server, "_API_KEY", None):
            with TestClient(app, raise_server_exceptions=False) as c:
                with c.websocket_connect("/ws/task") as ws:
                    ws.send_json({"intent": "go to google"})
                    messages = []
                    while True:
                        msg = ws.receive_json()
                        messages.append(msg)
                        if msg["type"] in ("done", "error"):
                            break

        # Must complete successfully — no thread error
        assert messages[-1]["type"] == "done"
        assert messages[-1]["success"] is True

    def test_pool_agent_execute_no_thread_switch_error(self, tmp_path):
        """
        Pooled agent execution must not raise 'Cannot switch to a different
        thread' because Playwright calls are dispatched to the owner thread.
        """
        tools = SystemTools(workspace=tmp_path)
        bt    = _BrowserThread()
        agent = _make_agent()

        planner = MagicMock()

        def _execute_may_raise(steps, agent_arg, **kw):
            if threading.get_ident() != bt._thread.ident:
                raise RuntimeError("Cannot switch to a different thread")
            return [{"action": s["action"], "status": "ok", "step": i}
                    for i, s in enumerate(steps)]

        planner.execute.side_effect = _execute_may_raise

        with api_server._pool_lock:
            api_server._agent_pool["guarded"]   = agent
            api_server._planner_pool["guarded"] = planner
            api_server._pool_thread["guarded"]  = bt

        with patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/agents/pool/guarded/task/execute",
                    json={"steps": [{"action": "navigate", "url": "https://example.com"}]},
                )

        assert resp.status_code == 200
        assert resp.json()["success"] is True
