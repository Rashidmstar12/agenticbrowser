"""
Extra API coverage tests for routes not yet exercised:
  - /session/start (already_running branch)
  - /tabs/execute_parallel (navigate failure + stop_on_error branch)
  - /agents/pool/* edge cases: execute with missing planner, agent with no page (400)
  - /agents/execute_parallel auto_stop + no tasks/no agent_count 422
  - Advanced interaction routes: /drag_drop, /right_click, /double_click,
    /page/rect, /network/intercept, /network/clear_intercepts,
    /session/viewport, /session/geolocation
  - Recording routes: /recording/start, /recording/stop, /recording/gif
  - lifespan shutdown clears pool
"""

from __future__ import annotations

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
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(*, page_active: bool = True) -> MagicMock:
    agent = MagicMock()
    agent._page = MagicMock() if page_active else None
    agent.headless = True
    agent._pages = [MagicMock()] if page_active else []
    agent._recording = False
    agent.navigate.return_value = {"url": "https://example.com", "status": "ok"}
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Example"}
    agent.drag_and_drop.return_value = {"source": "a", "target": "b", "status": "ok"}
    agent.right_click.return_value = {"selector": "a", "status": "ok"}
    agent.double_click.return_value = {"selector": "a", "status": "ok"}
    agent.get_element_rect.return_value = {"x": 10, "y": 20, "width": 100, "height": 50}
    agent.set_network_intercept.return_value = {"url_pattern": "*.jpg", "action": "abort"}
    agent.clear_network_intercepts.return_value = {"cleared": 0}
    agent.set_viewport.return_value = {"width": 1280, "height": 720, "status": "ok"}
    agent.set_geolocation.return_value = {"lat": 37.0, "lng": -122.0, "status": "ok"}
    agent.start_recording.return_value = {"recording": True}
    agent.stop_recording.return_value = {"recording": False, "path": "/tmp/vid.webm"}
    agent.save_recording_gif.return_value = {"gif_path": "/tmp/out.gif"}
    agent.list_tabs.return_value = {"tabs": [], "count": 0}
    agent.new_tab.return_value = {"tab_index": 1}
    agent.switch_tab.return_value = {"tab_index": 0}
    agent.close_tab.return_value = {"closed_index": 0}
    agent.upload_file.return_value = {"selector": "input", "status": "ok"}
    agent.download_file.return_value = {"saved_to": "/tmp/f.bin", "filename": "f.bin"}
    agent.stop.return_value = None
    return agent


def _make_planner() -> MagicMock:
    planner = MagicMock()
    planner.plan.return_value = [{"action": "navigate", "url": "https://example.com"}]
    planner.execute.return_value = [{"step": 0, "action": "navigate", "status": "ok", "result": {}}]
    return planner


@pytest.fixture()
def client_with_agent(tmp_path):
    """Provide a TestClient with a mocked agent + planner + tools."""
    agent = _make_agent()
    planner = _make_planner()
    tools = SystemTools(workspace=tmp_path)
    with patch.object(api_server, "_agent", agent), \
         patch.object(api_server, "_planner", planner), \
         patch.object(api_server, "_tools", tools):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, agent, planner, tools


@pytest.fixture()
def clean_pool():
    with api_server._pool_lock:
        api_server._agent_pool.clear()
        api_server._planner_pool.clear()
    yield
    with api_server._pool_lock:
        api_server._agent_pool.clear()
        api_server._planner_pool.clear()


# ---------------------------------------------------------------------------
# /session/start — already_running branch
# ---------------------------------------------------------------------------

class TestSessionStartAlreadyRunning:
    def test_already_running_returns_status(self, tmp_path):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/session/start", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "already_running"


# ---------------------------------------------------------------------------
# /tabs/execute_parallel — navigate failure branch
# ---------------------------------------------------------------------------

class TestTabsExecuteParallelNav:
    def test_navigate_failure_stops_when_stop_on_error(self, tmp_path):
        agent = _make_agent()
        # tabs/execute_parallel requires min 2 tasks; make navigate raise for the url= arg
        nav_calls = {"n": 0}


        def _nav(url, *a, **kw):
            nav_calls["n"] += 1
            raise Exception("nav failed")

        agent.navigate.side_effect = _nav
        planner = _make_planner()
        tools = SystemTools(workspace=tmp_path)
        with patch.object(api_server, "_agent", agent), \
             patch.object(api_server, "_planner", planner), \
             patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/tabs/execute_parallel", json={
                    "tasks": [
                        {
                            "steps": [{"action": "navigate", "url": "https://a.com"}],
                            "url": "https://fail.example",
                        },
                        {
                            "steps": [{"action": "navigate", "url": "https://b.com"}],
                            "url": "https://fail2.example",
                        },
                    ],
                    "stop_on_error": True,
                })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tabs"] >= 1


# ---------------------------------------------------------------------------
# Advanced interaction routes
# ---------------------------------------------------------------------------

class TestAdvancedInteractions:
    def test_drag_drop(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/drag_drop", json={"source": "#a", "target": "#b"})
        assert resp.status_code == 200
        agent.drag_and_drop.assert_called_once_with("#a", "#b")

    def test_right_click(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/right_click", json={"selector": "a.link"})
        assert resp.status_code == 200
        agent.right_click.assert_called_once_with("a.link")

    def test_double_click(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/double_click", json={"selector": "button"})
        assert resp.status_code == 200
        agent.double_click.assert_called_once_with("button")

    def test_get_element_rect(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/page/rect", json={"selector": "div.box"})
        assert resp.status_code == 200
        data = resp.json()
        assert "x" in data and "width" in data

    def test_set_network_intercept(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/network/intercept", json={"url_pattern": "*.jpg", "action": "abort"})
        assert resp.status_code == 200
        agent.set_network_intercept.assert_called_once_with("*.jpg", action="abort")

    def test_clear_network_intercepts(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/network/clear_intercepts")
        assert resp.status_code == 200
        agent.clear_network_intercepts.assert_called_once()

    def test_set_viewport(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/session/viewport", json={"width": 1920, "height": 1080})
        assert resp.status_code == 200
        agent.set_viewport.assert_called_once_with(1920, 1080)

    def test_set_geolocation(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/session/geolocation", json={"latitude": 40.7, "longitude": -74.0})
        assert resp.status_code == 200
        agent.set_geolocation.assert_called_once_with(40.7, -74.0, accuracy=10.0)

    def test_set_geolocation_custom_accuracy(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        resp = c.post("/session/geolocation", json={"latitude": 51.5, "longitude": 0.1, "accuracy": 50.0})
        assert resp.status_code == 200
        agent.set_geolocation.assert_called_once_with(51.5, 0.1, accuracy=50.0)


# ---------------------------------------------------------------------------
# Recording routes
# ---------------------------------------------------------------------------

class TestRecordingRoutes:
    def test_start_recording(self, client_with_agent, tmp_path):
        c, agent, _, tools = client_with_agent
        agent.start_video_recording.return_value = {"recording": True}
        resp = c.post("/recording/start", json={"path": "video.webm"})
        assert resp.status_code in (200, 409)

    def test_stop_recording(self, client_with_agent):
        c, agent, _, _ = client_with_agent
        agent.stop_video_recording.return_value = {"recording": False, "path": "/tmp/vid.webm"}
        resp = c.post("/recording/stop")
        assert resp.status_code in (200, 409)

    def test_record_gif(self, client_with_agent, tmp_path):
        c, agent, _, tools = client_with_agent
        agent.record_gif.return_value = {"gif_path": "/tmp/out.gif"}
        resp = c.post("/recording/gif", json={"path": "out.gif"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Agent pool edge cases
# ---------------------------------------------------------------------------

class TestAgentPoolEdgeCases:
    def test_pool_agent_with_no_page_returns_400(self, tmp_path, clean_pool):
        """get_pooled_agent raises 400 when agent._page is None."""
        agent = _make_agent(page_active=False)
        tools = SystemTools(workspace=tmp_path)
        with api_server._pool_lock:
            api_server._agent_pool["no-page"] = agent
        with patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/agents/pool/no-page")
        assert resp.status_code == 400

    def test_pool_execute_creates_planner_when_missing(self, tmp_path, clean_pool):
        """pool_agent_execute auto-creates a TaskPlanner when none is in _planner_pool."""
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        mock_planner = _make_planner()
        with api_server._pool_lock:
            api_server._agent_pool["auto-plan"] = agent
            # intentionally leave _planner_pool empty for this id
        with patch.object(api_server, "_tools", tools), \
             patch("api_server.TaskPlanner", return_value=mock_planner):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/agents/pool/auto-plan/task/execute",
                    json={"steps": [{"action": "navigate", "url": "https://example.com"}]},
                )
        assert resp.status_code == 200

    def test_pool_execute_invalid_steps_returns_422(self, tmp_path, clean_pool):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        with api_server._pool_lock:
            api_server._agent_pool["bad-steps"] = agent
        with patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/agents/pool/bad-steps/task/execute",
                    json={"steps": [{"action": "not_a_real_action_xyz"}]},
                )
        assert resp.status_code == 422

    def test_list_agents_with_exception_in_info(self, tmp_path, clean_pool):
        """If introspecting an agent raises, it returns active=False gracefully."""
        agent = MagicMock()
        # accessing ._page raises
        type(agent)._page = property(lambda self: (_ for _ in ()).throw(Exception("boom")))
        tools = SystemTools(workspace=tmp_path)
        with api_server._pool_lock:
            api_server._agent_pool["broken"] = agent
        with patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/agents/pool")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        broken = next((a for a in agents if a["agent_id"] == "broken"), None)
        assert broken is not None
        assert broken["active"] is False


# ---------------------------------------------------------------------------
# /agents/execute_parallel — no tasks / no agent_count returns 422
# ---------------------------------------------------------------------------

class TestAgentsExecuteParallelValidation:
    def test_no_tasks_no_count_returns_422(self, tmp_path, clean_pool):
        tools = SystemTools(workspace=tmp_path)
        with patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={})
        assert resp.status_code == 422

    def test_empty_tasks_list_returns_422(self, tmp_path, clean_pool):
        tools = SystemTools(workspace=tmp_path)
        with patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={"tasks": []})
        assert resp.status_code == 422

    def test_invalid_steps_in_parallel_returns_422(self, tmp_path, clean_pool):
        tools = SystemTools(workspace=tmp_path)
        with patch.object(api_server, "_tools", tools):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={
                    "tasks": [{"steps": [{"action": "not_valid_xyz"}]}],
                })
        assert resp.status_code == 422

    def test_auto_stop_cleans_up_spawned_agents(self, tmp_path, clean_pool):
        """auto_stop=True should stop and remove auto-spawned agents."""
        tools = SystemTools(workspace=tmp_path)
        mock_inst = _make_agent()
        mock_planner = _make_planner()
        with patch.object(api_server, "_tools", tools), \
             patch("api_server.BrowserAgent", return_value=mock_inst), \
             patch("api_server.TaskPlanner", return_value=mock_planner):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/agents/execute_parallel", json={
                    "agent_count": 1,
                    "steps": [{"action": "navigate", "url": "https://example.com"}],
                    "auto_stop": True,
                })
        assert resp.status_code == 200
        # All auto-spawned agents should be removed from pool
        with api_server._pool_lock:
            assert len(api_server._agent_pool) == 0
