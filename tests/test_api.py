from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from api_server import app

client = TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── /schema ───────────────────────────────────────────────────────────────────

def test_schema():
    response = client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    assert "actions" in data
    assert len(data["actions"]) == 32


def test_schema_contains_navigate():
    response = client.get("/schema")
    assert "navigate" in response.json()["actions"]


# ── /validate ─────────────────────────────────────────────────────────────────

def test_validate_valid_task():
    response = client.post("/validate", json={
        "name": "test",
        "steps": [{"action": "navigate", "params": {"url": "https://example.com"}}],
    })
    assert response.status_code == 200
    assert response.json() == {"valid": True}


def test_validate_invalid_action():
    response = client.post("/validate", json={
        "name": "test",
        "steps": [{"action": "fly_to_moon", "params": {}}],
    })
    assert response.status_code == 422


def test_validate_missing_name():
    response = client.post("/validate", json={
        "name": "",
        "steps": [{"action": "navigate", "params": {}}],
    })
    assert response.status_code == 422


def test_validate_empty_steps():
    response = client.post("/validate", json={
        "name": "test",
        "steps": [],
    })
    assert response.status_code == 422


def test_validate_multiple_steps():
    response = client.post("/validate", json={
        "name": "multi",
        "steps": [
            {"action": "navigate", "params": {}},
            {"action": "click", "params": {}},
        ],
    })
    assert response.status_code == 200


# ── /task/run ─────────────────────────────────────────────────────────────────

def test_task_run_success():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(return_value=[{"success": True}])
        response = client.post("/task/run", json={
            "name": "test",
            "steps": [{"action": "navigate", "params": {"url": "https://example.com"}}],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["name"] == "test"


def test_task_run_returns_results():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(return_value=[{"success": True, "url": "https://example.com"}])
        response = client.post("/task/run", json={
            "name": "t",
            "steps": [{"action": "get_url", "params": {}}],
        })
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1


def test_task_run_step_failure_returns_422():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(return_value=[{"success": False, "error": "oops"}])
        response = client.post("/task/run", json={
            "name": "test",
            "steps": [{"action": "navigate", "params": {}}],
        })
        assert response.status_code == 422


def test_task_run_stop_called_on_success():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(return_value=[{"success": True}])
        client.post("/task/run", json={
            "name": "t",
            "steps": [{"action": "navigate", "params": {}}],
        })
        instance.stop.assert_called_once()


def test_task_run_stop_called_on_failure():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(side_effect=RuntimeError("crash"))
        try:
            client.post("/task/run", json={
                "name": "t",
                "steps": [{"action": "navigate", "params": {}}],
            })
        except Exception:
            pass
        instance.stop.assert_called_once()


def test_task_run_summary_in_response():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(return_value=[{"success": True}])
        response = client.post("/task/run", json={
            "name": "t",
            "steps": [{"action": "navigate", "params": {}}],
        })
        assert "summary" in response.json()


def test_task_run_headless_true():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(return_value=[{"success": True}])
        client.post("/task/run", json={
            "name": "t",
            "steps": [{"action": "navigate", "params": {}}],
        })
        MockAgent.assert_called_with(headless=True)


def test_task_run_passes_steps():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(return_value=[{"success": True}])
        client.post("/task/run", json={
            "name": "t",
            "steps": [{"action": "navigate", "params": {"url": "https://example.com"}}],
        })
        call_args = instance.run_task.call_args[0][0]
        assert call_args[0]["action"] == "navigate"
        assert call_args[0]["url"] == "https://example.com"


# ── /tasks/schema ─────────────────────────────────────────────────────────────

def test_tasks_schema():
    response = client.get("/tasks/schema")
    assert response.status_code == 200
    data = response.json()
    assert "step_actions" in data
    assert data["count"] == 32


def test_tasks_schema_all_actions():
    response = client.get("/tasks/schema")
    actions = response.json()["step_actions"]
    assert "list_tabs" in actions
    assert "assert_url" in actions


# ── Multiple steps in task/run ─────────────────────────────────────────────────

def test_task_run_multiple_steps():
    with patch("api_server.BrowserAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.run_task = AsyncMock(return_value=[
            {"success": True},
            {"success": True},
        ])
        response = client.post("/task/run", json={
            "name": "multi",
            "steps": [
                {"action": "navigate", "params": {"url": "https://example.com"}},
                {"action": "click", "params": {"selector": "#btn"}},
            ],
        })
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2


def test_validate_returns_error_detail():
    response = client.post("/validate", json={
        "name": "bad",
        "steps": [{"action": "unknown_action", "params": {}}],
    })
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
