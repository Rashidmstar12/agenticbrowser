"""
tests/test_gui.py — Smoke tests for GUI routes and new API endpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures — same pattern as test_api.py
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_agent():
    agent = MagicMock()
    agent.is_active = True
    agent.page_url.return_value   = "https://example.com"
    agent.page_title.return_value = "Example"
    return agent


@pytest.fixture()
def client(mock_agent):
    import api_server
    with patch.object(api_server, "_agent", mock_agent), \
         patch.object(api_server, "_planner", MagicMock()), \
         patch.object(api_server, "_tools",   MagicMock()):
        yield TestClient(api_server.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GUI routes
# ---------------------------------------------------------------------------

def test_root_redirect(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (301, 302, 307, 308)
    assert res.headers["location"] in ("/ui", "http://testserver/ui")


def test_ui_serves_html(client):
    res = client.get("/ui")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    body = res.text
    assert "AgenticBrowser" in body
    assert "<script" in body


# ---------------------------------------------------------------------------
# GET /doctor
# ---------------------------------------------------------------------------

def test_doctor_returns_ok_structure(client):
    fake_checks = [
        MagicMock(to_dict=lambda: {"name": "python_version", "status": "ok",
                                   "message": "Python 3.12", "fixed": False}),
        MagicMock(to_dict=lambda: {"name": "workspace",     "status": "ok",
                                   "message": "exists",      "fixed": False}),
    ]
    with patch("api_server.run_checks", return_value=fake_checks):
        res = client.get("/doctor")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("ok", "degraded")
    assert isinstance(data["checks"], list)
    assert len(data["checks"]) == 2


def test_doctor_degraded_on_fail(client):
    fake_checks = [
        MagicMock(status="fail",
                  to_dict=lambda: {"name": "chromium", "status": "fail",
                                   "message": "not installed", "fixed": False}),
    ]
    with patch("api_server.run_checks", return_value=fake_checks):
        res = client.get("/doctor")
    assert res.status_code == 200
    assert res.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# Skills routes
# ---------------------------------------------------------------------------

def test_skills_list_empty(client):
    with patch("api_server.get_default_registry") as mock_reg:
        mock_reg.return_value.list_skills.return_value = []
        mock_reg.return_value.__len__ = lambda s: 0
        res = client.get("/skills")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 0
    assert data["skills"] == []


def test_skills_load_success(client):
    mock_skill = MagicMock()
    mock_skill.name = "test_skill"
    with patch("api_server.get_default_registry") as mock_reg:
        mock_reg.return_value.load_from_remote_source.return_value = [mock_skill]
        res = client.post("/skills/load", json={"source": "gh:example/skills"})
    assert res.status_code == 200
    data = res.json()
    assert data["loaded"] == 1
    assert "test_skill" in data["skills"]


def test_skills_load_invalid_source(client):
    from skills import SkillLoadError
    with patch("api_server.get_default_registry") as mock_reg:
        mock_reg.return_value.load_from_remote_source.side_effect = SkillLoadError("bad source")
        res = client.post("/skills/load", json={"source": "bad://source"})
    assert res.status_code == 422


def test_skills_get_found(client):
    mock_skill = MagicMock()
    mock_skill.to_dict.return_value = {
        "name": "my_skill", "description": "test", "version": "1.0",
        "author": "", "triggers": [], "parameters": {}, "steps": [], "source": "",
    }
    with patch("api_server.get_default_registry") as mock_reg:
        mock_reg.return_value.get.return_value = mock_skill
        res = client.get("/skills/my_skill")
    assert res.status_code == 200
    assert res.json()["name"] == "my_skill"


def test_skills_get_not_found(client):
    with patch("api_server.get_default_registry") as mock_reg:
        mock_reg.return_value.get.return_value = None
        res = client.get("/skills/nonexistent")
    assert res.status_code == 404


def test_skills_delete_found(client):
    with patch("api_server.get_default_registry") as mock_reg:
        mock_reg.return_value.unregister.return_value = True
        res = client.delete("/skills/my_skill")
    assert res.status_code == 200
    assert res.json()["unloaded"] == "my_skill"


def test_skills_delete_not_found(client):
    with patch("api_server.get_default_registry") as mock_reg:
        mock_reg.return_value.unregister.return_value = False
        res = client.delete("/skills/nonexistent")
    assert res.status_code == 404
