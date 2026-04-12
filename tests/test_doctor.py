"""
tests/test_doctor.py — Unit tests for doctor.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from doctor import (
    CheckResult,
    FailureReport,
    _check_chromium,
    _check_ollama,
    _check_openai,
    _check_packages,
    _check_port,
    _check_python_version,
    _check_workspace,
    explain_failure,
    run_checks,
)

# ---------------------------------------------------------------------------
# _check_python_version
# ---------------------------------------------------------------------------

def test_python_version_ok(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 12, 0, "final", 0))
    r = _check_python_version()
    assert r.status == "ok"
    assert "3.12" in r.message


def test_python_version_warn(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 8, 0, "final", 0))
    r = _check_python_version()
    assert r.status == "warn"
    assert "3.8" in r.message


# ---------------------------------------------------------------------------
# _check_packages
# ---------------------------------------------------------------------------

def test_packages_all_installed():
    with patch("importlib.util.find_spec", return_value=MagicMock()):
        results = _check_packages()
    assert all(r.status == "ok" for r in results)


def test_packages_missing_no_fix():
    with patch("importlib.util.find_spec", return_value=None):
        results = _check_packages(fix=False)
    assert all(r.status == "fail" for r in results)


def test_packages_missing_with_fix_success():
    with patch("importlib.util.find_spec", return_value=None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        results = _check_packages(fix=True)
    assert all(r.fixed for r in results)
    assert all(r.status == "ok" for r in results)


def test_packages_missing_with_fix_fail():
    with patch("importlib.util.find_spec", return_value=None), \
         patch("subprocess.run", side_effect=OSError("pip broken")):
        results = _check_packages(fix=True)
    assert all(r.status == "fail" for r in results)


# ---------------------------------------------------------------------------
# _check_chromium
# ---------------------------------------------------------------------------

def test_chromium_playwright_missing():
    with patch("importlib.util.find_spec", return_value=None):
        r = _check_chromium()
    assert r.status == "fail"
    assert "playwright" in r.message.lower()


def test_chromium_installed():
    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="chromium 1.0\n")
        r = _check_chromium()
    assert r.status == "ok"


def test_chromium_not_installed_no_fix():
    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        r = _check_chromium(fix=False)
    assert r.status == "fail"


def test_chromium_fix_success():
    call_count = [0]
    def fake_run(cmd, **_):
        call_count[0] += 1
        if "--list" in cmd:
            return MagicMock(returncode=0, stdout="")
        return MagicMock(returncode=0)

    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("subprocess.run", side_effect=fake_run):
        r = _check_chromium(fix=True)
    assert r.fixed
    assert r.status == "ok"


# ---------------------------------------------------------------------------
# _check_workspace
# ---------------------------------------------------------------------------

def test_workspace_exists(tmp_path):
    r = _check_workspace(str(tmp_path))
    assert r.status == "ok"


def test_workspace_missing_no_fix(tmp_path):
    r = _check_workspace(str(tmp_path / "nope"))
    assert r.status == "fail"


def test_workspace_missing_with_fix(tmp_path):
    target = tmp_path / "new_ws"
    r = _check_workspace(str(target), fix=True)
    assert r.status == "ok"
    assert r.fixed
    assert target.is_dir()


# ---------------------------------------------------------------------------
# _check_openai
# ---------------------------------------------------------------------------

def test_openai_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    r = _check_openai()
    assert r.status == "ok"


def test_openai_not_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = _check_openai()
    assert r.status == "warn"


# ---------------------------------------------------------------------------
# _check_ollama
# ---------------------------------------------------------------------------

def test_ollama_not_configured(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    r = _check_ollama()
    assert r.status == "warn"


def test_ollama_reachable(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    resp_mock = MagicMock()
    resp_mock.__enter__ = lambda s: s
    resp_mock.__exit__ = MagicMock(return_value=False)
    resp_mock.status = 200
    with patch("urllib.request.urlopen", return_value=resp_mock):
        r = _check_ollama()
    assert r.status == "ok"


def test_ollama_unreachable(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        r = _check_ollama()
    assert r.status == "fail"


# ---------------------------------------------------------------------------
# _check_port
# ---------------------------------------------------------------------------

def test_port_free():
    import socket
    with patch.object(socket.socket, "connect_ex", return_value=1):
        r = _check_port()
    assert r.status == "ok"


def test_port_in_use():
    import socket
    with patch.object(socket.socket, "connect_ex", return_value=0):
        r = _check_port()
    assert r.status == "warn"


# ---------------------------------------------------------------------------
# run_checks
# ---------------------------------------------------------------------------

def test_run_checks_returns_list():
    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("subprocess.run") as mock_run, \
         patch("socket.socket"):
        mock_run.return_value = MagicMock(returncode=0, stdout="chromium\n")
        checks = run_checks(workspace=".")
    assert isinstance(checks, list)
    assert len(checks) >= 7
    assert all(isinstance(c, CheckResult) for c in checks)


def test_check_result_to_dict():
    c = CheckResult("python_version", "ok", "Python 3.12")
    d = c.to_dict()
    assert d["name"]    == "python_version"
    assert d["status"]  == "ok"
    assert d["message"] == "Python 3.12"
    assert d["fixed"]   is False


# ---------------------------------------------------------------------------
# explain_failure
# ---------------------------------------------------------------------------

def test_explain_timeout():
    r = explain_failure("TimeoutError waiting for selector", action="click", step_index=2)
    assert isinstance(r, FailureReport)
    assert "timeout" in r.reason.lower()
    assert any("selector" in s.lower() or "query" in s.lower() for s in r.suggestions)


def test_explain_net_err():
    r = explain_failure("net::ERR_NAME_NOT_RESOLVED https://bad.host")
    assert "navigation" in r.reason.lower() or "dns" in r.reason.lower() or "ERR" in r.reason


def test_explain_missing_package():
    r = explain_failure("No module named 'playwright'")
    assert any("pip" in s for s in r.suggestions)


def test_explain_unknown_error():
    r = explain_failure("some completely unknown failure xyz")
    assert r.reason
    assert len(r.suggestions) >= 1


def test_explain_rerun_cmd():
    r = explain_failure("TimeoutError", rerun_cmd="python local_runner.py --intent 'test'")
    assert "local_runner.py" in r.rerun_hint


def test_explain_intent_in_suggestions():
    r = explain_failure("TimeoutError", intent="find the search box")
    assert any("find the search box" in s for s in r.suggestions)
