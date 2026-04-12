"""
Tests for TaskPlanner.execute(), TaskPlanner._execute_step(), TaskPlanner.run(),
_selectors_for(), _call_openai(), _call_ollama(), and _ollama_running().

All browser interactions use MagicMock so no real Playwright process is needed.
System-tool actions use a real SystemTools pointed at a temp directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from system_tools import SystemTools
from task_planner import (
    StepValidationError,
    TaskPlanner,
    _selectors_for,
    validate_steps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> MagicMock:
    """Return a MagicMock that passes the browser agent guard."""
    agent = MagicMock()
    agent.navigate.return_value = {"url": "https://example.com", "title": "Example"}
    agent.click.return_value    = {"clicked": "button", "url": "https://example.com"}
    agent.fill.return_value     = {"filled": "value", "selector": "input"}
    agent.type_text.return_value = {"typed": "text", "selector": "input"}
    agent.press_key.return_value = {"key": "Enter"}
    agent.wait_for_selector.return_value = {"visible": "div"}
    agent.wait_for_load_state.return_value = {"state": "networkidle"}
    agent.close_popups.return_value = {"dismissed": [], "count": 0}
    agent.scroll.return_value   = {"scrolled": {"x": 0, "y": 500}}
    agent.scroll_to_element.return_value = {"scrolled_to": "h1"}
    agent.screenshot.return_value = {"path": None, "base64": "abc="}
    agent.hover.return_value    = {"hovered": "a"}
    agent.select_option.return_value = {"selected": "opt", "selector": "select"}
    agent.evaluate.return_value = 42
    agent.get_text.return_value = "page text"
    agent.extract_links.return_value = {"links": [{"text": "Link", "href": "/link"}], "count": 1}
    agent.extract_table.return_value = {"headers": ["A", "B"], "rows": [{"A": "1", "B": "2"}], "count": 1}
    setattr(agent, "assert_text", MagicMock(return_value={"found": True, "text": "hello", "selector": "body"}))
    setattr(agent, "assert_url",  MagicMock(return_value={"url": "https://example.com", "pattern": "example", "matched": True}))
    agent.wait_text.return_value = {"found": True, "text": "loaded", "selector": "body"}
    agent.get_cookies.return_value = [{"name": "session", "value": "abc"}]
    agent.add_cookies.return_value = None
    agent.new_tab.return_value  = {"tab_index": 1, "url": "about:blank", "title": ""}
    agent.switch_tab.return_value = {"tab_index": 0, "url": "https://example.com", "title": "Example"}
    agent.close_tab.return_value = {"closed_index": 1, "remaining_tabs": 1}
    agent.list_tabs.return_value = {"tabs": [{"index": 0, "url": "https://example.com", "active": True}], "count": 1}
    return agent


def _planner_no_llm(tmp_path: Path | None = None) -> TaskPlanner:
    """Return a TaskPlanner with no LLM, optionally wired to a temp workspace."""
    with patch.object(TaskPlanner, "_detect_llm", return_value=None):
        planner = TaskPlanner()
    if tmp_path is not None:
        planner._system_tools = SystemTools(workspace=tmp_path)
    return planner


# ---------------------------------------------------------------------------
# _selectors_for
# ---------------------------------------------------------------------------

class TestSelectorsFor:
    def test_google(self) -> None:
        sels = _selectors_for("www.google.com")
        assert "search_input" in sels
        assert sels["search_input"]  # non-empty selector

    def test_bing(self) -> None:
        sels = _selectors_for("www.bing.com")
        assert "search_input" in sels

    def test_unknown_host_returns_empty(self) -> None:
        sels = _selectors_for("www.unknown-site.example")
        assert sels == {}

    def test_partial_domain_match(self) -> None:
        sels = _selectors_for("en.wikipedia.org")
        assert "search_input" in sels


# ---------------------------------------------------------------------------
# _ollama_running
# ---------------------------------------------------------------------------

class TestOllamaRunning:
    def test_reachable(self) -> None:
        from task_planner import _ollama_running
        with patch("urllib.request.urlopen", return_value=MagicMock()):
            assert _ollama_running() is True

    def test_unreachable(self) -> None:
        from task_planner import _ollama_running
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert _ollama_running() is False


# ---------------------------------------------------------------------------
# _call_openai
# ---------------------------------------------------------------------------

class TestCallOpenAI:
    def _good_response(self, content: str) -> MagicMock:
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def _mock_openai(self, content: str):
        """Context manager that mocks the openai module."""
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._good_response(content)
        mock_module.OpenAI.return_value = mock_client
        return patch.dict("sys.modules", {"openai": mock_module})

    def test_returns_list_directly(self, monkeypatch) -> None:
        from task_planner import _call_openai
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        steps = [{"action": "navigate", "url": "https://example.com"}]
        with self._mock_openai(json.dumps(steps)):
            result = _call_openai("go to example.com")
        assert result[0]["action"] == "navigate"

    def test_returns_wrapped_steps(self, monkeypatch) -> None:
        from task_planner import _call_openai
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = {"steps": [{"action": "navigate", "url": "https://example.com"}]}
        with self._mock_openai(json.dumps(payload)):
            result = _call_openai("go somewhere")
        assert result[0]["action"] == "navigate"

    def test_bad_shape_raises(self, monkeypatch) -> None:
        from task_planner import _call_openai
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with self._mock_openai(json.dumps({"unrecognized": "shape"})):
            with pytest.raises(StepValidationError):
                _call_openai("do something")

    def test_accepts_plan_key(self, monkeypatch) -> None:
        from task_planner import _call_openai
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = {"plan": [{"action": "close_popups"}]}
        with self._mock_openai(json.dumps(payload)):
            result = _call_openai("close popups")
        assert result[0]["action"] == "close_popups"


# ---------------------------------------------------------------------------
# _call_ollama
# ---------------------------------------------------------------------------

class TestCallOllama:
    def _make_response(self, content: str) -> MagicMock:
        resp_mock = MagicMock()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        resp_mock.read.return_value = json.dumps(
            {"message": {"content": content}}
        ).encode()
        return resp_mock

    def test_parses_json_array(self, monkeypatch) -> None:
        from task_planner import _call_ollama
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        steps = [{"action": "navigate", "url": "https://example.com"}]
        with patch("urllib.request.urlopen", return_value=self._make_response(json.dumps(steps))):
            result = _call_ollama("go to example.com")
        assert result[0]["action"] == "navigate"

    def test_strips_markdown_fences(self, monkeypatch) -> None:
        from task_planner import _call_ollama
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        content = "```json\n[{\"action\": \"close_popups\"}]\n```"
        with patch("urllib.request.urlopen", return_value=self._make_response(content)):
            result = _call_ollama("close popups")
        assert result[0]["action"] == "close_popups"

    def test_no_json_array_raises(self, monkeypatch) -> None:
        from task_planner import _call_ollama
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        with patch("urllib.request.urlopen", return_value=self._make_response("No JSON here")):
            with pytest.raises(StepValidationError, match="no JSON array"):
                _call_ollama("something")


# ---------------------------------------------------------------------------
# TaskPlanner.execute() — browser actions
# ---------------------------------------------------------------------------

class TestExecuteBrowserActions:
    """Tests that execute() correctly delegates each browser action to the agent."""

    def test_navigate(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "navigate", "url": "https://example.com"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.navigate.assert_called_once_with("https://example.com", wait_until="domcontentloaded")

    def test_click(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "click", "selector": "#btn"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.click.assert_called_once_with("#btn", timeout=None)

    def test_fill(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "fill", "selector": "input", "value": "hello"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.fill.assert_called_once_with("input", "hello")

    def test_type(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "type", "selector": "input", "text": "world"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.type_text.assert_called_once()

    def test_press(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "press", "key": "Enter"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.press_key.assert_called_once_with("Enter")

    def test_wait_selector(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "wait_selector", "selector": ".ready"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"

    def test_wait_state(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "wait_state", "state": "networkidle"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.wait_for_load_state.assert_called_once_with("networkidle")

    def test_close_popups(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "close_popups"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.close_popups.assert_called_once()

    def test_scroll(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "scroll", "y": 300}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.scroll.assert_called_once_with(x=0, y=300)

    def test_scroll_to_element(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "scroll_to_element", "selector": "h1"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.scroll_to_element.assert_called_once_with("h1")

    def test_screenshot_no_path(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "screenshot"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        # When no path, as_base64 should be True
        agent.screenshot.assert_called_once_with(path=None, full_page=False, as_base64=True)

    def test_hover(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "hover", "selector": "a.link"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.hover.assert_called_once_with("a.link")

    def test_select_option(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "select_option", "selector": "select", "value": "opt1"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.select_option.assert_called_once_with("select", "opt1")

    def test_evaluate(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "evaluate", "script": "document.title"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.evaluate.assert_called_once_with("document.title")

    def test_get_text_updates_last(self, tmp_path: Path) -> None:
        """get_text result should be used to interpolate {{last}} in later steps."""
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        agent.get_text.return_value = "extracted text"
        steps = validate_steps([
            {"action": "get_text", "selector": "body"},
            {"action": "write_file", "path": "out.txt", "content": "{{last}}"},
        ])
        results = planner.execute(steps, agent)
        assert all(r["status"] == "ok" for r in results)
        # The file should contain the extracted text
        content = (tmp_path / "out.txt").read_text()
        assert content == "extracted text"

    def test_extract_links(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "extract_links", "selector": "a", "limit": 50}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.extract_links.assert_called_once_with(selector="a", limit=50)

    def test_extract_table(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "extract_table"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"

    def test_assert_text(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "assert_text", "text": "hello", "selector": "body"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"

    def test_assert_url(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "assert_url", "pattern": "example"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.assert_url.assert_called_once_with("example")

    def test_wait_text(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "wait_text", "text": "loaded"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"

    def test_new_tab(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "new_tab"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.new_tab.assert_called_once_with(url=None)

    def test_switch_tab(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "switch_tab", "index": "0"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"

    def test_close_tab(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "close_tab"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"

    def test_list_tabs(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "list_tabs"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"


# ---------------------------------------------------------------------------
# TaskPlanner.execute() — system-tool actions (real SystemTools)
# ---------------------------------------------------------------------------

class TestExecuteSystemActions:
    def test_write_file(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "write_file", "path": "test.txt", "content": "hello"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        assert (tmp_path / "test.txt").read_text() == "hello"

    def test_append_file(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        (tmp_path / "append.txt").write_text("line1\n")
        steps = validate_steps([{"action": "append_file", "path": "append.txt", "content": "line2\n"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        assert "line2" in (tmp_path / "append.txt").read_text()

    def test_read_file(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        (tmp_path / "read.txt").write_text("content123")
        steps = validate_steps([{"action": "read_file", "path": "read.txt"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        assert results[0]["result"]["content"] == "content123"

    def test_read_file_updates_last(self, tmp_path: Path) -> None:
        """read_file content should be available as {{last}} in the next step."""
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        (tmp_path / "source.txt").write_text("source content")
        steps = validate_steps([
            {"action": "read_file", "path": "source.txt"},
            {"action": "write_file", "path": "dest.txt", "content": "{{last}}"},
        ])
        results = planner.execute(steps, agent)
        assert all(r["status"] == "ok" for r in results)
        assert (tmp_path / "dest.txt").read_text() == "source content"

    def test_list_dir(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        (tmp_path / "file.txt").write_text("x")
        steps = validate_steps([{"action": "list_dir"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        names = [e["name"] for e in results[0]["result"]["entries"]]
        assert "file.txt" in names

    def test_run_python(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "run_python", "code": "print('py_output')"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        assert "py_output" in results[0]["result"]["stdout"]

    def test_run_python_updates_last(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([
            {"action": "run_python", "code": "print('hello from py')"},
            {"action": "write_file", "path": "py_out.txt", "content": "{{last}}"},
        ])
        results = planner.execute(steps, agent)
        assert all(r["status"] == "ok" for r in results)
        content = (tmp_path / "py_out.txt").read_text()
        assert "hello from py" in content

    def test_run_shell(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "run_shell", "command": "echo shell_out"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        assert "shell_out" in results[0]["result"]["stdout"]

    def test_save_cookies(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([{"action": "save_cookies", "path": "cookies.json"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        cookies = json.loads((tmp_path / "cookies.json").read_text())
        assert len(cookies) == 1
        assert cookies[0]["name"] == "session"

    def test_load_cookies(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        cookies = [{"name": "tok", "value": "xyz"}]
        (tmp_path / "cookies.json").write_text(json.dumps(cookies))
        steps = validate_steps([{"action": "load_cookies", "path": "cookies.json"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        agent.add_cookies.assert_called_once_with(cookies)


# ---------------------------------------------------------------------------
# TaskPlanner.execute() — retry logic, stop_on_error, callbacks
# ---------------------------------------------------------------------------

class TestExecuteRetryAndControl:
    def test_stop_on_error_true(self, tmp_path: Path) -> None:
        """A failed step should stop execution when stop_on_error=True."""
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        agent.click.side_effect = Exception("element not found")
        steps = validate_steps([
            {"action": "click", "selector": "#missing"},
            {"action": "navigate", "url": "https://example.com"},
        ])
        results = planner.execute(steps, agent, stop_on_error=True)
        assert len(results) == 1
        assert results[0]["status"] == "error"
        agent.navigate.assert_not_called()

    def test_stop_on_error_false(self, tmp_path: Path) -> None:
        """With stop_on_error=False all steps run even on error."""
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        agent.click.side_effect = Exception("element not found")
        steps = validate_steps([
            {"action": "click", "selector": "#missing"},
            {"action": "navigate", "url": "https://example.com"},
        ])
        results = planner.execute(steps, agent, stop_on_error=False)
        assert len(results) == 2
        assert results[0]["status"] == "error"
        assert results[1]["status"] == "ok"

    def test_retry_succeeds_on_second_attempt(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        call_count = [0]
        def flaky_click(selector, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("transient error")
            return {"clicked": selector, "url": "https://example.com"}
        agent.click.side_effect = flaky_click

        steps = [{"action": "click", "selector": "#btn", "retry": 2, "retry_delay": 0}]
        # Manually build without validate_steps since retry is set directly
        with patch("time.sleep"):
            results = planner.execute(steps, agent)
        assert results[0]["status"] == "ok"
        assert call_count[0] == 2

    def test_retry_exhausted_gives_error(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        agent.click.side_effect = Exception("always fails")
        steps = [{"action": "click", "selector": "#btn", "retry": 1, "retry_delay": 0}]
        with patch("time.sleep"):
            results = planner.execute(steps, agent)
        assert results[0]["status"] == "error"
        assert "always fails" in results[0]["error"]

    def test_step_callback_called(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        collected = []
        steps = validate_steps([{"action": "close_popups"}])
        planner.execute(steps, agent, step_callback=collected.append)
        assert len(collected) == 1
        assert collected[0]["action"] == "close_popups"

    def test_step_start_callback_called(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        starts = []
        steps = validate_steps([{"action": "close_popups"}, {"action": "navigate", "url": "https://example.com"}])
        planner.execute(steps, agent, step_start_callback=lambda i, a: starts.append((i, a)))
        assert starts == [(0, "close_popups"), (1, "navigate")]

    def test_error_step_recorded_correctly(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        agent.close_popups.side_effect = RuntimeError("popup error")
        steps = validate_steps([{"action": "close_popups"}])
        results = planner.execute(steps, agent)
        assert results[0]["status"] == "error"
        assert "popup error" in results[0]["error"]
        assert results[0]["action"] == "close_popups"
        assert results[0]["step"] == 0


# ---------------------------------------------------------------------------
# TaskPlanner.execute() — {{last}} interpolation
# ---------------------------------------------------------------------------

class TestLastInterpolation:
    def test_extract_links_last_is_json(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([
            {"action": "extract_links"},
            {"action": "write_file", "path": "links.json", "content": "{{last}}"},
        ])
        results = planner.execute(steps, agent)
        assert all(r["status"] == "ok" for r in results)
        raw = (tmp_path / "links.json").read_text()
        parsed = json.loads(raw)
        assert "links" in parsed

    def test_run_shell_stdout_as_last(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        steps = validate_steps([
            {"action": "run_shell", "command": "echo hello_shell"},
            {"action": "write_file", "path": "shell_out.txt", "content": "{{last}}"},
        ])
        results = planner.execute(steps, agent)
        assert all(r["status"] == "ok" for r in results)
        assert "hello_shell" in (tmp_path / "shell_out.txt").read_text()


# ---------------------------------------------------------------------------
# TaskPlanner.run() — top-level plan + execute
# ---------------------------------------------------------------------------

class TestTaskPlannerRun:
    def test_run_success(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        summary = planner.run("go to google and search pytest", agent)
        assert summary["success"] is True
        assert summary["failed_count"] == 0
        assert len(summary["steps"]) > 0

    def test_run_planning_failure(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        summary = planner.run("completely unknown intent xyz999", agent)
        assert summary["success"] is False
        assert "error" in summary

    def test_run_step_failure(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        agent.navigate.side_effect = Exception("navigation failed")
        summary = planner.run("open example.com", agent)
        assert summary["success"] is False
        assert summary["failed_count"] >= 1

    def test_run_writes_log(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        summary = planner.run("go to google and search cats", agent, log_path="run.log")
        assert summary["success"] is True
        log = json.loads((tmp_path / "run.log").read_text())
        assert "intent" in log
        assert "success" in log
        assert "timestamp" in log

    def test_run_intent_preserved_in_summary(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        summary = planner.run("go to google and search python", agent)
        assert summary["intent"] == "go to google and search python"


# ---------------------------------------------------------------------------
# TaskPlanner.plan() with skill registry
# ---------------------------------------------------------------------------

class TestPlanWithSkillRegistry:
    def test_skill_match_used(self) -> None:
        from skills import SkillDef, SkillRegistry

        skill = SkillDef(
            name="my_skill",
            triggers=["do the thing {{x}}"],
            steps=[{"action": "navigate", "url": "https://example.com/{{x}}"}],
        )
        reg = SkillRegistry()
        reg.register(skill)
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            planner = TaskPlanner(skill_registry=reg)
        steps = planner.plan("do the thing hello")
        assert steps[0]["action"] == "navigate"
        assert "hello" in steps[0]["url"]

    def test_skill_fallback_to_template_on_no_match(self) -> None:
        from skills import SkillRegistry

        reg = SkillRegistry()
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            planner = TaskPlanner(skill_registry=reg)
        # Should fall through to template since no skills match
        steps = planner.plan("go to google and search cats")
        assert steps[0]["action"] == "navigate"

    def test_skill_builder_exception_falls_through(self) -> None:
        """When skill step resolution fails, plan() should fall back to template matching."""
        from skills import SkillDef, SkillRegistry

        skill = SkillDef(
            name="broken_skill",
            triggers=["search for {{q}} on google"],
            steps=[{"action": "navigate", "url": "https://google.com"}],
        )
        reg = SkillRegistry()
        reg.register(skill)
        with patch.object(TaskPlanner, "_detect_llm", return_value=None), \
             patch("skills.resolve_steps", side_effect=Exception("broken")):
            planner = TaskPlanner(skill_registry=reg)
        steps = planner.plan("go to google and search cats")
        assert steps[0]["action"] == "navigate"


# ---------------------------------------------------------------------------
# Additional template tests
# ---------------------------------------------------------------------------

class TestTemplateCoverage:
    def _planner(self) -> TaskPlanner:
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            return TaskPlanner()

    def test_navigate_with_https_url(self) -> None:
        planner = self._planner()
        steps = planner.plan("navigate to https://news.ycombinator.com")
        assert steps[0]["action"] == "navigate"
        assert "ycombinator" in steps[0]["url"]

    def test_navigate_visit(self) -> None:
        planner = self._planner()
        steps = planner.plan("visit https://example.com")
        assert steps[0]["url"] == "https://example.com"

    def test_scrape_and_save_bare_domain(self) -> None:
        planner = self._planner()
        steps = planner.plan("collect text from example.com and save to result.txt")
        actions = [s["action"] for s in steps]
        assert "navigate" in actions
        assert "get_text" in actions
        assert "write_file" in actions
        write = next(s for s in steps if s["action"] == "write_file")
        assert write["path"] == "result.txt"

    def test_youtube_search_variant(self) -> None:
        planner = self._planner()
        steps = planner.plan("search cats on youtube")
        assert steps[0]["url"] == "https://www.youtube.com"

    def test_wikipedia_search_variant(self) -> None:
        planner = self._planner()
        steps = planner.plan("search Python on wikipedia")
        assert steps[0]["url"] == "https://en.wikipedia.org"

    def test_detect_llm_openai(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        llm = TaskPlanner._detect_llm()
        assert llm == "openai"

    def test_detect_llm_ollama_env(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        llm = TaskPlanner._detect_llm()
        assert llm == "ollama"

    def test_detect_llm_none(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        with patch("task_planner._ollama_running", return_value=False):
            llm = TaskPlanner._detect_llm()
        assert llm is None


# ---------------------------------------------------------------------------
# Unknown action in _execute_step raises ValueError
# ---------------------------------------------------------------------------

class TestExecuteStepUnknownAction:
    def test_unknown_action_raises(self, tmp_path: Path) -> None:
        planner = _planner_no_llm(tmp_path)
        agent = _make_agent()
        with pytest.raises(ValueError, match="Unknown action"):
            planner._execute_step(agent, {"action": "fly_to_mars"})
