"""
Tests for TaskPlanner.execute(), run(), _execute_step(), and LLM backends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_planner import (
    StepValidationError,
    TaskPlanner,
    _call_ollama,
    _call_openai,
    _interpolate_last,
    _ollama_running,
    validate_steps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> MagicMock:
    """Return a MagicMock BrowserAgent that succeeds for all calls."""
    agent = MagicMock()
    agent.navigate.return_value = {"url": "https://example.com", "title": "Test"}
    agent.click.return_value = {"clicked": "button", "url": "https://example.com"}
    agent.fill.return_value = {"filled": "value", "selector": "input"}
    agent.type_text.return_value = {"typed": "text", "selector": "input"}
    agent.press_key.return_value = {"key": "Enter"}
    agent.wait_for_selector.return_value = {"visible": "div"}
    agent.wait_for_load_state.return_value = {"state": "networkidle"}
    agent.close_popups.return_value = {"dismissed": [], "count": 0}
    agent.scroll.return_value = {"scrolled": {"x": 0, "y": 500}}
    agent.scroll_to_element.return_value = {"scrolled_to": "h1"}
    agent.screenshot.return_value = {"path": None, "base64": "abc=="}
    agent.hover.return_value = {"hovered": "a"}
    agent.select_option.return_value = {"selected": "opt", "selector": "select"}
    agent.evaluate.return_value = 42
    agent.get_text.return_value = "some page text"
    agent.extract_links.return_value = {"links": [], "count": 0, "selector": "a"}
    agent.extract_table.return_value = {"headers": [], "rows": [], "count": 0}
    agent.assert_text = MagicMock(return_value={"found": True, "text": "txt", "selector": "body"})
    agent.assert_url = MagicMock(return_value={"url": "https://example.com", "pattern": "example", "matched": True})
    agent.wait_text.return_value = {"found": True, "text": "txt", "selector": "body"}
    agent.new_tab.return_value = {"tab_index": 1, "url": "about:blank", "title": "New Tab"}
    agent.switch_tab.return_value = {"tab_index": 0, "url": "https://example.com", "title": "Test"}
    agent.close_tab.return_value = {"closed_index": 0, "remaining_tabs": 1}
    agent.list_tabs.return_value = {"tabs": [], "count": 0}
    agent.get_cookies.return_value = [{"name": "sid", "value": "abc"}]
    agent.add_cookies.return_value = None
    return agent


def _make_planner(tmp_path=None) -> TaskPlanner:
    """Return a TaskPlanner with optional workspace."""
    with patch.object(TaskPlanner, "_detect_llm", return_value=None):
        planner = TaskPlanner()
    if tmp_path is not None:
        from system_tools import SystemTools
        planner._system_tools = SystemTools(workspace=tmp_path)
    return planner


# ---------------------------------------------------------------------------
# execute() — browser actions
# ---------------------------------------------------------------------------

class TestExecuteBrowserActions:
    def test_navigate_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        results = planner.execute(
            validate_steps([{"action": "navigate", "url": "https://example.com", "wait_until": "load"}]),
            agent,
        )
        assert results[0]["status"] == "ok"
        agent.navigate.assert_called_once_with("https://example.com", wait_until="load")

    def test_navigate_default_wait(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "navigate", "url": "https://example.com"}]), agent)
        agent.navigate.assert_called_once_with("https://example.com", wait_until="domcontentloaded")

    def test_click_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        results = planner.execute(validate_steps([{"action": "click", "selector": "button"}]), agent)
        assert results[0]["status"] == "ok"
        agent.click.assert_called_once_with("button", timeout=None)

    def test_fill_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "fill", "selector": "input", "value": "hello"}]), agent)
        agent.fill.assert_called_once_with("input", "hello")

    def test_type_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "type", "selector": "input", "text": "hello"}]), agent)
        agent.type_text.assert_called_once_with("input", "hello", clear_first=True)

    def test_press_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "press", "key": "Enter"}]), agent)
        agent.press_key.assert_called_once_with("Enter")

    def test_wait_selector_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "wait_selector", "selector": "div"}]), agent)
        agent.wait_for_selector.assert_called_once_with("div", timeout=None)

    def test_wait_state_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "wait_state", "state": "load"}]), agent)
        agent.wait_for_load_state.assert_called_once_with("load")

    def test_close_popups_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "close_popups"}]), agent)
        agent.close_popups.assert_called_once()

    def test_scroll_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "scroll", "x": 0, "y": 300}]), agent)
        agent.scroll.assert_called_once_with(x=0, y=300)

    def test_scroll_to_element_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "scroll_to_element", "selector": "h1"}]), agent)
        agent.scroll_to_element.assert_called_once_with("h1")

    def test_screenshot_with_path(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "screenshot", "path": "shot.png"}]), agent)
        agent.screenshot.assert_called_once_with(path="shot.png", full_page=False, as_base64=False)

    def test_screenshot_no_path_uses_base64(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "screenshot"}]), agent)
        _, kwargs = agent.screenshot.call_args
        assert kwargs["as_base64"] is True

    def test_hover_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "hover", "selector": "a"}]), agent)
        agent.hover.assert_called_once_with("a")

    def test_select_option_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "select_option", "selector": "select", "value": "opt"}]), agent)
        agent.select_option.assert_called_once_with("select", "opt")

    def test_evaluate_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "evaluate", "script": "document.title"}]), agent)
        agent.evaluate.assert_called_once_with("document.title")

    def test_get_text_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.get_text.return_value = "the text"
        results = planner.execute(validate_steps([{"action": "get_text", "selector": "body"}]), agent)
        assert results[0]["status"] == "ok"

    def test_extract_links_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "extract_links", "selector": "a", "limit": 50}]), agent)
        agent.extract_links.assert_called_once_with(selector="a", limit=50)

    def test_extract_table_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "extract_table", "selector": "table", "table_index": 0}]), agent)
        agent.extract_table.assert_called_once_with(selector="table", table_index=0)

    def test_assert_text_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(
            validate_steps([{"action": "assert_text", "text": "Hello", "selector": "body"}]),
            agent,
        )
        agent.assert_text.assert_called_once_with("Hello", selector="body", case_sensitive=False)

    def test_assert_url_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "assert_url", "pattern": "example"}]), agent)
        agent.assert_url.assert_called_once_with("example")

    def test_wait_text_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "wait_text", "text": "Done"}]), agent)
        agent.wait_text.assert_called_once_with("Done", selector="body", timeout=None)

    def test_new_tab_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "new_tab"}]), agent)
        agent.new_tab.assert_called_once_with(url=None)

    def test_switch_tab_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "switch_tab", "index": 0}]), agent)
        agent.switch_tab.assert_called_once_with(0)

    def test_close_tab_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "close_tab"}]), agent)
        agent.close_tab.assert_called_once_with(index=None)

    def test_list_tabs_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        planner.execute(validate_steps([{"action": "list_tabs"}]), agent)
        agent.list_tabs.assert_called_once()


# ---------------------------------------------------------------------------
# execute() — system actions
# ---------------------------------------------------------------------------

class TestExecuteSystemActions:
    def test_write_file(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        results = planner.execute(
            validate_steps([{"action": "write_file", "path": "out.txt", "content": "hello"}]),
            agent,
        )
        assert results[0]["status"] == "ok"
        assert (tmp_path / "out.txt").read_text() == "hello"

    def test_write_file_append_mode(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        (tmp_path / "f.txt").write_text("first\n")
        planner.execute(
            validate_steps([{"action": "write_file", "path": "f.txt", "content": "second", "mode": "a"}]),
            agent,
        )
        assert "second" in (tmp_path / "f.txt").read_text()

    def test_append_file(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        (tmp_path / "log.txt").write_text("existing\n")
        results = planner.execute(
            validate_steps([{"action": "append_file", "path": "log.txt", "content": "appended"}]),
            agent,
        )
        assert results[0]["status"] == "ok"
        assert "appended" in (tmp_path / "log.txt").read_text()

    def test_read_file(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        (tmp_path / "data.txt").write_text("content here")
        results = planner.execute(
            validate_steps([{"action": "read_file", "path": "data.txt"}]),
            agent,
        )
        assert results[0]["status"] == "ok"

    def test_list_dir(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        results = planner.execute(
            validate_steps([{"action": "list_dir", "path": "."}]),
            agent,
        )
        assert results[0]["status"] == "ok"

    def test_run_python(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        results = planner.execute(
            validate_steps([{"action": "run_python", "code": "print('hello')"}]),
            agent,
        )
        assert results[0]["status"] == "ok"

    def test_run_shell(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        results = planner.execute(
            validate_steps([{"action": "run_shell", "command": "echo hi"}]),
            agent,
        )
        assert results[0]["status"] == "ok"

    def test_save_cookies(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        results = planner.execute(
            validate_steps([{"action": "save_cookies", "path": "cookies.json"}]),
            agent,
        )
        assert results[0]["status"] == "ok"
        saved = json.loads((tmp_path / "cookies.json").read_text())
        assert isinstance(saved, list)

    def test_load_cookies(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        cookies = [{"name": "sid", "value": "abc"}]
        (tmp_path / "cookies.json").write_text(json.dumps(cookies))
        results = planner.execute(
            validate_steps([{"action": "load_cookies", "path": "cookies.json"}]),
            agent,
        )
        assert results[0]["status"] == "ok"
        agent.add_cookies.assert_called_once_with(cookies)

    def test_system_tools_lazily_created(self):
        """_get_system_tools should lazily create SystemTools if not set."""
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            planner = TaskPlanner()
        assert planner._system_tools is None
        tools = planner._get_system_tools()
        assert tools is not None
        # Second call returns same instance
        assert planner._get_system_tools() is tools


# ---------------------------------------------------------------------------
# execute() — {{last}} interpolation
# ---------------------------------------------------------------------------

class TestLastInterpolation:
    def test_get_text_updates_last_for_write_file(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.get_text.return_value = "captured text"

        planner.execute(
            validate_steps([
                {"action": "get_text", "selector": "body"},
                {"action": "write_file", "path": "out.txt", "content": "{{last}}"},
            ]),
            agent,
        )
        assert (tmp_path / "out.txt").read_text() == "captured text"

    def test_read_file_updates_last(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        (tmp_path / "source.txt").write_text("file content")

        planner.execute(
            validate_steps([
                {"action": "read_file", "path": "source.txt"},
                {"action": "write_file", "path": "copy.txt", "content": "{{last}}"},
            ]),
            agent,
        )
        assert (tmp_path / "copy.txt").read_text() == "file content"

    def test_run_python_stdout_updates_last(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()

        planner.execute(
            validate_steps([
                {"action": "run_python", "code": "print('py output', end='')"},
                {"action": "write_file", "path": "result.txt", "content": "{{last}}"},
            ]),
            agent,
        )
        assert "py output" in (tmp_path / "result.txt").read_text()

    def test_extract_links_updates_last_as_json(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.extract_links.return_value = {"links": [{"text": "a", "href": "/b"}], "count": 1, "selector": "a"}

        planner.execute(
            validate_steps([
                {"action": "extract_links"},
                {"action": "write_file", "path": "links.txt", "content": "{{last}}"},
            ]),
            agent,
        )
        content = (tmp_path / "links.txt").read_text()
        assert "links" in content

    def test_interpolate_last_function(self):
        step = {"action": "write_file", "path": "{{last}}", "content": "data {{last}}"}
        out = _interpolate_last(step, "result.txt")
        assert out["path"] == "result.txt"
        assert out["content"] == "data result.txt"

    def test_interpolate_last_non_string_unchanged(self):
        step = {"action": "scroll", "x": 0, "y": 500}
        out = _interpolate_last(step, "something")
        assert out["x"] == 0
        assert out["y"] == 500


# ---------------------------------------------------------------------------
# execute() — error handling
# ---------------------------------------------------------------------------

class TestExecuteErrors:
    def test_step_failure_recorded(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.navigate.side_effect = Exception("nav failed")
        results = planner.execute(
            validate_steps([{"action": "navigate", "url": "https://example.com"}]),
            agent,
        )
        assert results[0]["status"] == "error"
        assert "nav failed" in results[0]["error"]

    def test_stop_on_error_default(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.click.side_effect = Exception("click failed")
        results = planner.execute(
            validate_steps([
                {"action": "click", "selector": "button"},
                {"action": "press", "key": "Enter"},
            ]),
            agent,
        )
        assert len(results) == 1
        assert results[0]["status"] == "error"

    def test_stop_on_error_false_continues(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.click.side_effect = Exception("click failed")
        results = planner.execute(
            validate_steps([
                {"action": "click", "selector": "button"},
                {"action": "press", "key": "Enter"},
            ]),
            agent,
            stop_on_error=False,
        )
        assert len(results) == 2
        assert results[0]["status"] == "error"
        assert results[1]["status"] == "ok"

    def test_retry_succeeds_on_third_attempt(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        call_count = [0]

        def flaky(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("transient")
            return {"clicked": "btn", "url": ""}

        agent.click.side_effect = flaky
        results = planner.execute(
            [{"action": "click", "selector": "button", "retry": 2, "retry_delay": 0.0}],
            agent,
        )
        assert results[0]["status"] == "ok"
        assert call_count[0] == 3

    def test_retry_exhausted_records_error(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.click.side_effect = Exception("always fails")
        results = planner.execute(
            [{"action": "click", "selector": "button", "retry": 1, "retry_delay": 0.0}],
            agent,
        )
        assert results[0]["status"] == "error"
        assert agent.click.call_count == 2

    def test_step_callback_called_on_success(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        called = []
        planner.execute(
            validate_steps([{"action": "close_popups"}]),
            agent,
            step_callback=called.append,
        )
        assert len(called) == 1
        assert called[0]["status"] == "ok"

    def test_step_callback_called_on_error(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.close_popups.side_effect = Exception("fail")
        called = []
        planner.execute(
            validate_steps([{"action": "close_popups"}]),
            agent,
            step_callback=called.append,
        )
        assert called[0]["status"] == "error"

    def test_step_start_callback_called(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        starts = []
        planner.execute(
            validate_steps([{"action": "close_popups"}]),
            agent,
            step_start_callback=lambda i, a: starts.append((i, a)),
        )
        assert starts == [(0, "close_popups")]

    def test_unknown_action_raises_and_is_recorded(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        results = planner.execute(
            [{"action": "unknown_xyz"}],
            agent,
        )
        assert results[0]["status"] == "error"
        assert "Unknown action" in results[0]["error"]


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_with_template_intent_succeeds(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        summary = planner.run("go to google and search python", agent)
        assert summary["success"] is True
        assert len(summary["steps"]) > 0
        assert summary["intent"] == "go to google and search python"

    def test_run_unknown_intent_returns_failure(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        summary = planner.run("xyzzy frobulate unknown action 123", agent)
        assert summary["success"] is False
        assert "error" in summary

    def test_run_with_failing_step(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.navigate.side_effect = Exception("nav failed")
        summary = planner.run("go to google and search python", agent)
        assert summary["success"] is False
        assert summary["failed_count"] > 0

    def test_run_with_log_path(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        summary = planner.run("go to google and search python", agent, log_path="run.log")
        assert summary["success"] is True
        log_content = json.loads((tmp_path / "run.log").read_text())
        assert "intent" in log_content
        assert "timestamp" in log_content

    def test_run_stop_on_error_false(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        agent.navigate.side_effect = Exception("nav failed")
        summary = planner.run("go to google and search python", agent, stop_on_error=False)
        # Some steps should run even after failures
        assert len(summary["results"]) > 1

    def test_run_returns_summary_structure(self, tmp_path):
        planner = _make_planner(tmp_path)
        agent = _make_agent()
        summary = planner.run("go to google and search python", agent)
        for key in ("success", "intent", "steps", "results", "failed_count"):
            assert key in summary


# ---------------------------------------------------------------------------
# _call_openai
# ---------------------------------------------------------------------------

class TestCallOpenAI:
    def test_returns_steps_from_list_response(self):
        good_steps = [{"action": "navigate", "url": "https://example.com"}]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            mock_openai = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = json.dumps(good_steps)
            mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
            with patch.dict("sys.modules", {"openai": mock_openai}):
                result = _call_openai("navigate to example.com")
        assert len(result) == 1
        assert result[0]["action"] == "navigate"

    def test_returns_steps_from_dict_steps_wrapper(self):
        good_steps = [{"action": "navigate", "url": "https://example.com"}]
        wrapped = {"steps": good_steps}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            mock_openai = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = json.dumps(wrapped)
            mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
            with patch.dict("sys.modules", {"openai": mock_openai}):
                result = _call_openai("navigate to example.com")
        assert result[0]["action"] == "navigate"

    def test_returns_steps_from_plan_key(self):
        good_steps = [{"action": "close_popups"}]
        wrapped = {"plan": good_steps}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            mock_openai = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = json.dumps(wrapped)
            mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
            with patch.dict("sys.modules", {"openai": mock_openai}):
                result = _call_openai("close popups")
        assert result[0]["action"] == "close_popups"

    def test_raises_on_unexpected_json(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            mock_openai = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = json.dumps({"unknown_key": "value"})
            mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
            with patch.dict("sys.modules", {"openai": mock_openai}):
                with pytest.raises(StepValidationError):
                    _call_openai("some intent")


# ---------------------------------------------------------------------------
# _call_ollama
# ---------------------------------------------------------------------------

class TestCallOllama:
    def _mock_urlopen(self, content: str):
        resp_mock = MagicMock()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        response_body = {"message": {"content": content}}
        resp_mock.read.return_value = json.dumps(response_body).encode()
        return resp_mock

    def test_returns_steps_from_json_array(self):
        good_steps = [{"action": "navigate", "url": "https://example.com"}]
        resp = self._mock_urlopen(json.dumps(good_steps))
        with patch("urllib.request.urlopen", return_value=resp), \
             patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            result = _call_ollama("navigate to example.com")
        assert result[0]["action"] == "navigate"

    def test_strips_markdown_fences(self):
        good_steps = [{"action": "navigate", "url": "https://example.com"}]
        content = f"```json\n{json.dumps(good_steps)}\n```"
        resp = self._mock_urlopen(content)
        with patch("urllib.request.urlopen", return_value=resp), \
             patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            result = _call_ollama("navigate to example.com")
        assert result[0]["action"] == "navigate"

    def test_raises_when_no_json_array(self):
        resp = self._mock_urlopen("Here are your steps: none available")
        with patch("urllib.request.urlopen", return_value=resp), \
             patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with pytest.raises(StepValidationError, match="no JSON array"):
                _call_ollama("some intent")


# ---------------------------------------------------------------------------
# _ollama_running helper
# ---------------------------------------------------------------------------

class TestOllamaRunning:
    def test_returns_true_when_reachable(self):
        resp_mock = MagicMock()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp_mock):
            assert _ollama_running() is True

    def test_returns_false_when_unreachable(self):
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert _ollama_running() is False


# ---------------------------------------------------------------------------
# TaskPlanner.plan — skill registry integration
# ---------------------------------------------------------------------------

class TestPlanSkillRegistry:
    def test_skill_match_takes_priority_over_template(self):
        from skills import SkillDef, SkillRegistry
        registry = SkillRegistry()
        registry.register(SkillDef(
            name="my_skill",
            triggers=["do my custom task"],
            steps=[{"action": "close_popups"}],
        ))
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            planner = TaskPlanner(skill_registry=registry)
        steps = planner.plan("do my custom task")
        assert steps[0]["action"] == "close_popups"

    def test_skill_builder_failure_falls_through_to_template(self):
        from skills import SkillDef, SkillRegistry

        registry = SkillRegistry()
        skill = SkillDef(
            name="bad_skill",
            triggers=["go to google and search python"],
            steps=[{"action": "navigate", "url": "https://example.com"}],
        )
        registry.register(skill)

        # Make resolve_steps throw (imported inside task_planner as 'from skills import resolve_steps')
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            planner = TaskPlanner(skill_registry=registry)
        with patch("skills.resolve_steps", side_effect=Exception("bad")):
            steps = planner.plan("go to google and search python")
        # Should fall through to template
        assert any(s["action"] == "navigate" for s in steps)

    def test_plan_no_llm_raises_for_unknown(self):
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            planner = TaskPlanner()
        with pytest.raises(ValueError, match="No template matched"):
            planner.plan("something completely unknown xyz 999")

    def test_plan_detect_llm_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        llm = TaskPlanner._detect_llm()
        assert llm == "openai"

    def test_plan_detect_llm_ollama_host(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        llm = TaskPlanner._detect_llm()
        assert llm == "ollama"

    def test_plan_detect_llm_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        with patch("task_planner._ollama_running", return_value=False):
            llm = TaskPlanner._detect_llm()
        assert llm is None
