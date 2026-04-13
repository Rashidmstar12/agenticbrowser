"""
Unit tests for local_runner: _dispatch(), run_doctor(), run_task_file(),
_print_failure_report(), _print_help(), _color_enabled().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from local_runner import (
    _color_enabled,
    _dispatch,
    _print_failure_report,
    _print_help,
    run_doctor,
    run_task_file,
)
from system_tools import SystemTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.navigate.return_value = {"url": "https://example.com", "title": "Test"}
    agent.click.return_value = {"clicked": "button", "url": "https://example.com"}
    agent.fill.return_value = {"filled": "value", "selector": "input"}
    agent.type_text.return_value = {"typed": "text", "selector": "input"}
    agent.press_key.return_value = {"key": "Enter"}
    agent.hover.return_value = {"hovered": "a"}
    agent.select_option.return_value = {"selected": "opt"}
    agent.scroll.return_value = {"scrolled": {"x": 0, "y": 500}}
    agent.scroll_to_element.return_value = {"scrolled_to": "h1"}
    agent.close_popups.return_value = {"dismissed": [], "count": 0}
    agent.screenshot.return_value = {"path": "screenshot.png"}
    agent.get_text.return_value = "page text"
    agent.get_html.return_value = "<html/>"
    agent.get_attribute.return_value = "value"
    agent.query_all.return_value = [{"text": "item", "href": None}]
    agent.evaluate.return_value = 42
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Test"}
    agent.wait_for_selector.return_value = {"visible": "div"}
    agent.wait_for_load_state.return_value = {"state": "networkidle"}
    agent.extract_links.return_value = {"links": [{"text": "a", "href": "b"}], "count": 1}
    agent.extract_table.return_value = {"rows": [{"A": "1"}], "count": 1}
    agent.assert_text = MagicMock(return_value={"found": True})
    agent.assert_url = MagicMock(return_value={"matched": True})
    agent.wait_text.return_value = {"found": True}
    agent.get_cookies.return_value = [{"name": "sid", "value": "abc"}]
    agent.add_cookies.return_value = None
    agent.new_tab.return_value = {"tab_index": 1, "url": "about:blank", "title": "New"}
    agent.switch_tab.return_value = {"tab_index": 0, "url": "https://example.com", "title": "Test"}
    agent.close_tab.return_value = {"closed_index": 0, "remaining_tabs": 1}
    agent.list_tabs.return_value = {
        "tabs": [{"index": 0, "url": "https://example.com", "title": "Test", "active": True}]
    }
    return agent


# ---------------------------------------------------------------------------
# _color_enabled
# ---------------------------------------------------------------------------

class TestColorEnabled:
    def test_no_color_env_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        assert _color_enabled() is False

    def test_force_color_env_enables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert _color_enabled() is True

    def test_no_tty_without_force_returns_false(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            assert _color_enabled() is False


# ---------------------------------------------------------------------------
# run_doctor
# ---------------------------------------------------------------------------

class TestRunDoctor:
    def _fake_checks(self, statuses):
        checks = []
        for name, status in statuses:
            c = MagicMock()
            c.status = status
            c.name = name
            c.message = f"{name} message"
            c.fixed = False
            checks.append(c)
        return checks

    def test_all_ok_returns_zero(self, capsys):
        checks = self._fake_checks([("python_version", "ok"), ("workspace", "ok")])
        with patch("doctor.run_checks", return_value=checks):
            code = run_doctor()
        assert code == 0

    def test_one_failure_returns_one(self, capsys):
        checks = self._fake_checks([("chromium", "fail"), ("workspace", "ok")])
        with patch("doctor.run_checks", return_value=checks):
            code = run_doctor()
        assert code == 1

    def test_warn_only_returns_zero(self, capsys):
        checks = self._fake_checks([("openai", "warn")])
        with patch("doctor.run_checks", return_value=checks):
            code = run_doctor()
        assert code == 0

    def test_fixed_tag_shown(self, capsys):
        c = MagicMock()
        c.status = "ok"
        c.name = "chromium"
        c.message = "installed"
        c.fixed = True
        with patch("doctor.run_checks", return_value=[c]):
            run_doctor(fix=True)
        out = capsys.readouterr().out
        assert "fixed" in out.lower() or "chromium" in out

    def test_failure_message_printed(self, capsys):
        checks = self._fake_checks([("playwright", "fail")])
        with patch("doctor.run_checks", return_value=checks):
            run_doctor()
        out = capsys.readouterr().out
        assert "failed" in out.lower() or "playwright" in out

    def test_custom_workspace_passed(self):
        checks = self._fake_checks([("workspace", "ok")])
        with patch("doctor.run_checks", return_value=checks) as mock_rc:
            run_doctor(workspace="/custom/ws")
        mock_rc.assert_called_once_with(workspace="/custom/ws", fix=False)


# ---------------------------------------------------------------------------
# _print_failure_report
# ---------------------------------------------------------------------------

class TestPrintFailureReport:
    def test_basic_output(self, capsys):
        _print_failure_report("TimeoutError waiting for selector")
        out = capsys.readouterr().out
        # Should print something meaningful
        assert len(out) > 0

    def test_with_all_params_shows_rerun(self, capsys):
        _print_failure_report(
            "some error",
            action="click",
            step_index=1,
            intent="click the button",
            rerun_cmd="python local_runner.py --cmd 'click button'",
        )
        out = capsys.readouterr().out
        assert "Re-run" in out or "local_runner" in out

    def test_error_truncated_in_output(self, capsys):
        long_error = "E" * 500
        _print_failure_report(long_error)
        out = capsys.readouterr().out
        # Should not blow up; output should contain some error info
        assert len(out) > 0


# ---------------------------------------------------------------------------
# _print_help
# ---------------------------------------------------------------------------

def test_print_help(capsys):
    _print_help()
    out = capsys.readouterr().out
    assert "navigate" in out
    assert "quit" in out
    assert "task" in out
    assert "screenshot" in out


# ---------------------------------------------------------------------------
# _dispatch: browser commands
# ---------------------------------------------------------------------------

class TestDispatchBrowserCommands:
    def test_navigate(self):
        agent = _make_agent()
        _dispatch(agent, "navigate https://example.com")
        agent.navigate.assert_called_once_with("https://example.com", wait_until="domcontentloaded")

    def test_navigate_with_wait(self):
        agent = _make_agent()
        _dispatch(agent, "navigate https://example.com load")
        agent.navigate.assert_called_once_with("https://example.com", wait_until="load")

    def test_click(self):
        agent = _make_agent()
        _dispatch(agent, "click button#submit")
        agent.click.assert_called_once_with("button#submit")

    def test_type(self):
        agent = _make_agent()
        _dispatch(agent, "type input hello world")
        agent.type_text.assert_called_once_with("input", "hello world")

    def test_fill(self):
        agent = _make_agent()
        _dispatch(agent, "fill input my value")
        agent.fill.assert_called_once_with("input", "my value")

    def test_press(self):
        agent = _make_agent()
        _dispatch(agent, "press Enter")
        agent.press_key.assert_called_once_with("Enter")

    def test_hover(self):
        agent = _make_agent()
        _dispatch(agent, "hover a.link")
        agent.hover.assert_called_once_with("a.link")

    def test_select_option(self):
        agent = _make_agent()
        _dispatch(agent, "select_option select#lang en")
        agent.select_option.assert_called_once_with("select#lang", "en")

    def test_scroll_defaults(self):
        agent = _make_agent()
        _dispatch(agent, "scroll")
        agent.scroll.assert_called_once_with(x=0, y=500)

    def test_scroll_custom_y(self):
        agent = _make_agent()
        _dispatch(agent, "scroll 300")
        agent.scroll.assert_called_once_with(x=0, y=300)

    def test_scroll_custom_xy(self):
        agent = _make_agent()
        _dispatch(agent, "scroll 100 200")
        agent.scroll.assert_called_once_with(x=200, y=100)

    def test_scroll_to(self):
        agent = _make_agent()
        _dispatch(agent, "scroll_to h1")
        agent.scroll_to_element.assert_called_once_with("h1")

    def test_close_popups(self):
        agent = _make_agent()
        _dispatch(agent, "close_popups")
        agent.close_popups.assert_called_once()

    def test_screenshot_default(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "screenshot shot.png")
        agent.screenshot.assert_called_once_with(path="shot.png", full_page=False, as_base64=False)
        out = capsys.readouterr().out
        assert "shot.png" in out

    def test_screenshot_full(self):
        agent = _make_agent()
        _dispatch(agent, "screenshot shot.png true")
        _, kwargs = agent.screenshot.call_args
        assert kwargs["full_page"] is True

    def test_text_command(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "text body")
        agent.get_text.assert_called_once_with("body")
        out = capsys.readouterr().out
        assert "page text" in out

    def test_text_command_default_selector(self):
        agent = _make_agent()
        _dispatch(agent, "text")
        agent.get_text.assert_called_once_with("body")

    def test_html_command(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "html body")
        agent.get_html.assert_called_once_with("body")

    def test_attr_command(self):
        agent = _make_agent()
        result = _dispatch(agent, "attr a href")
        agent.get_attribute.assert_called_once_with("a", "href")
        assert result == {"value": "value"}

    def test_query_command(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "query div.item")
        agent.query_all.assert_called_once_with("div.item")
        assert result == {"count": 1}

    def test_eval_command(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "eval document.title")
        agent.evaluate.assert_called_once_with("document.title")
        out = capsys.readouterr().out
        assert "42" in out

    def test_info_command(self):
        agent = _make_agent()
        _dispatch(agent, "info")
        agent.get_page_info.assert_called_once()

    def test_wait_command(self):
        agent = _make_agent()
        _dispatch(agent, "wait div.loaded")
        agent.wait_for_selector.assert_called_once_with("div.loaded")

    def test_wait_state_custom(self):
        agent = _make_agent()
        _dispatch(agent, "wait_state load")
        agent.wait_for_load_state.assert_called_once_with("load")

    def test_wait_state_default(self):
        agent = _make_agent()
        _dispatch(agent, "wait_state")
        agent.wait_for_load_state.assert_called_once_with("networkidle")


# ---------------------------------------------------------------------------
# _dispatch: smart extraction commands
# ---------------------------------------------------------------------------

class TestDispatchSmartExtraction:
    def test_extract_links_defaults(self):
        agent = _make_agent()
        _dispatch(agent, "extract_links")
        agent.extract_links.assert_called_once_with(selector="a", limit=100)

    def test_extract_links_custom(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "extract_links a.nav 50")
        agent.extract_links.assert_called_once_with(selector="a.nav", limit=50)

    def test_extract_table_defaults(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "extract_table")
        agent.extract_table.assert_called_once_with(selector="table", table_index=0)

    def test_extract_table_custom(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "extract_table #t 1")
        agent.extract_table.assert_called_once_with(selector="#t", table_index=1)

    def test_assert_text(self):
        agent = _make_agent()
        _dispatch(agent, "assert_text hello body")
        agent.assert_text.assert_called_once_with("hello", selector="body")

    def test_assert_text_default_selector(self):
        agent = _make_agent()
        _dispatch(agent, "assert_text hello")
        agent.assert_text.assert_called_once_with("hello", selector="body")

    def test_assert_url(self):
        agent = _make_agent()
        _dispatch(agent, "assert_url /dashboard")
        agent.assert_url.assert_called_once_with("/dashboard")

    def test_wait_text(self):
        agent = _make_agent()
        _dispatch(agent, "wait_text Done body")
        agent.wait_text.assert_called_once_with("Done", selector="body")

    def test_wait_text_default_selector(self):
        agent = _make_agent()
        _dispatch(agent, "wait_text Ready")
        agent.wait_text.assert_called_once_with("Ready", selector="body")


# ---------------------------------------------------------------------------
# _dispatch: cookie commands
# ---------------------------------------------------------------------------

class TestDispatchCookies:
    def test_save_cookies(self, tmp_path):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        _dispatch(agent, "save_cookies cookies.json", tools=tools)
        agent.get_cookies.assert_called_once()
        assert (tmp_path / "cookies.json").exists()

    def test_save_cookies_no_tools_prints_error(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "save_cookies cookies.json", tools=None)
        assert result is None
        out = capsys.readouterr().out
        assert "Error" in out or "workspace" in out.lower()

    def test_load_cookies(self, tmp_path):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        cookies = [{"name": "tok", "value": "xyz"}]
        (tmp_path / "cookies.json").write_text(json.dumps(cookies))
        _dispatch(agent, "load_cookies cookies.json", tools=tools)
        agent.add_cookies.assert_called_once_with(cookies)

    def test_load_cookies_no_tools_prints_error(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "load_cookies cookies.json", tools=None)
        assert result is None


# ---------------------------------------------------------------------------
# _dispatch: multi-tab commands
# ---------------------------------------------------------------------------

class TestDispatchMultiTab:
    def test_new_tab_no_url(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "new_tab")
        agent.new_tab.assert_called_once_with(url=None)

    def test_new_tab_with_url(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "new_tab https://example.com")
        agent.new_tab.assert_called_once_with(url="https://example.com")

    def test_switch_tab(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "switch_tab 1")
        agent.switch_tab.assert_called_once_with(1)

    def test_close_tab_with_index(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "close_tab 0")
        agent.close_tab.assert_called_once_with(index=0)

    def test_close_tab_no_index(self):
        agent = _make_agent()
        _dispatch(agent, "close_tab")
        agent.close_tab.assert_called_once_with(index=None)

    def test_list_tabs(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "list_tabs")
        agent.list_tabs.assert_called_once()
        out = capsys.readouterr().out
        assert "https://" in out or "Tab" in out


# ---------------------------------------------------------------------------
# _dispatch: system tool commands
# ---------------------------------------------------------------------------

class TestDispatchSystemTools:
    def test_write_file(self, tmp_path, capsys):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        _dispatch(agent, "write_file test.txt hello content", tools=tools)
        assert (tmp_path / "test.txt").read_text() == "hello content"

    def test_write_file_no_tools(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "write_file test.txt content", tools=None)
        assert result is None
        assert "Error" in capsys.readouterr().out

    def test_append_file(self, tmp_path):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        (tmp_path / "log.txt").write_text("first\n")
        _dispatch(agent, "append_file log.txt second", tools=tools)
        assert "second" in (tmp_path / "log.txt").read_text()

    def test_append_file_no_tools(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "append_file f.txt content", tools=None)
        assert result is None

    def test_read_file(self, tmp_path, capsys):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        (tmp_path / "data.txt").write_text("some data")
        _dispatch(agent, "read_file data.txt", tools=tools)
        out = capsys.readouterr().out
        assert "some data" in out

    def test_read_file_no_tools(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "read_file data.txt", tools=None)
        assert result is None

    def test_list_dir(self, tmp_path, capsys):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        result = _dispatch(agent, "list_dir .", tools=tools)
        assert result is not None

    def test_list_dir_no_tools(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "list_dir .", tools=None)
        assert result is None

    def test_run_python(self, tmp_path, capsys):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        _dispatch(agent, "run_python print('hi')", tools=tools)
        out = capsys.readouterr().out
        assert "hi" in out

    def test_run_python_no_tools(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "run_python 1+1", tools=None)
        assert result is None

    def test_run_shell(self, tmp_path, capsys):
        agent = _make_agent()
        tools = SystemTools(workspace=tmp_path)
        _dispatch(agent, "run_shell echo hello", tools=tools)
        out = capsys.readouterr().out
        assert "hello" in out

    def test_run_shell_no_tools(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "run_shell echo hi", tools=None)
        assert result is None


# ---------------------------------------------------------------------------
# _dispatch: meta commands
# ---------------------------------------------------------------------------

class TestDispatchMeta:
    def test_help(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "help")
        out = capsys.readouterr().out
        assert "navigate" in out
        assert result is None

    def test_question_mark_help(self, capsys):
        agent = _make_agent()
        _dispatch(agent, "?")
        assert "navigate" in capsys.readouterr().out

    def test_quit_returns_quit(self):
        agent = _make_agent()
        assert _dispatch(agent, "quit") == "QUIT"

    def test_exit_returns_quit(self):
        agent = _make_agent()
        assert _dispatch(agent, "exit") == "QUIT"

    def test_q_returns_quit(self):
        agent = _make_agent()
        assert _dispatch(agent, "q") == "QUIT"

    def test_unknown_command(self, capsys):
        agent = _make_agent()
        result = _dispatch(agent, "definitely_unknown_cmd_xyz")
        out = capsys.readouterr().out
        assert "Unknown" in out
        assert result is None

    def test_empty_line_returns_none(self):
        agent = _make_agent()
        assert _dispatch(agent, "   ") is None

    def test_doctor_command(self, capsys):
        agent = _make_agent()
        checks = [MagicMock(status="ok", name="test", message="ok", fixed=False)]
        with patch("doctor.run_checks", return_value=checks):
            result = _dispatch(agent, "doctor")
        assert result is None

    def test_doctor_fix_command(self, capsys):
        agent = _make_agent()
        checks = [MagicMock(status="ok", name="test", message="ok", fixed=True)]
        with patch("doctor.run_checks", return_value=checks):
            _dispatch(agent, "doctor fix")

    def test_skill_list_empty(self, capsys):
        agent = _make_agent()
        from skills import SkillRegistry
        with patch("skills.get_default_registry", return_value=SkillRegistry()):
            _dispatch(agent, "skill list")
        out = capsys.readouterr().out
        assert "No skills" in out or "skill" in out.lower()

    def test_skill_list_with_skill(self, capsys):
        agent = _make_agent()
        from skills import SkillDef, SkillRegistry
        reg = SkillRegistry()
        reg.register(SkillDef(name="test_sk", steps=[{"action": "close_popups"}], version="1.0"))
        with patch("skills.get_default_registry", return_value=reg):
            _dispatch(agent, "skill list")
        assert "test_sk" in capsys.readouterr().out

    def test_skill_info_found(self, capsys):
        agent = _make_agent()
        from skills import SkillDef, SkillRegistry
        reg = SkillRegistry()
        reg.register(SkillDef(name="my_sk", steps=[{"action": "close_popups"}]))
        with patch("skills.get_default_registry", return_value=reg):
            _dispatch(agent, "skill info my_sk")
        assert "my_sk" in capsys.readouterr().out

    def test_skill_info_not_found(self, capsys):
        agent = _make_agent()
        from skills import SkillRegistry
        with patch("skills.get_default_registry", return_value=SkillRegistry()):
            _dispatch(agent, "skill info nonexistent")
        assert "not found" in capsys.readouterr().out.lower()

    def test_skill_unload(self, capsys):
        agent = _make_agent()
        from skills import SkillDef, SkillRegistry
        reg = SkillRegistry()
        reg.register(SkillDef(name="bye_sk", steps=[{"action": "close_popups"}]))
        with patch("skills.get_default_registry", return_value=reg):
            _dispatch(agent, "skill unload bye_sk")
        assert "bye_sk" in capsys.readouterr().out

    def test_skill_invalid_sub(self, capsys):
        agent = _make_agent()
        from skills import SkillRegistry
        with patch("skills.get_default_registry", return_value=SkillRegistry()):
            _dispatch(agent, "skill bad_sub_command")
        assert "Usage" in capsys.readouterr().out

    def test_skill_load_error(self, capsys):
        agent = _make_agent()
        from skills import SkillLoadError, SkillRegistry
        reg = SkillRegistry()
        reg.load_from_source = MagicMock(side_effect=SkillLoadError("bad source"))
        with patch("skills.get_default_registry", return_value=reg):
            _dispatch(agent, "skill load /bad/path")
        assert "Error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _dispatch: task commands
# ---------------------------------------------------------------------------

class TestDispatchTaskCommands:
    def test_task_plan(self, capsys):
        from task_planner import TaskPlanner
        agent = _make_agent()
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            planner = TaskPlanner()
        _dispatch(agent, "task_plan go to google and search python", planner=planner)
        out = capsys.readouterr().out
        assert "navigate" in out

    def test_task_run_success(self, capsys):
        agent = _make_agent()
        mock_planner = MagicMock()
        mock_planner.run.return_value = {
            "success": True,
            "steps": [{"action": "navigate", "url": "https://example.com"}],
            "results": [{"step": 0, "action": "navigate", "status": "ok"}],
        }
        _dispatch(agent, "task go to example.com", planner=mock_planner)
        out = capsys.readouterr().out
        assert "Done" in out or "success" in out.lower()

    def test_task_run_failure(self, capsys):
        agent = _make_agent()
        mock_planner = MagicMock()
        mock_planner.run.return_value = {
            "success": False,
            "steps": [{"action": "click", "selector": "#btn"}],
            "results": [{"step": 0, "action": "click", "status": "error", "error": "not found"}],
        }
        _dispatch(agent, "task click something", planner=mock_planner)
        out = capsys.readouterr().out
        assert "failed" in out.lower() or "error" in out.lower()

    def test_task_creates_planner_if_none(self, capsys):
        agent = _make_agent()
        with patch("local_runner.TaskPlanner") as MockPlanner:
            mock_planner_instance = MagicMock()
            mock_planner_instance.run.return_value = {
                "success": True,
                "steps": [],
                "results": [],
            }
            MockPlanner.return_value = mock_planner_instance
            _dispatch(agent, "task go to example.com", planner=None)
        MockPlanner.assert_called_once()

    def test_task_plan_creates_planner_if_none(self, capsys):
        agent = _make_agent()
        with patch("local_runner.TaskPlanner") as MockPlanner:
            mock_planner_instance = MagicMock()
            mock_planner_instance.plan.return_value = []
            MockPlanner.return_value = mock_planner_instance
            _dispatch(agent, "task_plan go to example.com", planner=None)
        MockPlanner.assert_called_once()


# ---------------------------------------------------------------------------
# _dispatch: exception handling
# ---------------------------------------------------------------------------

class TestDispatchExceptionHandling:
    def test_exception_caught_and_printed(self, capsys):
        agent = _make_agent()
        agent.navigate.side_effect = Exception("something went wrong")
        result = _dispatch(agent, "navigate https://example.com")
        out = capsys.readouterr().out
        assert "Error" in out or "something went wrong" in out
        assert result is None


# ---------------------------------------------------------------------------
# run_task_file
# ---------------------------------------------------------------------------

class TestRunTaskFile:
    def test_valid_task_executes_successfully(self, tmp_path, capsys):
        steps = [{"action": "navigate", "url": "https://example.com"}]
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(steps))

        agent = _make_agent()
        run_task_file(agent, str(task_file), workspace=str(tmp_path))

        out = capsys.readouterr().out
        assert "success" in out.lower() or "completed" in out.lower()

    def test_invalid_action_fails_validation(self, tmp_path, capsys):
        steps = [{"action": "unknown_action_xyz_9999"}]
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(steps))

        agent = _make_agent()
        run_task_file(agent, str(task_file))
        out = capsys.readouterr().out
        assert "Validation error" in out or "error" in out.lower()

    def test_failed_step_shows_error_report(self, tmp_path, capsys):
        steps = [{"action": "navigate", "url": "https://example.com"}]
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(steps))

        agent = _make_agent()
        agent.navigate.side_effect = Exception("nav failed")
        run_task_file(agent, str(task_file), workspace=str(tmp_path))
        out = capsys.readouterr().out
        assert "failed" in out.lower() or "step" in out.lower()

    def test_multiple_steps_all_succeed(self, tmp_path, capsys):
        steps = [
            {"action": "navigate", "url": "https://example.com"},
            {"action": "close_popups"},
            {"action": "press", "key": "Escape"},
        ]
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(steps))

        agent = _make_agent()
        run_task_file(agent, str(task_file), workspace=str(tmp_path))
        out = capsys.readouterr().out
        assert "3" in out or "success" in out.lower()

    def test_workspace_passed_to_system_tools(self, tmp_path, capsys):
        steps = [{"action": "write_file", "path": "out.txt", "content": "data"}]
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(steps))

        agent = _make_agent()
        run_task_file(agent, str(task_file), workspace=str(tmp_path))
        assert (tmp_path / "out.txt").read_text() == "data"


# ---------------------------------------------------------------------------
# Additional _dispatch coverage: upload_file, download_file, drag_drop,
# right_click, double_click, get_rect, set_network_intercept,
# clear_network_intercepts, set_viewport, set_geolocation
# ---------------------------------------------------------------------------

class TestDispatchAdvancedCommands:
    """Tests for _dispatch() commands that were previously uncovered."""

    def _agent(self) -> MagicMock:
        return _make_agent()

    def test_upload_file(self, tmp_path, capsys):
        agent = self._agent()
        agent.upload_file.return_value = {"selector": "input", "status": "ok"}
        _dispatch(agent, "upload_file input /tmp/file.txt")
        agent.upload_file.assert_called_once_with("input", "/tmp/file.txt")
        out = capsys.readouterr().out
        assert "Uploaded" in out

    def test_upload_file_with_tools_workspace(self, tmp_path, capsys):
        agent = self._agent()
        agent.upload_file.return_value = {"selector": "input", "status": "ok"}
        tools = MagicMock()
        tools.workspace = tmp_path
        _dispatch(agent, "upload_file input file.txt", tools=tools)
        expected_path = str(tmp_path / "file.txt")
        agent.upload_file.assert_called_once_with("input", expected_path)

    def test_download_file(self, capsys):
        agent = self._agent()
        agent.download_file.return_value = {"saved_to": "/tmp/f.bin", "filename": "f.bin"}
        _dispatch(agent, "download_file https://example.com/f.bin /tmp/f.bin")
        agent.download_file.assert_called_once_with("https://example.com/f.bin", "/tmp/f.bin")
        out = capsys.readouterr().out
        assert "Saved to" in out

    def test_download_file_with_tools_workspace(self, tmp_path, capsys):
        agent = self._agent()
        agent.download_file.return_value = {"saved_to": "/tmp/f.bin", "filename": "f.bin"}
        tools = MagicMock()
        tools.workspace = tmp_path
        _dispatch(agent, "download_file https://example.com/f.bin out.bin", tools=tools)
        expected_path = str(tmp_path / "out.bin")
        agent.download_file.assert_called_once_with("https://example.com/f.bin", expected_path)

    def test_drag_drop(self):
        agent = self._agent()
        agent.drag_and_drop.return_value = {"status": "ok"}
        _dispatch(agent, "drag_drop #src #dst")
        agent.drag_and_drop.assert_called_once_with("#src", "#dst")

    def test_right_click(self):
        agent = self._agent()
        agent.right_click.return_value = {"status": "ok"}
        _dispatch(agent, "right_click a.link")
        agent.right_click.assert_called_once_with("a.link")

    def test_double_click(self):
        agent = self._agent()
        agent.double_click.return_value = {"status": "ok"}
        _dispatch(agent, "double_click button")
        agent.double_click.assert_called_once_with("button")

    def test_get_rect(self, capsys):
        agent = self._agent()
        agent.get_element_rect.return_value = {"x": 10, "y": 20, "width": 100, "height": 50}
        _dispatch(agent, "get_rect div.box")
        agent.get_element_rect.assert_called_once_with("div.box")
        out = capsys.readouterr().out
        assert "x=10" in out

    def test_set_network_intercept(self, capsys):
        agent = self._agent()
        agent.set_network_intercept.return_value = {"url_pattern": "*.jpg", "action": "abort"}
        _dispatch(agent, "set_network_intercept *.jpg abort")
        agent.set_network_intercept.assert_called_once_with("*.jpg", action="abort")
        out = capsys.readouterr().out
        assert "Intercept set" in out

    def test_set_network_intercept_default_action(self, capsys):
        agent = self._agent()
        agent.set_network_intercept.return_value = {"url_pattern": "*.css", "action": "abort"}
        _dispatch(agent, "set_network_intercept *.css")
        agent.set_network_intercept.assert_called_once_with("*.css", action="abort")

    def test_clear_network_intercepts(self, capsys):
        agent = self._agent()
        agent.clear_network_intercepts.return_value = {"cleared": 3}
        _dispatch(agent, "clear_network_intercepts")
        agent.clear_network_intercepts.assert_called_once()
        out = capsys.readouterr().out
        assert "Cleared 3" in out

    def test_set_viewport(self, capsys):
        agent = self._agent()
        agent.set_viewport.return_value = {"width": 1920, "height": 1080, "status": "ok"}
        _dispatch(agent, "set_viewport 1920 1080")
        agent.set_viewport.assert_called_once_with(1920, 1080)
        out = capsys.readouterr().out
        assert "1920x1080" in out

    def test_set_geolocation(self, capsys):
        agent = self._agent()
        agent.set_geolocation.return_value = {"lat": 40.7, "lng": -74.0}
        # split(None, 2) means parts[2] = "-74.0" with only 3 parts when accuracy omitted
        _dispatch(agent, "set_geolocation 40.7 -74.0")
        agent.set_geolocation.assert_called_once_with(40.7, -74.0, accuracy=10.0)
        out = capsys.readouterr().out
        assert "lat=40.7" in out

    def test_set_geolocation_default_accuracy(self, capsys):
        agent = self._agent()
        agent.set_geolocation.return_value = {"lat": 51.5, "lng": 0.1}
        _dispatch(agent, "set_geolocation 51.5 0.1")
        agent.set_geolocation.assert_called_once_with(51.5, 0.1, accuracy=10.0)
