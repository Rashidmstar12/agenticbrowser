"""Tests for agenticbrowser/cli.py (CLI commands via Click test runner)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from agenticbrowser.cli import main
from agenticbrowser.models import (
    BrowserOptions,
    Element,
    ElementsResult,
    NavigateResult,
    ScreenshotResult,
    TextResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_browser(
    nav_success=True,
    nav_error=None,
    text_success=True,
    text_content="Hello World",
    screenshot_success=True,
    screenshot_error=None,
    links=None,
):
    """Return a MagicMock Browser pre-configured for common scenarios."""
    browser = MagicMock()
    browser.__enter__ = lambda s: s
    browser.__exit__ = MagicMock(return_value=False)

    nav_result = NavigateResult(
        success=nav_success,
        url="https://example.com",
        title="Example",
        status=200,
        error=nav_error,
    )
    browser.navigate.return_value = nav_result

    text_result = TextResult(
        success=text_success,
        text=text_content,
        url="https://example.com",
    )
    browser.get_text.return_value = text_result

    sr = ScreenshotResult(
        success=screenshot_success,
        path="/tmp/out.png",
        data=b"\x89PNG",
        error=screenshot_error,
    )
    browser.screenshot.return_value = sr

    if links is None:
        links = [Element(tag="a", text="Link 1", href="https://example.com/a")]
    links_result = ElementsResult(success=True, elements=links, count=len(links))
    browser.find_links.return_value = links_result

    return browser


def _make_agent(nav_success=True, action_success=True):
    """Return a MagicMock AgentBrowser."""
    agent = MagicMock()
    agent.__enter__ = lambda s: s
    agent.__exit__ = MagicMock(return_value=False)

    nav_result = NavigateResult(
        success=nav_success,
        url="https://example.com",
        title="Example",
        status=200,
    )
    agent.run_action.side_effect = lambda action: (
        nav_result if action.get("action") == "navigate"
        else action
    )
    return agent


# ---------------------------------------------------------------------------
# navigate command
# ---------------------------------------------------------------------------

class TestNavigateCommand:
    def test_navigate_prints_title_url_status(self, tmp_path):
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["navigate", "https://example.com"])
        assert result.exit_code == 0
        assert "Title" in result.output
        assert "Example" in result.output
        assert "URL" in result.output

    def test_navigate_failure_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        browser = _make_browser(nav_success=False, nav_error="timeout")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["navigate", "https://bad.example"])
        assert result.exit_code != 0

    def test_navigate_with_show_text(self):
        runner = CliRunner()
        browser = _make_browser(text_content="visible page text")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["navigate", "--text", "https://example.com"])
        assert result.exit_code == 0
        assert "visible page text" in result.output

    def test_navigate_with_screenshot(self, tmp_path):
        out = str(tmp_path / "snap.png")
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["navigate", "--screenshot", out, "https://example.com"])
        assert result.exit_code == 0
        assert "Screenshot saved" in result.output

    def test_navigate_screenshot_failure(self, tmp_path):
        out = str(tmp_path / "snap.png")
        runner = CliRunner()
        browser = _make_browser(screenshot_success=False, screenshot_error="disk full")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["navigate", "--screenshot", out, "https://example.com"])
        assert "Screenshot failed" in result.output or result.exit_code == 0

    def test_navigate_text_failure(self):
        runner = CliRunner()
        browser = _make_browser(text_success=False)
        # When get_text fails, navigate still succeeds (text is best-effort)
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["navigate", "--text", "https://example.com"])
        assert result.exit_code == 0

    def test_navigate_custom_browser_options(self):
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser", return_value=browser) as mock_cls:
            result = runner.invoke(
                main,
                ["navigate", "--browser", "firefox", "--no-headless", "https://example.com"],
            )
        assert result.exit_code == 0
        call_opts = mock_cls.call_args[0][0]
        assert isinstance(call_opts, BrowserOptions)
        assert call_opts.browser_type == "firefox"
        assert call_opts.headless is False


# ---------------------------------------------------------------------------
# screenshot command
# ---------------------------------------------------------------------------

class TestScreenshotCommand:
    def test_screenshot_success(self, tmp_path):
        out = str(tmp_path / "out.png")
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["screenshot", "https://example.com", out])
        assert result.exit_code == 0
        assert "Saved screenshot" in result.output

    def test_screenshot_nav_failure(self, tmp_path):
        out = str(tmp_path / "out.png")
        runner = CliRunner()
        browser = _make_browser(nav_success=False, nav_error="timeout")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["screenshot", "https://bad.example", out])
        assert result.exit_code != 0
        assert "Navigation error" in result.output

    def test_screenshot_screenshot_failure(self, tmp_path):
        out = str(tmp_path / "out.png")
        runner = CliRunner()
        browser = _make_browser(screenshot_success=False, screenshot_error="crash")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["screenshot", "https://example.com", out])
        assert result.exit_code != 0

    def test_screenshot_full_page_flag(self, tmp_path):
        out = str(tmp_path / "full.png")
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(
                main, ["screenshot", "--full-page", "https://example.com", out]
            )
        assert result.exit_code == 0
        # Verify screenshot was called with full_page=True
        browser.screenshot.assert_called_once()
        _, kwargs = browser.screenshot.call_args
        assert kwargs.get("full_page") is True

    def test_screenshot_default_output(self, tmp_path):
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            with runner.isolated_filesystem():
                result = runner.invoke(main, ["screenshot", "https://example.com"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# text command
# ---------------------------------------------------------------------------

class TestTextCommand:
    def test_text_prints_content(self):
        runner = CliRunner()
        browser = _make_browser(text_content="page content here")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["text", "https://example.com"])
        assert result.exit_code == 0
        assert "page content here" in result.output

    def test_text_nav_failure(self):
        runner = CliRunner()
        browser = _make_browser(nav_success=False, nav_error="timeout")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["text", "https://bad.example"])
        assert result.exit_code != 0

    def test_text_get_text_failure(self):
        runner = CliRunner()
        browser = _make_browser(text_success=False)
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["text", "https://example.com"])
        assert result.exit_code != 0

    def test_text_custom_selector(self):
        runner = CliRunner()
        browser = _make_browser(text_content="article text")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["text", "--selector", "article", "https://example.com"])
        assert result.exit_code == 0
        browser.get_text.assert_called_once_with("article")


# ---------------------------------------------------------------------------
# links command
# ---------------------------------------------------------------------------

class TestLinksCommand:
    def test_links_plain_output(self):
        runner = CliRunner()
        links = [Element(tag="a", text="Home", href="https://example.com/")]
        browser = _make_browser(links=links)
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["links", "https://example.com"])
        assert result.exit_code == 0
        assert result.output.count("https://example.com/") >= 1

    def test_links_json_output(self):
        runner = CliRunner()
        links = [Element(tag="a", text="Home", href="https://example.com/")]
        browser = _make_browser(links=links)
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["links", "--json", "https://example.com"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["href"] == "https://example.com/"
        assert data[0]["text"] == "Home"

    def test_links_nav_failure(self):
        runner = CliRunner()
        browser = _make_browser(nav_success=False, nav_error="timeout")
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["links", "https://bad.example"])
        assert result.exit_code != 0

    def test_links_find_links_failure(self):
        runner = CliRunner()
        browser = _make_browser()
        browser.find_links.return_value = ElementsResult(
            success=False, error="no links found"
        )
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["links", "https://example.com"])
        assert result.exit_code != 0

    def test_links_empty_list(self):
        runner = CliRunner()
        browser = _make_browser(links=[])
        with patch("agenticbrowser.cli.Browser", return_value=browser):
            result = runner.invoke(main, ["links", "--json", "https://example.com"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []


# ---------------------------------------------------------------------------
# run command (action file runner)
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_run_actions_success(self, tmp_path):
        actions = [
            {"action": "navigate", "url": "https://example.com"},
            {"action": "get_text"},
        ]
        actions_file = tmp_path / "actions.json"
        actions_file.write_text(json.dumps(actions))

        runner = CliRunner()
        agent = _make_agent()
        with patch("agenticbrowser.cli.AgentBrowser", return_value=agent):
            result = runner.invoke(main, ["run", str(actions_file)])
        assert result.exit_code == 0

    def test_run_actions_displays_step_count(self, tmp_path):
        actions = [{"action": "navigate", "url": "https://example.com"}]
        actions_file = tmp_path / "actions.json"
        actions_file.write_text(json.dumps(actions))

        runner = CliRunner()
        agent = _make_agent()
        with patch("agenticbrowser.cli.AgentBrowser", return_value=agent):
            result = runner.invoke(main, ["run", str(actions_file)])
        assert "[1/1]" in result.output

    def test_run_actions_value_error_exits(self, tmp_path):
        actions = [{"action": "bad_action"}]
        actions_file = tmp_path / "actions.json"
        actions_file.write_text(json.dumps(actions))

        runner = CliRunner()
        agent = _make_agent()
        agent.run_action.side_effect = ValueError("Unknown action: bad_action")
        with patch("agenticbrowser.cli.AgentBrowser", return_value=agent):
            result = runner.invoke(main, ["run", str(actions_file)])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_run_actions_multiple_steps(self, tmp_path):
        actions = [
            {"action": "navigate", "url": "https://example.com"},
            {"action": "click", "selector": "button"},
            {"action": "get_text"},
        ]
        actions_file = tmp_path / "actions.json"
        actions_file.write_text(json.dumps(actions))

        runner = CliRunner()
        agent = _make_agent()
        with patch("agenticbrowser.cli.AgentBrowser", return_value=agent):
            result = runner.invoke(main, ["run", str(actions_file)])
        assert result.exit_code == 0
        assert "[3/3]" in result.output

    def test_run_nonexistent_file(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["run", str(tmp_path / "missing.json")])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Version / help
# ---------------------------------------------------------------------------

class TestMainGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "agenticbrowser" in result.output.lower()

    def test_navigate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["navigate", "--help"])
        assert result.exit_code == 0

    def test_screenshot_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["screenshot", "--help"])
        assert result.exit_code == 0

    def test_text_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["text", "--help"])
        assert result.exit_code == 0

    def test_links_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["links", "--help"])
        assert result.exit_code == 0

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        # Should either succeed or emit an error — not crash
        assert result.exit_code in (0, 1)

    def test_make_options_firefox_no_headless(self):
        """_make_options correctly builds BrowserOptions."""
        from agenticbrowser.cli import _make_options
        opts = _make_options("firefox", False, 5000)
        assert opts.browser_type == "firefox"
        assert opts.headless is False
        assert opts.timeout == 5000

    def test_make_options_webkit(self):
        from agenticbrowser.cli import _make_options
        opts = _make_options("webkit", True, 10000)
        assert opts.browser_type == "webkit"
        assert opts.headless is True
