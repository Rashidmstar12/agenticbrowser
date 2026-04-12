"""
tests/test_cli.py — Unit tests for agenticbrowser/cli.py

Uses Click's CliRunner to invoke each command without launching a real browser.
All Browser / AgentBrowser calls are mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from click.testing import CliRunner

from agenticbrowser.cli import main
from agenticbrowser.models import (
    ActionResult,
    ElementsResult,
    NavigateResult,
    ScreenshotResult,
    TextResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_browser(
    *,
    nav_success: bool = True,
    nav_error: str | None = None,
    title: str = "Test Page",
    url: str = "https://example.com",
    status: int = 200,
    text: str = "page text",
    screenshot_success: bool = True,
    links: list | None = None,
) -> MagicMock:
    """Build a mock Browser instance wired to common return values."""
    browser = MagicMock()

    nav_result = NavigateResult(
        success=nav_success,
        url=url,
        title=title,
        status=status,
        error=nav_error,
    )
    browser.navigate.return_value = nav_result
    browser.get_text.return_value = TextResult(
        success=True, text=text, url=url, title=title
    )
    browser.screenshot.return_value = ScreenshotResult(
        success=screenshot_success,
        path="/tmp/shot.png" if screenshot_success else None,
        data=b"\x89PNG" if screenshot_success else None,
        error=None if screenshot_success else "screenshot failed",
    )
    browser.find_links.return_value = ElementsResult(
        success=True,
        elements=links or [],
        count=len(links) if links else 0,
    )
    # Context-manager protocol
    browser.__enter__ = lambda s: s
    browser.__exit__ = MagicMock(return_value=False)
    return browser


# ---------------------------------------------------------------------------
# navigate command
# ---------------------------------------------------------------------------

class TestNavigateCommand:
    def test_success(self) -> None:
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["navigate", "https://example.com"])
        assert result.exit_code == 0
        assert "Test Page" in result.output
        assert "200" in result.output

    def test_navigation_error(self) -> None:
        runner = CliRunner()
        browser = _make_browser(nav_success=False, nav_error="timeout")
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["navigate", "https://bad-url"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_with_show_text(self) -> None:
        runner = CliRunner()
        browser = _make_browser(text="hello world")
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["navigate", "https://example.com", "--text"])
        assert result.exit_code == 0
        assert "hello world" in result.output

    def test_with_screenshot(self, tmp_path: Path) -> None:
        shot_path = str(tmp_path / "shot.png")
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(
                main, ["navigate", "https://example.com", "--screenshot", shot_path]
            )
        assert result.exit_code == 0
        browser.screenshot.assert_called_once_with(shot_path, full_page=True)


# ---------------------------------------------------------------------------
# screenshot command
# ---------------------------------------------------------------------------

class TestScreenshotCommand:
    def test_success(self, tmp_path: Path) -> None:
        out_path = str(tmp_path / "out.png")
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["screenshot", "https://example.com", out_path])
        assert result.exit_code == 0
        assert out_path in result.output or "Saved" in result.output

    def test_navigation_error(self) -> None:
        runner = CliRunner()
        browser = _make_browser(nav_success=False, nav_error="DNS failure")
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["screenshot", "https://bad-url", "out.png"])
        assert result.exit_code == 1

    def test_screenshot_failure(self) -> None:
        runner = CliRunner()
        browser = _make_browser(screenshot_success=False)
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["screenshot", "https://example.com", "out.png"])
        assert result.exit_code == 1

    def test_full_page_flag(self, tmp_path: Path) -> None:
        out_path = str(tmp_path / "full.png")
        runner = CliRunner()
        browser = _make_browser()
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(
                main, ["screenshot", "https://example.com", out_path, "--full-page"]
            )
        assert result.exit_code == 0
        browser.screenshot.assert_called_once_with(out_path, full_page=True)


# ---------------------------------------------------------------------------
# text command
# ---------------------------------------------------------------------------

class TestTextCommand:
    def test_prints_text(self) -> None:
        runner = CliRunner()
        browser = _make_browser(text="article content here")
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["text", "https://example.com"])
        assert result.exit_code == 0
        assert "article content here" in result.output

    def test_navigation_failure(self) -> None:
        runner = CliRunner()
        browser = _make_browser(nav_success=False, nav_error="timeout")
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["text", "https://bad-url"])
        assert result.exit_code == 1

    def test_text_failure(self) -> None:
        runner = CliRunner()
        browser = _make_browser()
        browser.get_text.return_value = TextResult(
            success=False, error="no text found"
        )
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["text", "https://example.com"])
        assert result.exit_code == 1

    def test_custom_selector(self) -> None:
        runner = CliRunner()
        browser = _make_browser(text="header text")
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(
                main, ["text", "https://example.com", "--selector", "h1"]
            )
        assert result.exit_code == 0
        browser.get_text.assert_called_once_with("h1")


# ---------------------------------------------------------------------------
# links command
# ---------------------------------------------------------------------------

class TestLinksCommand:
    def _make_element(self, href: str, text: str) -> MagicMock:
        el = MagicMock()
        el.href = href
        el.text = text
        return el

    def test_prints_links(self) -> None:
        runner = CliRunner()
        links = [
            self._make_element("https://example.com/a", "Link A"),
            self._make_element("https://example.com/b", "Link B"),
        ]
        browser = _make_browser(links=links)
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["links", "https://example.com"])
        assert result.exit_code == 0
        assert "Link A" in result.output
        assert "Link B" in result.output

    def test_as_json(self) -> None:
        runner = CliRunner()
        links = [self._make_element("https://example.com/a", "Link A")]
        browser = _make_browser(links=links)
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["links", "https://example.com", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["href"] == "https://example.com/a"
        assert data[0]["text"] == "Link A"

    def test_navigation_failure(self) -> None:
        runner = CliRunner()
        browser = _make_browser(nav_success=False, nav_error="refused")
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["links", "https://bad-url"])
        assert result.exit_code == 1

    def test_find_links_failure(self) -> None:
        runner = CliRunner()
        browser = _make_browser()
        browser.find_links.return_value = ElementsResult(
            success=False, error="find failed"
        )
        with patch("agenticbrowser.cli.Browser") as MockBrowser:
            MockBrowser.return_value = browser
            result = runner.invoke(main, ["links", "https://example.com"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_runs_actions_from_file(self, tmp_path: Path) -> None:
        actions = [
            {"action": "navigate", "url": "https://example.com"},
            {"action": "get_text"},
        ]
        actions_file = tmp_path / "actions.json"
        actions_file.write_text(json.dumps(actions))

        runner = CliRunner()
        agent = MagicMock()
        agent.__enter__ = lambda s: s
        agent.__exit__ = MagicMock(return_value=False)
        agent.run_action.return_value = ActionResult(success=True)

        with patch("agenticbrowser.cli.AgentBrowser") as MockAgent:
            MockAgent.return_value = agent
            result = runner.invoke(main, ["run", str(actions_file)])
        assert result.exit_code == 0
        assert agent.run_action.call_count == 2

    def test_unknown_action_exits_1(self, tmp_path: Path) -> None:
        actions = [{"action": "fly_to_mars"}]
        actions_file = tmp_path / "actions.json"
        actions_file.write_text(json.dumps(actions))

        runner = CliRunner()
        agent = MagicMock()
        agent.__enter__ = lambda s: s
        agent.__exit__ = MagicMock(return_value=False)
        agent.run_action.side_effect = ValueError("Unknown action 'fly_to_mars'")

        with patch("agenticbrowser.cli.AgentBrowser") as MockAgent:
            MockAgent.return_value = agent
            result = runner.invoke(main, ["run", str(actions_file)])
        assert result.exit_code == 1

    def test_empty_actions_runs_cleanly(self, tmp_path: Path) -> None:
        actions_file = tmp_path / "empty.json"
        actions_file.write_text(json.dumps([]))

        runner = CliRunner()
        agent = MagicMock()
        agent.__enter__ = lambda s: s
        agent.__exit__ = MagicMock(return_value=False)

        with patch("agenticbrowser.cli.AgentBrowser") as MockAgent:
            MockAgent.return_value = agent
            result = runner.invoke(main, ["run", str(actions_file)])
        assert result.exit_code == 0
        agent.run_action.assert_not_called()
