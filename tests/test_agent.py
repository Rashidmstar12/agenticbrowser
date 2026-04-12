"""Tests for AgentBrowser."""

from __future__ import annotations

import pytest

from agenticbrowser.agent import AgentBrowser
from agenticbrowser.models import ActionResult, NavigateResult, TextResult


class TestAgentBrowserActions:
    """Tests for AgentBrowser.run_action with mocked Browser internals."""

    def _make_agent(self, monkeypatch, **results):
        """Build an AgentBrowser with its Browser's page mocked."""
        agent = AgentBrowser()
        # Inject a fake browser via the underlying Browser attribute
        fake_browser = _FakeBrowserBackend(**results)
        agent._browser = fake_browser
        return agent

    def test_unknown_action_raises(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        with pytest.raises(ValueError, match="Unknown action"):
            agent.run_action({"action": "does_not_exist"})

    def test_navigate(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "navigate", "url": "https://example.com"})
        assert result.success is True
        assert result.url == "https://example.com"

    def test_get_text(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "get_text"})
        assert result.success is True
        assert result.text == "fake text"

    def test_get_url(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        url = agent.run_action({"action": "get_url"})
        assert url == "https://example.com"

    def test_get_title(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        title = agent.run_action({"action": "get_title"})
        assert title == "Fake Page"

    def test_click(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "click", "selector": "button"})
        assert result.success is True

    def test_type(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "type", "selector": "input", "text": "hello"})
        assert result.success is True

    def test_press(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "press", "key": "Enter"})
        assert result.success is True

    def test_scroll(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "scroll", "y": 500})
        assert result.success is True

    def test_scroll_bottom(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "scroll_bottom"})
        assert result.success is True

    def test_scroll_top(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "scroll_top"})
        assert result.success is True

    def test_find_links(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "find_links"})
        assert result.success is True

    def test_find_buttons(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "find_buttons"})
        assert result.success is True

    def test_evaluate(self, monkeypatch):
        agent = self._make_agent(monkeypatch)
        result = agent.run_action({"action": "evaluate", "script": "1+1"})
        assert result.success is True

    def test_context_manager(self):
        calls = []
        agent = AgentBrowser()
        agent._browser._options  # ensure _browser exists

        import agenticbrowser.browser as bmod
        original_start = bmod.Browser.start
        original_close = bmod.Browser.close

        bmod.Browser.start = lambda self: calls.append("start")
        bmod.Browser.close = lambda self: calls.append("close")
        try:
            with agent:
                pass
        finally:
            bmod.Browser.start = original_start
            bmod.Browser.close = original_close

        assert calls == ["start", "close"]


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------

class _FakeBrowserBackend:
    """Minimal fake that mirrors Browser's public API."""

    def start(self):
        pass

    def close(self):
        pass

    def navigate(self, url, wait_until="domcontentloaded"):
        return NavigateResult(success=True, url=url, title="Fake Page", status=200)

    def go_back(self):
        return NavigateResult(success=True, url="https://example.com", title="Back")

    def go_forward(self):
        return NavigateResult(success=True, url="https://example.com", title="Fwd")

    def reload(self):
        return NavigateResult(success=True, url="https://example.com", title="Reload")

    def click(self, selector, timeout=None):
        return ActionResult(success=True)

    def type_text(self, selector, text, clear=True, timeout=None):
        return ActionResult(success=True)

    def press_key(self, key, selector=None):
        return ActionResult(success=True)

    def hover(self, selector):
        return ActionResult(success=True)

    def select_option(self, selector, value):
        return ActionResult(success=True)

    def scroll(self, x=0, y=500):
        return ActionResult(success=True)

    def scroll_to_bottom(self):
        return ActionResult(success=True)

    def scroll_to_top(self):
        return ActionResult(success=True)

    def get_text(self, selector="body"):
        return TextResult(success=True, text="fake text", url="https://example.com")

    def get_html(self, selector="html"):
        return TextResult(success=True, text="<html/>")

    def get_url(self):
        return "https://example.com"

    def get_title(self):
        return "Fake Page"

    def find_elements(self, selector):
        from agenticbrowser.models import ElementsResult
        return ElementsResult(success=True, elements=[], count=0)

    def find_links(self):
        from agenticbrowser.models import ElementsResult
        return ElementsResult(success=True, elements=[], count=0)

    def find_buttons(self):
        from agenticbrowser.models import ElementsResult
        return ElementsResult(success=True, elements=[], count=0)

    def find_inputs(self):
        from agenticbrowser.models import ElementsResult
        return ElementsResult(success=True, elements=[], count=0)

    def screenshot(self, path=None, full_page=False, selector=None):
        from agenticbrowser.models import ScreenshotResult
        return ScreenshotResult(success=True, data=b"\x89PNG")

    def screenshot_base64(self, full_page=False):
        import base64
        return base64.b64encode(b"\x89PNG").decode()

    def wait_for_selector(self, selector, state="visible", timeout=None):
        return ActionResult(success=True)

    def wait_for_load(self, timeout=None):
        return ActionResult(success=True)

    def wait_for_network_idle(self, timeout=None):
        return ActionResult(success=True)

    def evaluate(self, script):
        return ActionResult(success=True)
