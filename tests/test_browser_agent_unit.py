"""
tests/test_browser_agent_unit.py — Unit tests for browser_agent.BrowserAgent
using mocked Playwright page and context.

No real browser is launched; all Playwright internals are replaced with fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from browser_agent import _SELECTOR_FALLBACKS, BrowserAgent

# ---------------------------------------------------------------------------
# Fake helpers (mimic Playwright internals)
# ---------------------------------------------------------------------------

class _FakeKeyboard:
    def press(self, key, **_):
        pass


class _FakeMouse:
    def wheel(self, x, y):
        pass


class _FakeLocator:
    def __init__(self, count: int = 1):
        self._count = count

    def count(self):
        return self._count


class _FakeElement:
    def __init__(self, *, visible: bool = True, raise_on_click: bool = False):
        self._visible = visible
        self._raise_on_click = raise_on_click

    def is_visible(self):
        return self._visible

    def click(self, **_):
        if self._raise_on_click:
            raise Exception("click failed")

    def scroll_into_view_if_needed(self):
        pass

    def inner_text(self):
        return "element text"

    def get_attribute(self, name):
        return f"attr_{name}"


class _FakePage:
    def __init__(
        self,
        url: str = "https://example.com",
        title: str = "Example",
        inner_text: str = "page text",
        inner_html: str = "<p>html</p>",
        screenshot_data: bytes = b"\x89PNG",
        query_selector_result=None,
        evaluate_result=None,
    ):
        self._url = url
        self._title = title
        self._inner_text = inner_text
        self._inner_html = inner_html
        self._screenshot_data = screenshot_data
        self._query_selector_result = query_selector_result
        self._evaluate_result = evaluate_result
        self.keyboard = _FakeKeyboard()
        self.mouse = _FakeMouse()
        self.called_methods: list[str] = []

    @property
    def url(self):
        return self._url

    def title(self):
        return self._title

    def goto(self, url, wait_until="domcontentloaded"):
        self.called_methods.append("goto")
        resp = MagicMock()
        resp.status = 200
        return resp

    def click(self, selector, **_):
        self.called_methods.append(f"click:{selector}")

    def fill(self, selector, value, **_):
        self.called_methods.append(f"fill:{selector}")

    def type(self, selector, text, **_):
        self.called_methods.append(f"type:{selector}")

    def hover(self, selector, **_):
        self.called_methods.append(f"hover:{selector}")

    def select_option(self, selector, value, **_):
        self.called_methods.append(f"select:{selector}")

    def press(self, selector, key, **_):
        self.called_methods.append(f"press:{selector}:{key}")

    def inner_text(self, selector="body"):
        return self._inner_text

    def inner_html(self, selector="body"):
        return self._inner_html

    def get_attribute(self, selector, attribute):
        return f"val_{attribute}"

    def screenshot(self, **_):
        return self._screenshot_data

    def evaluate(self, script, *args):
        self.called_methods.append("evaluate")
        return self._evaluate_result

    def wait_for_selector(self, selector, **_):
        self.called_methods.append(f"wait_for_selector:{selector}")

    def wait_for_load_state(self, state, **_):
        self.called_methods.append(f"wait_for_load_state:{state}")

    def wait_for_function(self, script, **_):
        self.called_methods.append("wait_for_function")

    def query_selector(self, selector):
        return self._query_selector_result

    def query_selector_all(self, selector):
        el = MagicMock()
        el.inner_text.return_value = "text"
        el.get_attribute.return_value = "/link"
        return [el]

    def locator(self, selector):
        return _FakeLocator(count=1)

    def viewport_size(self):
        return {"width": 1280, "height": 800}

    def on(self, event, handler):
        pass

    def close(self):
        pass


class _FakeContext:
    def __init__(self, cookies=None):
        self._cookies = cookies or []

    def cookies(self):
        return self._cookies

    def add_cookies(self, cookies):
        self._cookies.extend(cookies)

    def set_default_timeout(self, timeout):
        pass

    def new_page(self):
        return _FakePage()


def _make_agent_with_page(page=None, ctx=None) -> BrowserAgent:
    """Return a BrowserAgent with fake internals (no Playwright launch)."""
    page = page or _FakePage()
    ctx  = ctx or _FakeContext()
    agent = BrowserAgent.__new__(BrowserAgent)
    agent.headless          = True
    agent.slow_mo           = 0
    agent.auto_close_popups = False  # keep tests simple by default
    agent.default_timeout   = 30_000
    agent._playwright       = None
    agent._browser          = None
    agent._context          = ctx
    agent._page             = page
    agent._pages            = [page]
    return agent


# ---------------------------------------------------------------------------
# Property guards
# ---------------------------------------------------------------------------

class TestBrowserAgentGuards:
    def test_page_property_raises_when_not_started(self) -> None:
        agent = BrowserAgent.__new__(BrowserAgent)
        agent._page = None
        with pytest.raises(RuntimeError, match="not started"):
            _ = agent.page


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class TestBrowserAgentNavigation:
    def test_navigate_returns_url_and_title(self) -> None:
        page = _FakePage(url="https://example.com", title="Example")
        agent = _make_agent_with_page(page)
        result = agent.navigate("https://example.com")
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example"

    def test_navigate_calls_goto(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        agent.navigate("https://example.com", wait_until="domcontentloaded")
        assert "goto" in page.called_methods

    def test_navigate_networkidle_path(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        # wait_until="networkidle" triggers _try_networkidle
        result = agent.navigate("https://example.com", wait_until="networkidle")
        assert result["url"] == "https://example.com"

    def test_get_title(self) -> None:
        page = _FakePage(title="My Title")
        agent = _make_agent_with_page(page)
        assert agent.get_title() == "My Title"

    def test_get_url(self) -> None:
        page = _FakePage(url="https://test.com")
        agent = _make_agent_with_page(page)
        assert agent.get_url() == "https://test.com"

    def test_get_text(self) -> None:
        page = _FakePage(inner_text="hello world")
        agent = _make_agent_with_page(page)
        assert agent.get_text() == "hello world"

    def test_get_html(self) -> None:
        page = _FakePage(inner_html="<h1>header</h1>")
        agent = _make_agent_with_page(page)
        assert "<h1>" in agent.get_html()

    def test_get_attribute(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        val = agent.get_attribute("a", "href")
        assert val == "val_href"

    def test_get_page_info(self) -> None:
        page = _FakePage(url="https://example.com", title="Example")
        agent = _make_agent_with_page(page)
        info = agent.get_page_info()
        assert "url" in info
        assert "title" in info


# ---------------------------------------------------------------------------
# Element interactions
# ---------------------------------------------------------------------------

class TestBrowserAgentInteractions:
    def test_press_key(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        result = agent.press_key("Enter")
        assert result["key"] == "Enter"

    def test_scroll(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        result = agent.scroll(0, 300)
        assert result["scrolled"]["y"] == 300

    def test_evaluate(self) -> None:
        page = _FakePage(evaluate_result=42)
        agent = _make_agent_with_page(page)
        result = agent.evaluate("1 + 1")
        assert result == 42

    def test_screenshot_base64(self) -> None:
        page = _FakePage(screenshot_data=b"PNG")
        agent = _make_agent_with_page(page)
        result = agent.screenshot(as_base64=True)
        assert "base64" in result
        import base64
        assert base64.b64decode(result["base64"]) == b"PNG"

    def test_screenshot_with_path(self, tmp_path: Path) -> None:
        page = _FakePage(screenshot_data=b"PNG")
        agent = _make_agent_with_page(page)
        out = str(tmp_path / "shot.png")
        result = agent.screenshot(path=out)
        assert result["path"] == out
        # When path is given, base64 is still added (as_base64 defaults False
        # but path is given so no base64 key unless as_base64=True)

    def test_wait_for_selector(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        result = agent.wait_for_selector("button")
        assert result["visible"] == "button"

    def test_wait_for_load_state(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        result = agent.wait_for_load_state("load")
        assert result["state"] == "load"

    def test_wait_for_load_state_networkidle(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        # networkidle path uses _try_networkidle
        result = agent.wait_for_load_state("networkidle")
        assert result["state"] == "networkidle"


# ---------------------------------------------------------------------------
# Selector resolution
# ---------------------------------------------------------------------------

class TestResolveSelector:
    def test_returns_primary_selector_when_found(self) -> None:
        page = _FakePage(query_selector_result=_FakeElement())
        agent = _make_agent_with_page(page)
        resolved = agent.resolve_selector("button.primary")
        assert resolved == "button.primary"

    def test_falls_back_to_secondary(self) -> None:
        """When primary returns None but fallback returns an element."""
        primary, fallbacks = _SELECTOR_FALLBACKS[0]  # Google search box
        call_count = [0]
        def fake_qs(selector):
            call_count[0] += 1
            if selector == primary:
                return None
            return _FakeElement()  # fallback found
        page = _FakePage()
        page.query_selector = fake_qs
        agent = _make_agent_with_page(page)
        resolved = agent.resolve_selector(primary)
        assert resolved != primary  # resolved to a fallback

    def test_raises_when_nothing_found(self) -> None:
        page = _FakePage(query_selector_result=None)
        agent = _make_agent_with_page(page)
        with pytest.raises(ValueError, match="No element found"):
            agent.resolve_selector("#nonexistent-unique-selector")

    def test_semantic_selector_found(self) -> None:
        page = _FakePage()
        page.locator = lambda sel: _FakeLocator(count=1)
        agent = _make_agent_with_page(page)
        resolved = agent.resolve_selector("text=Submit")
        assert resolved == "text=Submit"

    def test_semantic_selector_not_found_raises(self) -> None:
        page = _FakePage()
        page.locator = lambda sel: _FakeLocator(count=0)
        agent = _make_agent_with_page(page)
        with pytest.raises(ValueError, match="No element found for semantic selector"):
            agent.resolve_selector("text=NotHere")


# ---------------------------------------------------------------------------
# close_popups
# ---------------------------------------------------------------------------

class TestClosePopups:
    def test_no_popups(self) -> None:
        page = _FakePage(evaluate_result=False)
        page.query_selector = lambda sel: None  # nothing found
        agent = _make_agent_with_page(page)
        result = agent.close_popups()
        assert result["count"] == 0
        assert result["dismissed"] == []

    def test_dismisses_visible_popup(self) -> None:
        el = _FakeElement(visible=True)
        page = _FakePage(evaluate_result=False)
        call_count = [0]
        def qs(sel):
            call_count[0] += 1
            # Return the fake element for the first popup selector
            if call_count[0] == 1:
                return el
            return None
        page.query_selector = qs
        agent = _make_agent_with_page(page)
        result = agent.close_popups()
        assert result["count"] >= 1

    def test_invisible_popup_not_dismissed(self) -> None:
        el = _FakeElement(visible=False)
        page = _FakePage(evaluate_result=False)
        page.query_selector = lambda sel: el  # always returns invisible element
        agent = _make_agent_with_page(page)
        result = agent.close_popups()
        assert result["count"] == 0

    def test_dialog_dismissed_with_escape(self) -> None:
        page = _FakePage(evaluate_result=True)  # has_dialog=True
        page.query_selector = lambda sel: None
        agent = _make_agent_with_page(page)
        result = agent.close_popups()
        assert result["count"] >= 1
        assert any("Escape" in d for d in result["dismissed"])


# ---------------------------------------------------------------------------
# Smart extraction
# ---------------------------------------------------------------------------

class TestExtractLinks:
    def test_returns_links(self) -> None:
        page = _FakePage(evaluate_result=[{"text": "Example", "href": "https://example.com"}])
        agent = _make_agent_with_page(page)
        result = agent.extract_links()
        assert result["count"] == 1
        assert result["links"][0]["href"] == "https://example.com"

    def test_with_selector_and_limit(self) -> None:
        page = _FakePage(evaluate_result=[])
        agent = _make_agent_with_page(page)
        result = agent.extract_links(selector="nav a", limit=10)
        assert "links" in result
        assert "selector" in result
        assert result["selector"] == "nav a"


class TestExtractTable:
    def test_returns_table_data(self) -> None:
        table_data = {
            "headers": ["Name", "Value"],
            "rows": [{"Name": "foo", "Value": "bar"}],
            "count": 1,
        }
        page = _FakePage(evaluate_result=table_data)
        agent = _make_agent_with_page(page)
        result = agent.extract_table()
        assert result["count"] == 1
        assert result["headers"] == ["Name", "Value"]

    def test_no_table_raises(self) -> None:
        page = _FakePage(evaluate_result=None)
        agent = _make_agent_with_page(page)
        with pytest.raises(ValueError, match="No table found"):
            agent.extract_table()


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

class TestAssertions:
    def test_assert_text_found(self) -> None:
        page = _FakePage(inner_text="Hello World")
        agent = _make_agent_with_page(page)
        result = agent.assert_text("hello")  # case-insensitive by default
        assert result["found"] is True

    def test_assert_text_not_found_raises(self) -> None:
        page = _FakePage(inner_text="Hello World")
        agent = _make_agent_with_page(page)
        with pytest.raises(AssertionError, match="not found"):
            agent.assert_text("missing content")

    def test_assert_text_case_sensitive(self) -> None:
        page = _FakePage(inner_text="Hello World")
        agent = _make_agent_with_page(page)
        # "hello" should NOT match "Hello" when case_sensitive=True
        with pytest.raises(AssertionError):
            agent.assert_text("hello", case_sensitive=True)
        # But "Hello" should match
        result = agent.assert_text("Hello", case_sensitive=True)
        assert result["found"] is True

    def test_assert_url_found(self) -> None:
        page = _FakePage(url="https://example.com/dashboard")
        agent = _make_agent_with_page(page)
        result = agent.assert_url("/dashboard")
        assert result["matched"] is True

    def test_assert_url_not_found_raises(self) -> None:
        page = _FakePage(url="https://example.com/home")
        agent = _make_agent_with_page(page)
        with pytest.raises(AssertionError, match="does not contain"):
            agent.assert_url("/dashboard")


# ---------------------------------------------------------------------------
# Cookie management
# ---------------------------------------------------------------------------

class TestCookieManagement:
    def test_get_cookies(self) -> None:
        ctx = _FakeContext(cookies=[{"name": "session", "value": "abc"}])
        agent = _make_agent_with_page(ctx=ctx)
        cookies = agent.get_cookies()
        assert len(cookies) == 1
        assert cookies[0]["name"] == "session"

    def test_add_cookies(self) -> None:
        ctx = _FakeContext()
        agent = _make_agent_with_page(ctx=ctx)
        agent.add_cookies([{"name": "tok", "value": "xyz", "domain": "example.com"}])
        assert len(ctx.cookies()) == 1


# ---------------------------------------------------------------------------
# Multi-tab management
# ---------------------------------------------------------------------------

class TestMultiTabManagement:
    def test_new_tab_adds_page(self) -> None:
        ctx = _FakeContext()
        agent = _make_agent_with_page(ctx=ctx)
        result = agent.new_tab()
        assert "tab_index" in result
        assert result["tab_index"] == 1
        assert len(agent._pages) == 2

    def test_new_tab_with_url(self) -> None:
        ctx = _FakeContext()
        agent = _make_agent_with_page(ctx=ctx)
        result = agent.new_tab(url="https://example.com")
        assert result["tab_index"] == 1

    def test_new_tab_without_context_raises(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        agent._context = None
        with pytest.raises(RuntimeError, match="not started"):
            agent.new_tab()

    def test_switch_tab(self) -> None:
        page1 = _FakePage(url="https://page1.com", title="Page 1")
        page2 = _FakePage(url="https://page2.com", title="Page 2")
        agent = _make_agent_with_page(page1)
        agent._pages = [page1, page2]
        result = agent.switch_tab(1)
        assert result["tab_index"] == 1
        assert agent._page is page2

    def test_switch_tab_out_of_range_raises(self) -> None:
        agent = _make_agent_with_page()
        with pytest.raises(ValueError, match="out of range"):
            agent.switch_tab(99)

    def test_close_tab(self) -> None:
        page1 = _FakePage()
        page2 = _FakePage()
        page2.close = MagicMock()
        agent = _make_agent_with_page(page1)
        agent._pages = [page1, page2]
        agent._page = page2
        result = agent.close_tab()
        assert result["remaining_tabs"] == 1
        page2.close.assert_called_once()
        assert agent._page is page1

    def test_close_last_tab_raises(self) -> None:
        agent = _make_agent_with_page()
        with pytest.raises(RuntimeError, match="Cannot close the last"):
            agent.close_tab()

    def test_close_tab_out_of_range_raises(self) -> None:
        page1 = _FakePage()
        page2 = _FakePage()
        agent = _make_agent_with_page(page1)
        agent._pages = [page1, page2]
        with pytest.raises(ValueError, match="out of range"):
            agent.close_tab(index=99)

    def test_list_tabs(self) -> None:
        page1 = _FakePage(url="https://p1.com", title="P1")
        page2 = _FakePage(url="https://p2.com", title="P2")
        agent = _make_agent_with_page(page1)
        agent._pages = [page1, page2]
        agent._page = page1
        result = agent.list_tabs()
        assert result["count"] == 2
        assert result["tabs"][0]["active"] is True
        assert result["tabs"][1]["active"] is False

    def test_query_all(self) -> None:
        page = _FakePage()
        agent = _make_agent_with_page(page)
        results = agent.query_all("a")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert "text" in results[0]
        assert "href" in results[0]


# ---------------------------------------------------------------------------
# BrowserAgent lifecycle (stop)
# ---------------------------------------------------------------------------

class TestBrowserAgentLifecycle:
    def test_stop_clears_state(self) -> None:
        agent = _make_agent_with_page()
        mock_browser = MagicMock()
        mock_playwright = MagicMock()
        agent._browser = mock_browser
        agent._playwright = mock_playwright
        agent.stop()
        assert agent._page is None
        assert agent._context is None
        assert agent._browser is None
        assert agent._playwright is None
        assert agent._pages == []
        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()

    def test_context_manager(self) -> None:
        with patch.object(BrowserAgent, "start", return_value=None) as mock_start, \
             patch.object(BrowserAgent, "stop") as mock_stop:
            agent = BrowserAgent()
            agent._page = None
            with agent:
                pass
        mock_start.assert_called_once()
        mock_stop.assert_called_once()

    def test_scroll_to_element(self) -> None:
        el = _FakeElement()
        page = _FakePage(query_selector_result=el)
        agent = _make_agent_with_page(page)
        result = agent.scroll_to_element("h1")
        assert result["scrolled_to"] == "h1"

    def test_scroll_to_element_disappeared_raises(self) -> None:
        page = _FakePage(query_selector_result=None)
        agent = _make_agent_with_page(page)
        # resolve_selector would succeed first (returns the selector since
        # there's no fallback for "h1"), so we bypass resolve_selector
        with patch.object(agent, "resolve_selector", return_value="h1"):
            with pytest.raises(ValueError, match="disappeared"):
                agent.scroll_to_element("h1")
