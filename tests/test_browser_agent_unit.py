"""
Unit tests for browser_agent.BrowserAgent.

All Playwright calls are mocked — no real browser is launched.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from browser_agent import BrowserAgent

# ---------------------------------------------------------------------------
# Helper — build an agent with a fully mocked Playwright page/context
# ---------------------------------------------------------------------------

def _make_agent(**kwargs) -> tuple[BrowserAgent, MagicMock, MagicMock]:
    """Return (agent, mock_page, mock_context) with BrowserAgent fully wired."""
    agent = BrowserAgent(**kwargs)

    page = MagicMock()
    page.url = "https://example.com"
    page.title.return_value = "Test Page"
    page.viewport_size = {"width": 1280, "height": 800}
    page.inner_text.return_value = "inner text content"
    page.inner_html.return_value = "<p>html</p>"
    page.get_attribute.return_value = "attr_val"
    page.evaluate.return_value = None
    page.screenshot.return_value = b"\x89PNG"
    page.query_selector.return_value = MagicMock(is_visible=MagicMock(return_value=False))
    page.query_selector_all.return_value = []
    page.keyboard = MagicMock()
    page.mouse = MagicMock()

    locator_mock = MagicMock()
    locator_mock.count.return_value = 0
    page.locator.return_value = locator_mock

    context = MagicMock()
    context.cookies.return_value = []

    agent._page = page
    agent._context = context
    agent._pages = [page]
    return agent, page, context


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_stop_clears_all_state(self):
        agent, page, context = _make_agent()
        mock_browser = MagicMock()
        mock_playwright = MagicMock()
        agent._browser = mock_browser
        agent._playwright = mock_playwright

        agent.stop()

        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()
        assert agent._page is None
        assert agent._context is None
        assert agent._browser is None
        assert agent._playwright is None
        assert agent._pages == []

    def test_stop_when_nothing_started(self):
        agent = BrowserAgent()
        agent.stop()  # should not raise

    def test_page_property_raises_when_not_started(self):
        agent = BrowserAgent()
        with pytest.raises(RuntimeError, match="not started"):
            _ = agent.page

    def test_context_manager_calls_start_and_stop(self):
        agent = BrowserAgent()
        with patch.object(agent, "start", return_value=agent) as mock_start, \
             patch.object(agent, "stop") as mock_stop:
            with agent:
                pass
        mock_start.assert_called_once()
        mock_stop.assert_called_once()

    def test_context_manager_stops_even_on_exception(self):
        agent = BrowserAgent()
        with patch.object(agent, "start", return_value=agent), \
             patch.object(agent, "stop") as mock_stop:
            with pytest.raises(ValueError):
                with agent:
                    raise ValueError("boom")
        mock_stop.assert_called_once()

    def test_handle_dialog_dismisses(self):
        agent, _, _ = _make_agent()
        dialog = MagicMock()
        agent._handle_dialog(dialog)
        dialog.dismiss.assert_called_once()


# ---------------------------------------------------------------------------
# _try_networkidle
# ---------------------------------------------------------------------------

class TestTryNetworkidle:
    def test_returns_true_on_success(self):
        agent, page, _ = _make_agent()
        page.wait_for_load_state.return_value = None
        assert agent._try_networkidle() is True

    def test_returns_false_and_falls_back_on_timeout(self):
        agent, page, _ = _make_agent()

        def side_effect(state, timeout=None):
            if state == "networkidle":
                raise Exception("timeout")

        page.wait_for_load_state.side_effect = side_effect
        assert agent._try_networkidle() is False

    def test_returns_false_when_both_fail(self):
        agent, page, _ = _make_agent()
        page.wait_for_load_state.side_effect = Exception("all fail")
        assert agent._try_networkidle() is False


# ---------------------------------------------------------------------------
# navigate
# ---------------------------------------------------------------------------

class TestNavigate:
    def test_basic_navigate(self):
        agent, page, _ = _make_agent(auto_close_popups=False)
        result = agent.navigate("https://example.com")
        page.goto.assert_called_with("https://example.com", wait_until="domcontentloaded")
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"

    def test_navigate_with_load_wait(self):
        agent, page, _ = _make_agent(auto_close_popups=False)
        agent.navigate("https://example.com", wait_until="load")
        page.goto.assert_called_with("https://example.com", wait_until="load")

    def test_navigate_networkidle_uses_domcontentloaded(self):
        agent, page, _ = _make_agent(auto_close_popups=False)
        with patch.object(agent, "_try_networkidle", return_value=True) as mock_ni:
            agent.navigate("https://example.com", wait_until="networkidle")
        page.goto.assert_called_with("https://example.com", wait_until="domcontentloaded")
        mock_ni.assert_called_once()

    def test_navigate_calls_close_popups_when_enabled(self):
        agent, page, _ = _make_agent(auto_close_popups=True)
        page.query_selector.return_value = None
        page.evaluate.return_value = False
        result = agent.navigate("https://example.com")
        assert result["url"] == "https://example.com"

    def test_navigate_skips_close_popups_when_disabled(self):
        agent, page, _ = _make_agent(auto_close_popups=False)
        with patch.object(agent, "close_popups") as mock_cp:
            agent.navigate("https://example.com")
        mock_cp.assert_not_called()


# ---------------------------------------------------------------------------
# close_popups
# ---------------------------------------------------------------------------

class TestClosePopups:
    def test_no_popups_returns_empty(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = None
        page.evaluate.return_value = False
        result = agent.close_popups()
        assert result["count"] == 0
        assert result["dismissed"] == []

    def test_dismisses_visible_element(self):
        agent, page, _ = _make_agent()
        el = MagicMock()
        el.is_visible.return_value = True
        page.query_selector.side_effect = [el] + [None] * 100
        page.evaluate.return_value = False
        result = agent.close_popups()
        el.click.assert_called_with(timeout=2_000)
        assert result["count"] >= 1

    def test_invisible_element_not_clicked(self):
        agent, page, _ = _make_agent()
        el = MagicMock()
        el.is_visible.return_value = False
        page.query_selector.return_value = el
        page.evaluate.return_value = False
        agent.close_popups()
        el.click.assert_not_called()

    def test_dialog_escape_pressed(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = None
        page.evaluate.return_value = True
        result = agent.close_popups()
        page.keyboard.press.assert_called_with("Escape")
        assert any("Escape" in d for d in result["dismissed"])

    def test_exception_in_selector_is_swallowed(self):
        agent, page, _ = _make_agent()
        page.query_selector.side_effect = Exception("playwright error")
        page.evaluate.return_value = False
        result = agent.close_popups()
        assert result["count"] == 0

    def test_exception_in_dialog_check_swallowed(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = None
        page.evaluate.side_effect = Exception("evaluate failed")
        result = agent.close_popups()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# resolve_selector
# ---------------------------------------------------------------------------

class TestResolveSelector:
    def test_returns_matching_css(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = MagicMock()
        assert agent.resolve_selector("button") == "button"

    def test_raises_when_not_found(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = None
        with pytest.raises(ValueError, match="No element found"):
            agent.resolve_selector("nonexistent")

    def test_semantic_prefix_text(self):
        agent, page, _ = _make_agent()
        locator = MagicMock()
        locator.count.return_value = 1
        page.locator.return_value = locator
        assert agent.resolve_selector("text=Click me") == "text=Click me"

    def test_semantic_prefix_role(self):
        agent, page, _ = _make_agent()
        locator = MagicMock()
        locator.count.return_value = 1
        page.locator.return_value = locator
        assert agent.resolve_selector("role=button") == "role=button"

    def test_semantic_prefix_not_found_raises(self):
        agent, page, _ = _make_agent()
        locator = MagicMock()
        locator.count.return_value = 0
        page.locator.return_value = locator
        with pytest.raises(ValueError, match="semantic selector"):
            agent.resolve_selector("text=Nonexistent")

    def test_semantic_prefix_exception_raises_value_error(self):
        agent, page, _ = _make_agent()
        page.locator.side_effect = Exception("locator error")
        with pytest.raises(ValueError, match="semantic selector"):
            agent.resolve_selector("text=Something")

    def test_fallback_google_selector(self):
        agent, page, _ = _make_agent()

        def qsel(sel):
            if sel == "textarea[name='q']":
                return None
            return MagicMock()

        page.query_selector.side_effect = qsel
        resolved = agent.resolve_selector("textarea[name='q']")
        assert resolved != "textarea[name='q']"

    def test_query_selector_exception_continues_to_fallback(self):
        agent, page, _ = _make_agent()
        call_count = [0]

        def qsel(sel):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("dom error")  # primary raises
            return MagicMock()  # fallback succeeds

        page.query_selector.side_effect = qsel
        # Use a selector that has registered fallbacks
        result = agent.resolve_selector("textarea[name='q']")
        assert result == "input[name='q']"


# ---------------------------------------------------------------------------
# Element interaction
# ---------------------------------------------------------------------------

class TestInteraction:
    def test_click(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = MagicMock()
        result = agent.click("button")
        page.click.assert_called_once()
        assert result["clicked"] == "button"
        assert "url" in result

    def test_click_with_custom_timeout(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = MagicMock()
        agent.click("button", timeout=5000)
        args, kwargs = page.click.call_args
        assert kwargs.get("timeout") == 5000 or (len(args) > 1 and args[1] == 5000)

    def test_click_default_timeout(self):
        agent, page, _ = _make_agent(default_timeout=10000)
        page.query_selector.return_value = MagicMock()
        agent.click("button")
        _, kwargs = page.click.call_args
        assert kwargs.get("timeout") == 10000

    def test_type_text_clear_first(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = MagicMock()
        result = agent.type_text("input", "hello", clear_first=True)
        page.fill.assert_called_once_with("input", "", timeout=agent.default_timeout)
        page.type.assert_called_once_with("input", "hello")
        assert result["typed"] == "hello"

    def test_type_text_no_clear(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = MagicMock()
        agent.type_text("input", "hello", clear_first=False)
        # page.fill should NOT have been called for clearing
        for call_args in page.fill.call_args_list:
            args, _ = call_args
            assert not (len(args) >= 2 and args[1] == "")

    def test_fill(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = MagicMock()
        result = agent.fill("input", "world")
        page.fill.assert_called_once_with("input", "world", timeout=agent.default_timeout)
        assert result["filled"] == "world"

    def test_press_key(self):
        agent, page, _ = _make_agent()
        result = agent.press_key("Enter")
        page.keyboard.press.assert_called_once_with("Enter")
        assert result["key"] == "Enter"

    def test_hover(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = MagicMock()
        result = agent.hover("a.link")
        page.hover.assert_called_once_with("a.link")
        assert result["hovered"] == "a.link"

    def test_select_option(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = MagicMock()
        result = agent.select_option("select#lang", "en")
        page.select_option.assert_called_once_with("select#lang", "en")
        assert result["selected"] == "en"


# ---------------------------------------------------------------------------
# Scrolling
# ---------------------------------------------------------------------------

class TestScrolling:
    def test_scroll_default(self):
        agent, page, _ = _make_agent()
        result = agent.scroll()
        page.mouse.wheel.assert_called_once_with(0, 500)
        assert result == {"scrolled": {"x": 0, "y": 500}}

    def test_scroll_custom(self):
        agent, page, _ = _make_agent()
        result = agent.scroll(x=100, y=300)
        page.mouse.wheel.assert_called_once_with(100, 300)
        assert result["scrolled"] == {"x": 100, "y": 300}

    def test_scroll_to_element(self):
        agent, page, _ = _make_agent()
        el = MagicMock()
        page.query_selector.return_value = el
        result = agent.scroll_to_element("h1")
        el.scroll_into_view_if_needed.assert_called_once()
        assert result["scrolled_to"] == "h1"

    def test_scroll_to_element_disappeared_after_resolve(self):
        agent, page, _ = _make_agent()
        call_count = [0]

        def qsel(sel):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock()  # resolve_selector passes
            return None  # second call returns None → element disappeared

        page.query_selector.side_effect = qsel
        with pytest.raises(ValueError, match="disappeared"):
            agent.scroll_to_element("h1")


# ---------------------------------------------------------------------------
# Information extraction
# ---------------------------------------------------------------------------

class TestInformationExtraction:
    def test_get_title(self):
        agent, page, _ = _make_agent()
        assert agent.get_title() == "Test Page"

    def test_get_url(self):
        agent, page, _ = _make_agent()
        assert agent.get_url() == "https://example.com"

    def test_get_text_default_selector(self):
        agent, page, _ = _make_agent()
        page.inner_text.return_value = "hello world"
        result = agent.get_text()
        page.inner_text.assert_called_with("body")
        assert result == "hello world"

    def test_get_text_custom_selector(self):
        agent, page, _ = _make_agent()
        agent.get_text("h1")
        page.inner_text.assert_called_with("h1")

    def test_get_html(self):
        agent, page, _ = _make_agent()
        page.inner_html.return_value = "<h1>Title</h1>"
        assert agent.get_html("body") == "<h1>Title</h1>"

    def test_get_attribute(self):
        agent, page, _ = _make_agent()
        page.get_attribute.return_value = "https://link.com"
        assert agent.get_attribute("a", "href") == "https://link.com"

    def test_query_all_empty(self):
        agent, page, _ = _make_agent()
        page.query_selector_all.return_value = []
        assert agent.query_all("div") == []

    def test_query_all_with_elements(self):
        agent, page, _ = _make_agent()
        el1 = MagicMock()
        el1.inner_text.return_value = " Item 1 "
        el1.get_attribute.return_value = "/page1"
        el2 = MagicMock()
        el2.inner_text.return_value = "Item 2"
        el2.get_attribute.return_value = None
        page.query_selector_all.return_value = [el1, el2]
        result = agent.query_all("a")
        assert len(result) == 2
        assert result[0]["text"] == "Item 1"
        assert result[0]["href"] == "/page1"
        assert result[1]["href"] is None


# ---------------------------------------------------------------------------
# JavaScript / screenshots
# ---------------------------------------------------------------------------

class TestJsAndScreenshot:
    def test_evaluate(self):
        agent, page, _ = _make_agent()
        page.evaluate.return_value = 42
        assert agent.evaluate("1 + 1") == 42

    def test_screenshot_returns_base64_when_no_path(self):
        agent, page, _ = _make_agent()
        page.screenshot.return_value = b"\x89PNG"
        result = agent.screenshot()
        assert "base64" in result
        assert result["base64"] == base64.b64encode(b"\x89PNG").decode()

    def test_screenshot_with_path(self):
        agent, page, _ = _make_agent()
        result = agent.screenshot(path="/tmp/shot.png")
        _, kwargs = page.screenshot.call_args
        assert kwargs.get("path") == "/tmp/shot.png"
        assert result["path"] == "/tmp/shot.png"

    def test_screenshot_as_base64_flag(self):
        agent, page, _ = _make_agent()
        page.screenshot.return_value = b"\x89PNG"
        result = agent.screenshot(path="/tmp/shot.png", as_base64=True)
        assert "base64" in result

    def test_screenshot_full_page(self):
        agent, page, _ = _make_agent()
        agent.screenshot(full_page=True)
        _, kwargs = page.screenshot.call_args
        assert kwargs.get("full_page") is True


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------

class TestWaitHelpers:
    def test_wait_for_selector(self):
        agent, page, _ = _make_agent()
        result = agent.wait_for_selector("div.container")
        page.wait_for_selector.assert_called_once_with("div.container", timeout=agent.default_timeout)
        assert result["visible"] == "div.container"

    def test_wait_for_selector_custom_timeout(self):
        agent, page, _ = _make_agent()
        agent.wait_for_selector("div", timeout=5000)
        _, kwargs = page.wait_for_selector.call_args
        assert kwargs.get("timeout") == 5000

    def test_wait_for_load_state(self):
        agent, page, _ = _make_agent()
        result = agent.wait_for_load_state("load")
        page.wait_for_load_state.assert_called_once_with("load")
        assert result["state"] == "load"

    def test_wait_for_load_state_domcontentloaded(self):
        agent, page, _ = _make_agent()
        result = agent.wait_for_load_state("domcontentloaded")
        assert result["state"] == "domcontentloaded"

    def test_wait_for_load_state_networkidle_delegates(self):
        agent, page, _ = _make_agent()
        with patch.object(agent, "_try_networkidle", return_value=True) as mock_ni:
            result = agent.wait_for_load_state("networkidle")
        mock_ni.assert_called_once()
        assert result["state"] == "networkidle"
        assert result["reached"] is True

    def test_wait_for_navigation(self):
        agent, page, _ = _make_agent()
        result = agent.wait_for_navigation()
        # Should use wait_for_load_state, not expect_navigation
        page.wait_for_load_state.assert_called_once_with("domcontentloaded")
        assert "url" in result

    def test_wait_for_navigation_handles_exception(self):
        """wait_for_navigation should not raise if the page is already loaded."""
        agent, page, _ = _make_agent()
        page.wait_for_load_state.side_effect = Exception("already loaded")
        result = agent.wait_for_navigation()
        assert "url" in result

    def test_get_page_info(self):
        agent, page, _ = _make_agent()
        result = agent.get_page_info()
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"
        assert "viewport" in result


# ---------------------------------------------------------------------------
# Smart extraction
# ---------------------------------------------------------------------------

class TestSmartExtraction:
    def test_extract_links_returns_list(self):
        agent, page, _ = _make_agent()
        page.evaluate.return_value = [
            {"text": "Link 1", "href": "https://example.com/1"},
            {"text": "Link 2", "href": "https://example.com/2"},
        ]
        result = agent.extract_links()
        assert result["count"] == 2
        assert len(result["links"]) == 2
        assert result["selector"] == "a"

    def test_extract_links_custom_params(self):
        agent, page, _ = _make_agent()
        page.evaluate.return_value = []
        result = agent.extract_links(selector="a.nav", limit=10)
        assert result["selector"] == "a.nav"
        assert page.evaluate.called

    def test_extract_table_success(self):
        agent, page, _ = _make_agent()
        table_data = {
            "headers": ["Name", "Age"],
            "rows": [{"Name": "Alice", "Age": "30"}],
            "count": 1,
        }
        page.evaluate.return_value = table_data
        result = agent.extract_table()
        assert result["count"] == 1
        assert result["headers"] == ["Name", "Age"]

    def test_extract_table_not_found_raises(self):
        agent, page, _ = _make_agent()
        page.evaluate.return_value = None
        with pytest.raises(ValueError, match="No table found"):
            agent.extract_table()

    def test_extract_table_custom_index(self):
        agent, page, _ = _make_agent()
        table_data = {"headers": ["A"], "rows": [], "count": 0}
        page.evaluate.return_value = table_data
        agent.extract_table(selector="#mytable", table_index=2)
        assert page.evaluate.called


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

class TestAssertions:
    def test_assert_text_found_case_insensitive(self):
        agent, page, _ = _make_agent()
        page.inner_text.return_value = "Hello World on the page"
        result = agent.assert_text("hello world")
        assert result["found"] is True

    def test_assert_text_not_found_raises(self):
        agent, page, _ = _make_agent()
        page.inner_text.return_value = "Nothing relevant here"
        with pytest.raises(AssertionError):
            agent.assert_text("missing text")

    def test_assert_text_case_sensitive_match(self):
        agent, page, _ = _make_agent()
        page.inner_text.return_value = "Hello World"
        result = agent.assert_text("Hello World", case_sensitive=True)
        assert result["found"] is True

    def test_assert_text_case_sensitive_fail(self):
        agent, page, _ = _make_agent()
        page.inner_text.return_value = "Hello World"
        with pytest.raises(AssertionError):
            agent.assert_text("hello world", case_sensitive=True)

    def test_assert_text_custom_selector(self):
        agent, page, _ = _make_agent()
        page.inner_text.return_value = "found it"
        result = agent.assert_text("found it", selector="h1")
        page.inner_text.assert_called_with("h1")
        assert result["selector"] == "h1"

    def test_assert_url_matches(self):
        agent, page, _ = _make_agent()
        page.url = "https://example.com/dashboard"
        result = agent.assert_url("/dashboard")
        assert result["matched"] is True

    def test_assert_url_not_matched_raises(self):
        agent, page, _ = _make_agent()
        page.url = "https://example.com"
        with pytest.raises(AssertionError, match="does not contain"):
            agent.assert_url("/missing")


# ---------------------------------------------------------------------------
# Wait for dynamic content
# ---------------------------------------------------------------------------

class TestWaitText:
    def test_wait_text_success(self):
        agent, page, _ = _make_agent()
        result = agent.wait_text("Loading complete")
        page.wait_for_function.assert_called_once()
        assert result["found"] is True
        assert result["text"] == "Loading complete"

    def test_wait_text_custom_selector(self):
        agent, page, _ = _make_agent()
        agent.wait_text("Done", selector=".status")
        _, kwargs = page.wait_for_function.call_args
        assert kwargs.get("arg") == [".status", "Done"]

    def test_wait_text_custom_timeout(self):
        agent, page, _ = _make_agent()
        agent.wait_text("Done", timeout=5000)
        _, kwargs = page.wait_for_function.call_args
        assert kwargs.get("timeout") == 5000

    def test_wait_text_default_timeout(self):
        agent, page, _ = _make_agent(default_timeout=15000)
        agent.wait_text("Ready")
        _, kwargs = page.wait_for_function.call_args
        assert kwargs.get("timeout") == 15000


# ---------------------------------------------------------------------------
# Cookie persistence
# ---------------------------------------------------------------------------

class TestCookies:
    def test_get_cookies(self):
        agent, page, context = _make_agent()
        context.cookies.return_value = [{"name": "sid", "value": "abc"}]
        cookies = agent.get_cookies()
        assert len(cookies) == 1
        assert cookies[0]["name"] == "sid"

    def test_add_cookies(self):
        agent, page, context = _make_agent()
        agent.add_cookies([{"name": "tok", "value": "xyz"}])
        context.add_cookies.assert_called_once_with([{"name": "tok", "value": "xyz"}])


# ---------------------------------------------------------------------------
# Multi-tab management
# ---------------------------------------------------------------------------

class TestMultiTab:
    def test_new_tab_without_url(self):
        agent, page, context = _make_agent()
        new_page = MagicMock()
        new_page.url = "about:blank"
        new_page.title.return_value = "New Tab"
        context.new_page.return_value = new_page

        result = agent.new_tab()

        assert result["tab_index"] == 1
        assert len(agent._pages) == 2
        assert agent._page is new_page

    def test_new_tab_with_url_navigates(self):
        agent, page, context = _make_agent()
        new_page = MagicMock()
        new_page.url = "https://example.com"
        new_page.title.return_value = "Example"
        context.new_page.return_value = new_page

        with patch.object(agent, "navigate", return_value={}) as mock_nav:
            agent.new_tab(url="https://example.com")

        mock_nav.assert_called_once_with("https://example.com")

    def test_new_tab_dialog_handler_registered(self):
        agent, page, context = _make_agent()
        new_page = MagicMock()
        new_page.url = "about:blank"
        new_page.title.return_value = "Tab"
        context.new_page.return_value = new_page

        agent.new_tab()
        new_page.on.assert_called_once_with("dialog", agent._handle_dialog)

    def test_new_tab_requires_context(self):
        agent, page, context = _make_agent()
        agent._context = None
        with pytest.raises(RuntimeError, match="not started"):
            agent.new_tab()

    def test_switch_tab_valid_index(self):
        agent, page, context = _make_agent()
        page2 = MagicMock()
        page2.url = "https://tab2.com"
        page2.title.return_value = "Tab 2"
        agent._pages = [page, page2]

        result = agent.switch_tab(1)

        assert result["tab_index"] == 1
        assert agent._page is page2

    def test_switch_tab_out_of_range(self):
        agent, page, context = _make_agent()
        with pytest.raises(ValueError, match="out of range"):
            agent.switch_tab(5)

    def test_switch_tab_negative_index(self):
        agent, page, context = _make_agent()
        with pytest.raises(ValueError, match="out of range"):
            agent.switch_tab(-1)

    def test_close_tab_active_switches_to_last(self):
        agent, page, context = _make_agent()
        page2 = MagicMock()
        page2.url = "https://tab2.com"
        page2.title.return_value = "Tab 2"
        agent._pages = [page, page2]
        agent._page = page

        result = agent.close_tab(index=0)

        assert result["closed_index"] == 0
        assert result["remaining_tabs"] == 1
        assert agent._page is page2

    def test_close_tab_current_tab_by_default(self):
        agent, page, context = _make_agent()
        page2 = MagicMock()
        agent._pages = [page, page2]
        agent._page = page2

        result = agent.close_tab()

        assert result["closed_index"] == 1

    def test_close_tab_last_tab_raises(self):
        agent, page, context = _make_agent()
        with pytest.raises(RuntimeError, match="Cannot close the last"):
            agent.close_tab()

    def test_close_tab_out_of_range(self):
        agent, page, context = _make_agent()
        page2 = MagicMock()
        agent._pages = [page, page2]
        with pytest.raises(ValueError, match="out of range"):
            agent.close_tab(index=5)

    def test_list_tabs_single_tab(self):
        agent, page, context = _make_agent()
        result = agent.list_tabs()
        assert result["count"] == 1
        assert result["tabs"][0]["active"] is True

    def test_list_tabs_multiple_tabs(self):
        agent, page, context = _make_agent()
        page2 = MagicMock()
        page2.url = "https://tab2.com"
        page2.title.return_value = "Tab 2"
        agent._pages = [page, page2]

        result = agent.list_tabs()

        assert result["count"] == 2
        assert result["tabs"][0]["active"] is True
        assert result["tabs"][1]["active"] is False

    def test_list_tabs_handles_closed_page_exception(self):
        agent, page, context = _make_agent()
        bad_page = MagicMock()
        bad_page.title.side_effect = Exception("page closed")
        agent._pages = [page, bad_page]

        result = agent.list_tabs()

        assert result["count"] == 2
        assert result["tabs"][1]["url"] == "unknown"


# ---------------------------------------------------------------------------
# URL scheme validation
# ---------------------------------------------------------------------------

class TestNavigateURLValidation:
    def test_navigate_http_allowed(self):
        agent, page, _ = _make_agent()
        page.url = "http://example.com"
        result = agent.navigate("http://example.com")
        assert "url" in result

    def test_navigate_https_allowed(self):
        agent, page, _ = _make_agent()
        result = agent.navigate("https://example.com")
        assert "url" in result

    def test_navigate_javascript_blocked(self):
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            agent.navigate("javascript:alert(1)")

    def test_navigate_file_blocked(self):
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            agent.navigate("file:///etc/passwd")

    def test_navigate_data_blocked(self):
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            agent.navigate("data:text/html,<h1>xss</h1>")

    def test_navigate_no_scheme_blocked(self):
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            agent.navigate("example.com")


# ---------------------------------------------------------------------------
# _intercept_patterns initialized in __init__
# ---------------------------------------------------------------------------

class TestInterceptPatternsInit:
    def test_intercept_patterns_initialized(self):
        """_intercept_patterns should be a list from the start, not require hasattr."""
        agent = BrowserAgent()
        assert hasattr(agent, "_intercept_patterns")
        assert isinstance(agent._intercept_patterns, list)
        assert len(agent._intercept_patterns) == 0

    def test_clear_network_intercepts_on_fresh_agent(self):
        """clear_network_intercepts should work even without prior set_network_intercept."""
        agent, page, _ = _make_agent()
        result = agent.clear_network_intercepts()
        assert result["cleared"] == 0
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# set_network_intercept intercept_action key (Bug fix regression)
# ---------------------------------------------------------------------------

class TestSetNetworkIntercept:
    def test_set_intercept_appends_pattern(self):
        agent, page, _ = _make_agent()
        agent.set_network_intercept("**/*.png", action="abort")
        assert "**/*.png" in agent._intercept_patterns

    def test_set_intercept_invalid_action_raises(self):
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="action must be 'abort' or 'continue'"):
            agent.set_network_intercept("**/*.png", action="block")

    def test_clear_intercepts_resets_patterns(self):
        agent, page, _ = _make_agent()
        agent.set_network_intercept("**/*.png", action="abort")
        result = agent.clear_network_intercepts()
        assert result["cleared"] == 1
        assert len(agent._intercept_patterns) == 0


# ---------------------------------------------------------------------------
# download_file URL scheme validation (Bug fix regression)
# ---------------------------------------------------------------------------

class TestDownloadFileURLValidation:
    def test_download_http_allowed(self):
        """http:// URLs should be accepted and page.goto called."""
        agent, page, _ = _make_agent()
        dl = MagicMock()
        dl.suggested_filename = "file.zip"
        page.expect_download.return_value.__enter__ = MagicMock(return_value=MagicMock(value=dl))
        page.expect_download.return_value.__exit__ = MagicMock(return_value=False)
        # Just asserting no ValueError is raised
        try:
            agent.download_file("http://example.com/file.zip", "/tmp/file.zip")
        except Exception as exc:
            # Any exception other than ValueError for scheme is acceptable
            # (the mock may raise on save_as, etc.)
            assert "Unsafe URL scheme" not in str(exc)

    def test_download_https_allowed(self):
        """https:// URLs should be accepted."""
        agent, page, _ = _make_agent()
        try:
            agent.download_file("https://example.com/file.zip", "/tmp/file.zip")
        except Exception as exc:
            assert "Unsafe URL scheme" not in str(exc)

    def test_download_javascript_blocked(self):
        """javascript: URLs must be rejected before page.goto is called."""
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            agent.download_file("javascript:alert(1)", "/tmp/evil.js")
        page.goto.assert_not_called()

    def test_download_file_blocked(self):
        """file:// URLs must be rejected."""
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            agent.download_file("file:///etc/passwd", "/tmp/passwd")
        page.goto.assert_not_called()

    def test_download_data_blocked(self):
        """data: URIs must be rejected."""
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            agent.download_file("data:text/html,<h1>xss</h1>", "/tmp/xss.html")
        page.goto.assert_not_called()

    def test_download_no_scheme_blocked(self):
        """A plain hostname with no scheme must be rejected."""
        agent, page, _ = _make_agent()
        with pytest.raises(ValueError, match="Unsafe URL scheme"):
            agent.download_file("example.com/file.zip", "/tmp/file.zip")
        page.goto.assert_not_called()


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

class TestVideoRecording:
    """BrowserAgent.start_video_recording / stop_video_recording."""

    def _make_recording_agent(self):
        """Agent with a fully mocked browser so _make_context() works."""
        agent, page, context = _make_agent()

        browser = MagicMock()
        new_ctx = MagicMock()
        new_page = MagicMock()
        new_page.url = "about:blank"
        new_page.video = MagicMock()
        new_page.video.path.return_value = "/tmp/playwright-video-xyz.webm"
        new_ctx.new_page.return_value = new_page
        browser.new_context.return_value = new_ctx
        agent._browser = browser

        # pre-existing page URL so navigate-back can be tested
        page.url = "about:blank"
        return agent, page, context, browser, new_ctx, new_page

    def test_start_sets_recording_flag(self, tmp_path):
        agent, *_ = self._make_recording_agent()
        save = str(tmp_path / "out.webm")
        agent.start_video_recording(save)
        assert agent._recording is True
        assert agent._recording_path == save

    def test_start_creates_context_with_record_video_dir(self, tmp_path):
        agent, _, _, browser, *_ = self._make_recording_agent()
        save = str(tmp_path / "video.webm")
        agent.start_video_recording(save)
        call_kwargs = browser.new_context.call_args[1]
        assert "record_video_dir" in call_kwargs
        assert call_kwargs["record_video_dir"] == str(tmp_path)

    def test_start_raises_if_already_recording(self, tmp_path):
        agent, *_ = self._make_recording_agent()
        agent.start_video_recording(str(tmp_path / "a.webm"))
        with pytest.raises(RuntimeError, match="already active"):
            agent.start_video_recording(str(tmp_path / "b.webm"))

    def test_stop_clears_recording_flag(self, tmp_path):
        agent, *_, new_ctx, new_page = self._make_recording_agent()
        save = str(tmp_path / "out.webm")
        agent.start_video_recording(save)
        # point _page at new_page so stop_video_recording can read .video.path()
        agent._page = new_page
        agent._pages = [new_page]
        with patch("shutil.move"):
            result = agent.stop_video_recording()
        assert agent._recording is False
        assert agent._recording_path is None
        assert result["ok"] is True

    def test_stop_raises_if_not_recording(self):
        agent, *_ = _make_agent()
        agent._browser = MagicMock()
        with pytest.raises(RuntimeError, match="No video recording"):
            agent.stop_video_recording()

    def test_stop_renames_video_file(self, tmp_path):
        agent, _, _, browser, new_ctx, new_page = self._make_recording_agent()
        desired = str(tmp_path / "session.webm")
        raw_path = "/tmp/playwright-video-xyz.webm"
        new_page.video.path.return_value = raw_path

        agent.start_video_recording(desired)
        agent._page = new_page
        agent._pages = [new_page]

        with patch("shutil.move") as mv:
            result = agent.stop_video_recording()
        mv.assert_called_once_with(raw_path, desired)
        assert result["saved_to"] == desired

    def test_stop_warning_on_shutdown_while_recording(self):
        agent, *_ = _make_agent()
        agent._recording = True
        agent._browser = MagicMock()
        agent._playwright = MagicMock()
        import logging
        with patch.object(logging.getLogger("browser_agent"), "warning") as warn:
            agent.stop()
        warn.assert_called()


# ---------------------------------------------------------------------------
# GIF recording
# ---------------------------------------------------------------------------

class TestRecordGif:
    """BrowserAgent.record_gif."""

    def test_record_gif_saves_file(self, tmp_path):
        agent, page, _ = _make_agent()
        dest = tmp_path / "out.gif"

        # Stub PIL.Image so we don't need a real PNG decoder.
        fake_img = MagicMock()
        fake_img.save = MagicMock()
        with patch("PIL.Image.open", return_value=fake_img), \
             patch("time.sleep"):
            result = agent.record_gif(str(dest), duration=1.0, fps=2)

        assert result["ok"] is True
        assert result["frames"] == 2
        assert result["fps"] == 2
        fake_img.save.assert_called_once()

    def test_record_gif_respects_fps_duration(self, tmp_path):
        agent, page, _ = _make_agent()
        fake_img = MagicMock()
        with patch("PIL.Image.open", return_value=fake_img), \
             patch("time.sleep"):
            result = agent.record_gif(str(tmp_path / "g.gif"), duration=2.0, fps=3)
        assert result["frames"] == 6  # ceil(2.0 * 3)

    def test_record_gif_creates_parent_dirs(self, tmp_path):
        agent, page, _ = _make_agent()
        dest = tmp_path / "subdir" / "deep" / "out.gif"
        fake_img = MagicMock()
        with patch("PIL.Image.open", return_value=fake_img), \
             patch("time.sleep"):
            agent.record_gif(str(dest), duration=0.5, fps=1)
        assert dest.parent.exists()

    def test_record_gif_min_one_frame(self, tmp_path):
        """duration * fps < 1 should still produce at least one frame."""
        agent, page, _ = _make_agent()
        fake_img = MagicMock()
        with patch("PIL.Image.open", return_value=fake_img), \
             patch("time.sleep"):
            result = agent.record_gif(str(tmp_path / "g.gif"), duration=0.1, fps=1)
        assert result["frames"] >= 1
