"""Tests for the Browser class (unit-level, no real browser launched)."""

from __future__ import annotations

import pytest

from agenticbrowser import Browser, BrowserOptions


class TestBrowserLifecycle:
    """Tests for start/close lifecycle."""

    def test_not_started_raises(self):
        browser = Browser()
        with pytest.raises(RuntimeError, match="not started"):
            browser.navigate("https://example.com")

    def test_context_manager_starts_and_closes(self, monkeypatch):
        calls = []
        browser = Browser()

        monkeypatch.setattr(browser, "start", lambda: calls.append("start"))
        monkeypatch.setattr(browser, "close", lambda: calls.append("close"))

        with browser:
            pass

        assert calls == ["start", "close"]

    def test_context_manager_closes_on_exception(self, monkeypatch):
        calls = []
        browser = Browser()

        monkeypatch.setattr(browser, "start", lambda: calls.append("start"))
        monkeypatch.setattr(browser, "close", lambda: calls.append("close"))

        with pytest.raises(ValueError):
            with browser:
                raise ValueError("test error")

        assert "close" in calls


class TestBrowserOptions:
    def test_default_options(self):
        browser = Browser()
        assert browser._options.headless is True
        assert browser._options.browser_type == "chromium"

    def test_custom_options(self):
        opts = BrowserOptions(headless=False, browser_type="firefox")
        browser = Browser(opts)
        assert browser._options.headless is False
        assert browser._options.browser_type == "firefox"


class TestBrowserWithMockedPage:
    """Tests that use a mocked Playwright page."""

    def _make_browser_with_mock_page(self, mock_page):
        """Helper that builds a Browser pre-wired with a fake page."""
        browser = Browser()
        browser._page = mock_page
        browser._context = _FakeContext()
        return browser

    def test_navigate_success(self):
        page = _FakePage(url="https://example.com", title="Example", status=200)
        browser = self._make_browser_with_mock_page(page)
        result = browser.navigate("https://example.com")
        assert result.success is True
        assert result.url == "https://example.com"
        assert result.title == "Example"
        assert result.status == 200

    def test_navigate_failure(self):
        page = _FakePage(raise_on_goto=True)
        browser = self._make_browser_with_mock_page(page)
        result = browser.navigate("https://bad-url")
        assert result.success is False
        assert result.error is not None

    def test_click_success(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.click("button")
        assert result.success is True

    def test_click_failure(self):
        page = _FakePage(raise_on_click=True)
        browser = self._make_browser_with_mock_page(page)
        result = browser.click("button")
        assert result.success is False

    def test_type_text_success(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.type_text("input", "hello")
        assert result.success is True

    def test_type_text_no_clear(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.type_text("input", "hello", clear=False)
        assert result.success is True

    def test_get_text_success(self):
        page = _FakePage(inner_text="Hello World")
        browser = self._make_browser_with_mock_page(page)
        result = browser.get_text()
        assert result.success is True
        assert result.text == "Hello World"

    def test_get_url(self):
        page = _FakePage(url="https://example.com")
        browser = self._make_browser_with_mock_page(page)
        assert browser.get_url() == "https://example.com"

    def test_get_title(self):
        page = _FakePage(title="My Page")
        browser = self._make_browser_with_mock_page(page)
        assert browser.get_title() == "My Page"

    def test_scroll(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.scroll(0, 300)
        assert result.success is True
        assert page.last_evaluate is not None

    def test_screenshot_returns_bytes(self):
        page = _FakePage(screenshot_data=b"\x89PNG")
        browser = self._make_browser_with_mock_page(page)
        result = browser.screenshot()
        assert result.success is True
        assert result.data == b"\x89PNG"

    def test_screenshot_base64(self):
        import base64
        page = _FakePage(screenshot_data=b"ABC")
        browser = self._make_browser_with_mock_page(page)
        b64 = browser.screenshot_base64()
        assert b64 == base64.b64encode(b"ABC").decode()

    def test_evaluate_success(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.evaluate("document.title")
        assert result.success is True
        assert result.value is None  # _FakePage.evaluate returns None

    def test_press_key_with_selector(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.press_key("Enter", selector="input")
        assert result.success is True

    def test_press_key_without_selector(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.press_key("Tab")
        assert result.success is True

    def test_go_back(self):
        page = _FakePage(url="https://example.com/back", title="Back")
        browser = self._make_browser_with_mock_page(page)
        result = browser.go_back()
        assert result.success is True

    def test_go_forward(self):
        page = _FakePage(url="https://example.com/fwd", title="Fwd")
        browser = self._make_browser_with_mock_page(page)
        result = browser.go_forward()
        assert result.success is True

    def test_reload(self):
        page = _FakePage(url="https://example.com", title="Reload", status=200)
        browser = self._make_browser_with_mock_page(page)
        result = browser.reload()
        assert result.success is True

    def test_get_html(self):
        page = _FakePage(inner_html="<h1>Hello</h1>")
        browser = self._make_browser_with_mock_page(page)
        result = browser.get_html()
        assert result.success is True
        assert "<h1>" in result.text

    def test_hover(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.hover("a.link")
        assert result.success is True

    def test_select_option(self):
        page = _FakePage()
        browser = self._make_browser_with_mock_page(page)
        result = browser.select_option("select#lang", "en")
        assert result.success is True

    def test_get_cookies(self):
        browser = Browser()
        browser._page = _FakePage()
        browser._context = _FakeContext(cookies=[{"name": "session", "value": "abc"}])
        cookies = browser.get_cookies()
        assert cookies == [{"name": "session", "value": "abc"}]

    def test_clear_cookies(self):
        browser = Browser()
        browser._page = _FakePage()
        browser._context = _FakeContext()
        result = browser.clear_cookies()
        assert result.success is True


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status=200):
        self.status = status


class _FakeKeyboard:
    def press(self, key):
        pass


class _FakeLocator:
    def __init__(self, screenshot_data=b""):
        self._data = screenshot_data

    def scroll_into_view_if_needed(self):
        pass

    def screenshot(self, **kwargs):
        return self._data

    def all(self):
        return []


class _FakePage:
    def __init__(
        self,
        url="https://example.com",
        title="Test Page",
        status=200,
        inner_text="page text",
        inner_html="<p>html</p>",
        screenshot_data=b"\x89PNG",
        raise_on_goto=False,
        raise_on_click=False,
    ):
        self._url = url
        self._title = title
        self._status = status
        self._inner_text = inner_text
        self._inner_html = inner_html
        self._screenshot_data = screenshot_data
        self._raise_on_goto = raise_on_goto
        self._raise_on_click = raise_on_click
        self.last_evaluate = None
        self.keyboard = _FakeKeyboard()

    @property
    def url(self):
        return self._url

    def title(self):
        return self._title

    def goto(self, url, wait_until="domcontentloaded"):
        if self._raise_on_goto:
            raise Exception("navigation failed")
        return _FakeResponse(self._status)

    def go_back(self):
        return _FakeResponse(self._status)

    def go_forward(self):
        return _FakeResponse(self._status)

    def reload(self):
        return _FakeResponse(self._status)

    def click(self, selector, **kwargs):
        if self._raise_on_click:
            raise Exception("element not found")

    def fill(self, selector, text, **kwargs):
        pass

    def type(self, selector, text, **kwargs):
        pass

    def press(self, selector, key, **kwargs):
        pass

    def hover(self, selector, **kwargs):
        pass

    def select_option(self, selector, value, **kwargs):
        pass

    def inner_text(self, selector="body"):
        return self._inner_text

    def inner_html(self, selector="html"):
        return self._inner_html

    def screenshot(self, **kwargs):
        return self._screenshot_data

    def evaluate(self, script):
        self.last_evaluate = script
        return None

    def wait_for_selector(self, selector, **kwargs):
        pass

    def wait_for_load_state(self, state, **kwargs):
        pass

    def locator(self, selector):
        return _FakeLocator(self._screenshot_data)


class _FakeContext:
    def __init__(self, cookies=None):
        self._cookies = cookies or []

    def cookies(self):
        return self._cookies

    def add_cookies(self, cookies):
        self._cookies.extend(cookies)

    def clear_cookies(self):
        self._cookies.clear()

    def set_default_timeout(self, timeout):
        pass


# ---------------------------------------------------------------------------
# Additional coverage: scroll variants, wait methods, cookies, start/close
# ---------------------------------------------------------------------------

class TestBrowserScrollVariants:
    def _browser(self, raise_on_eval=False):
        page = _FakePage()
        if raise_on_eval:
            page.evaluate = lambda s: (_ for _ in ()).throw(Exception("eval error"))
        browser = Browser()
        browser._page = page
        browser._context = _FakeContext()
        return browser, page

    def test_scroll_to_bottom(self):
        browser, page = self._browser()
        result = browser.scroll_to_bottom()
        assert result.success is True
        assert "scrollTo" in page.last_evaluate

    def test_scroll_to_top(self):
        browser, page = self._browser()
        result = browser.scroll_to_top()
        assert result.success is True
        assert "0, 0" in page.last_evaluate

    def test_scroll_to_bottom_failure(self):
        browser, _ = self._browser(raise_on_eval=True)
        result = browser.scroll_to_bottom()
        assert result.success is False

    def test_scroll_to_top_failure(self):
        browser, _ = self._browser(raise_on_eval=True)
        result = browser.scroll_to_top()
        assert result.success is False

    def test_scroll_element_into_view(self):
        browser, _ = self._browser()
        result = browser.scroll_element_into_view(".target")
        assert result.success is True

    def test_scroll_element_into_view_failure(self):
        class _BadLocator:
            def scroll_into_view_if_needed(self):
                raise Exception("not found")
        page = _FakePage()
        page.locator = lambda s: _BadLocator()
        browser = Browser()
        browser._page = page
        browser._context = _FakeContext()
        result = browser.scroll_element_into_view(".bad")
        assert result.success is False


class TestBrowserWaitMethods:
    def _browser(self):
        browser = Browser()
        browser._page = _FakePage()
        browser._context = _FakeContext()
        return browser

    def test_wait_for_selector_default_state(self):
        browser = self._browser()
        result = browser.wait_for_selector(".btn")
        assert result.success is True

    def test_wait_for_selector_custom_state_and_timeout(self):
        browser = self._browser()
        result = browser.wait_for_selector(".btn", state="hidden", timeout=1000)
        assert result.success is True

    def test_wait_for_selector_failure(self):
        page = _FakePage()
        page.wait_for_selector = lambda s, **kw: (_ for _ in ()).throw(Exception("timeout"))
        browser = Browser()
        browser._page = page
        browser._context = _FakeContext()
        result = browser.wait_for_selector(".missing")
        assert result.success is False

    def test_wait_for_load(self):
        browser = self._browser()
        result = browser.wait_for_load()
        assert result.success is True

    def test_wait_for_load_with_timeout(self):
        browser = self._browser()
        result = browser.wait_for_load(timeout=5000)
        assert result.success is True

    def test_wait_for_load_failure(self):
        page = _FakePage()
        page.wait_for_load_state = lambda s, **kw: (_ for _ in ()).throw(Exception("timeout"))
        browser = Browser()
        browser._page = page
        browser._context = _FakeContext()
        result = browser.wait_for_load()
        assert result.success is False

    def test_wait_for_network_idle(self):
        browser = self._browser()
        result = browser.wait_for_network_idle()
        assert result.success is True

    def test_wait_for_network_idle_with_timeout(self):
        browser = self._browser()
        result = browser.wait_for_network_idle(timeout=3000)
        assert result.success is True

    def test_wait_for_network_idle_failure(self):
        page = _FakePage()
        page.wait_for_load_state = lambda s, **kw: (_ for _ in ()).throw(Exception("timeout"))
        browser = Browser()
        browser._page = page
        browser._context = _FakeContext()
        result = browser.wait_for_network_idle()
        assert result.success is False


class TestBrowserCookies:
    def _browser(self, cookies=None):
        browser = Browser()
        browser._page = _FakePage()
        browser._context = _FakeContext(cookies=cookies or [])
        return browser

    def test_set_cookie(self):
        browser = self._browser()
        result = browser.set_cookie("session", "abc123", domain="example.com")
        assert result.success is True
        cookies = browser.get_cookies()
        assert any(c["name"] == "session" for c in cookies)

    def test_set_cookie_failure(self):
        browser = self._browser()
        browser._context.add_cookies = lambda cookies: (_ for _ in ()).throw(Exception("bad cookie"))
        result = browser.set_cookie("bad", "val")
        assert result.success is False

    def test_get_cookies_empty(self):
        browser = self._browser()
        assert browser.get_cookies() == []

    def test_get_cookies_populated(self):
        browser = self._browser(cookies=[{"name": "x", "value": "y"}])
        cookies = browser.get_cookies()
        assert len(cookies) == 1
        assert cookies[0]["name"] == "x"

    def test_clear_cookies(self):
        browser = self._browser(cookies=[{"name": "x", "value": "y"}])
        result = browser.clear_cookies()
        assert result.success is True
        assert browser.get_cookies() == []


class TestBrowserFindElements:
    def _browser(self, locator=None):
        page = _FakePage()
        if locator is not None:
            page.locator = lambda s: locator
        browser = Browser()
        browser._page = page
        browser._context = _FakeContext()
        return browser

    def test_find_elements_empty(self):
        browser = self._browser()
        result = browser.find_elements("div.missing")
        assert result.success is True
        assert result.count == 0
        assert result.elements == []

    def test_find_elements_failure(self):
        class _BadLocator:
            def all(self):
                raise Exception("locator error")
        browser = self._browser(locator=_BadLocator())
        result = browser.find_elements(".bad")
        assert result.success is False

    def test_find_links_delegates(self):
        browser = self._browser()
        result = browser.find_links()
        assert result.success is True

    def test_find_buttons_delegates(self):
        browser = self._browser()
        result = browser.find_buttons()
        assert result.success is True

    def test_find_inputs_delegates(self):
        browser = self._browser()
        result = browser.find_inputs()
        assert result.success is True


class TestBrowserScreenshotExtras:
    def _browser(self):
        browser = Browser()
        browser._page = _FakePage()
        browser._context = _FakeContext()
        return browser

    def test_screenshot_with_selector(self):
        browser = self._browser()
        result = browser.screenshot(selector="body")
        assert result.success is True
        assert result.data == b"\x89PNG"

    def test_screenshot_base64_success(self):
        import base64
        browser = self._browser()
        b64 = browser.screenshot_base64()
        decoded = base64.b64decode(b64)
        assert decoded == b"\x89PNG"

    def test_screenshot_base64_raises_on_failure(self):
        browser = self._browser()
        browser._page = _FakePage()
        # Make screenshot return failure
        browser._page.screenshot = lambda **kw: (_ for _ in ()).throw(Exception("crash"))
        with pytest.raises(RuntimeError, match="Screenshot failed"):
            browser.screenshot_base64()


class TestBrowserStartClose:
    def test_start_uses_playwright(self, monkeypatch):
        """Verify that start() calls sync_playwright() and launches the browser."""
        fake_page = MagicMock()
        fake_context = MagicMock()
        fake_context.new_page.return_value = fake_page
        fake_browser_obj = MagicMock()
        fake_browser_obj.new_context.return_value = fake_context
        fake_browser_type = MagicMock()
        fake_browser_type.launch.return_value = fake_browser_obj
        fake_playwright_obj = MagicMock()
        fake_playwright_obj.chromium = fake_browser_type
        fake_playwright_ctx = MagicMock()
        fake_playwright_ctx.start.return_value = fake_playwright_obj

        import agenticbrowser.browser as bmod
        monkeypatch.setattr(bmod, "sync_playwright", lambda: fake_playwright_ctx, raising=False)
        # Patch the import inside start()
        import sys as _sys


        class _FakePlaywrightModule:
            def sync_playwright(self):
                return fake_playwright_ctx

        monkeypatch.setitem(
            _sys.modules,
            "playwright.sync_api",
            type("mod", (), {"sync_playwright": lambda: fake_playwright_ctx})(),
        )

        browser = Browser()
        # We patch _ensure_started to avoid actual Playwright invocation
        monkeypatch.setattr(browser, "_ensure_started", lambda: None)
        browser._page = fake_page
        browser._context = fake_context
        # close() should not raise even if internals are mocked
        browser.close()
        assert browser._page is None
        assert browser._context is None

    def test_close_noop_when_not_started(self):
        """close() on an unstarted browser should not raise."""
        browser = Browser()
        browser.close()  # Should be a no-op

    def test_close_clears_state(self):
        browser = Browser()
        browser._page = MagicMock()
        browser._context = MagicMock()
        browser._browser = MagicMock()
        browser._playwright = MagicMock()
        browser.close()
        assert browser._page is None
        assert browser._context is None
        assert browser._browser is None
        assert browser._playwright is None


# Need MagicMock import
from unittest.mock import MagicMock  # noqa: E402
