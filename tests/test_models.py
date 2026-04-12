"""Tests for agenticbrowser models."""

from agenticbrowser.models import (
    ActionResult,
    BrowserOptions,
    Element,
    ElementsResult,
    NavigateResult,
    ScreenshotResult,
    TextResult,
)


class TestActionResult:
    def test_success_is_truthy(self):
        result = ActionResult(success=True)
        assert bool(result) is True

    def test_failure_is_falsy(self):
        result = ActionResult(success=False, error="something went wrong")
        assert bool(result) is False

    def test_error_is_none_by_default(self):
        result = ActionResult(success=True)
        assert result.error is None


class TestNavigateResult:
    def test_defaults(self):
        result = NavigateResult(success=True)
        assert result.url == ""
        assert result.title == ""
        assert result.status is None

    def test_fields(self):
        result = NavigateResult(
            success=True,
            url="https://example.com",
            title="Example Domain",
            status=200,
        )
        assert result.url == "https://example.com"
        assert result.title == "Example Domain"
        assert result.status == 200

    def test_failure(self):
        result = NavigateResult(success=False, error="timeout", url="https://x.com")
        assert not result
        assert result.error == "timeout"


class TestScreenshotResult:
    def test_defaults(self):
        result = ScreenshotResult(success=True)
        assert result.path == ""
        assert result.data is None

    def test_with_data(self):
        data = b"\x89PNG\r\n"
        result = ScreenshotResult(success=True, path="/tmp/shot.png", data=data)
        assert result.path == "/tmp/shot.png"
        assert result.data == data


class TestTextResult:
    def test_defaults(self):
        result = TextResult(success=True)
        assert result.text == ""
        assert result.url == ""
        assert result.title == ""

    def test_with_content(self):
        result = TextResult(
            success=True,
            text="Hello World",
            url="https://example.com",
            title="Test",
        )
        assert result.text == "Hello World"


class TestElement:
    def test_basic(self):
        el = Element(tag="a", text="Click me", href="https://example.com")
        assert el.tag == "a"
        assert el.text == "Click me"
        assert el.href == "https://example.com"

    def test_optional_fields_none_by_default(self):
        el = Element(tag="div", text="content")
        assert el.href is None
        assert el.id is None
        assert el.class_name is None
        assert el.selector is None

    def test_attributes_empty_by_default(self):
        el = Element(tag="span", text="")
        assert el.attributes == {}


class TestElementsResult:
    def test_empty(self):
        result = ElementsResult(success=True)
        assert result.elements == []
        assert result.count == 0

    def test_with_elements(self):
        elements = [
            Element(tag="a", text="Link 1", href="/page1"),
            Element(tag="a", text="Link 2", href="/page2"),
        ]
        result = ElementsResult(success=True, elements=elements, count=2)
        assert len(result.elements) == 2
        assert result.count == 2


class TestBrowserOptions:
    def test_defaults(self):
        opts = BrowserOptions()
        assert opts.headless is True
        assert opts.browser_type == "chromium"
        assert opts.timeout == 30_000
        assert opts.viewport_width == 1280
        assert opts.viewport_height == 720
        assert opts.user_agent is None
        assert opts.slow_mo == 0
        assert opts.proxy is None
        assert opts.ignore_https_errors is False

    def test_custom(self):
        opts = BrowserOptions(
            headless=False,
            browser_type="firefox",
            timeout=10_000,
        )
        assert opts.headless is False
        assert opts.browser_type == "firefox"
        assert opts.timeout == 10_000
