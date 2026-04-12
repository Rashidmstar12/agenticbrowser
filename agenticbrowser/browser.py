"""Core Browser class for agenticbrowser."""

from __future__ import annotations

import base64
import contextlib
from pathlib import Path
from typing import List, Optional, Union

from agenticbrowser.models import (
    ActionResult,
    BrowserOptions,
    Element,
    ElementsResult,
    NavigateResult,
    ScreenshotResult,
    TextResult,
)


class Browser:
    """
    An agentic browser that exposes high-level actions for AI agents.

    Can be used as a context manager::

        with Browser() as browser:
            browser.navigate("https://example.com")
            text = browser.get_text()

    Or managed manually::

        browser = Browser()
        browser.start()
        browser.navigate("https://example.com")
        browser.close()
    """

    def __init__(self, options: Optional[BrowserOptions] = None) -> None:
        self._options = options or BrowserOptions()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the browser and create an initial page."""
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        self._playwright = sync_playwright().start()
        browser_type = getattr(self._playwright, self._options.browser_type)
        launch_kwargs: dict = {
            "headless": self._options.headless,
            "slow_mo": self._options.slow_mo,
        }
        if self._options.proxy:
            launch_kwargs["proxy"] = {"server": self._options.proxy}

        self._browser = browser_type.launch(**launch_kwargs)

        context_kwargs: dict = {
            "viewport": {
                "width": self._options.viewport_width,
                "height": self._options.viewport_height,
            },
            "ignore_https_errors": self._options.ignore_https_errors,
        }
        if self._options.user_agent:
            context_kwargs["user_agent"] = self._options.user_agent

        self._context = self._browser.new_context(**context_kwargs)
        self._context.set_default_timeout(self._options.timeout)
        self._page = self._context.new_page()

    def close(self) -> None:
        """Close the browser and release all resources."""
        with contextlib.suppress(Exception):
            if self._page:
                self._page.close()
        with contextlib.suppress(Exception):
            if self._context:
                self._context.close()
        with contextlib.suppress(Exception):
            if self._browser:
                self._browser.close()
        with contextlib.suppress(Exception):
            if self._playwright:
                self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    def __enter__(self) -> "Browser":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._page is None:
            raise RuntimeError(
                "Browser is not started. Call start() first or use the context manager."
            )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> NavigateResult:
        """
        Navigate to *url* and return a :class:`NavigateResult`.

        :param url: The URL to navigate to.
        :param wait_until: When to consider navigation finished.
            One of ``"load"``, ``"domcontentloaded"``, ``"networkidle"``,
            ``"commit"``.
        """
        self._ensure_started()
        try:
            response = self._page.goto(url, wait_until=wait_until)
            status = response.status if response else None
            return NavigateResult(
                success=True,
                url=self._page.url,
                title=self._page.title(),
                status=status,
            )
        except Exception as exc:
            return NavigateResult(success=False, error=str(exc), url=url)

    def go_back(self) -> NavigateResult:
        """Navigate to the previous page in browser history."""
        self._ensure_started()
        try:
            self._page.go_back()
            return NavigateResult(
                success=True,
                url=self._page.url,
                title=self._page.title(),
            )
        except Exception as exc:
            return NavigateResult(success=False, error=str(exc))

    def go_forward(self) -> NavigateResult:
        """Navigate to the next page in browser history."""
        self._ensure_started()
        try:
            self._page.go_forward()
            return NavigateResult(
                success=True,
                url=self._page.url,
                title=self._page.title(),
            )
        except Exception as exc:
            return NavigateResult(success=False, error=str(exc))

    def reload(self) -> NavigateResult:
        """Reload the current page."""
        self._ensure_started()
        try:
            response = self._page.reload()
            status = response.status if response else None
            return NavigateResult(
                success=True,
                url=self._page.url,
                title=self._page.title(),
                status=status,
            )
        except Exception as exc:
            return NavigateResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Element interaction
    # ------------------------------------------------------------------

    def click(self, selector: str, *, timeout: Optional[int] = None) -> ActionResult:
        """
        Click on the element matching *selector*.

        :param selector: CSS selector, XPath, or accessible name.
        :param timeout: Override the default timeout (milliseconds).
        """
        self._ensure_started()
        try:
            kwargs: dict = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._page.click(selector, **kwargs)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def type_text(
        self,
        selector: str,
        text: str,
        *,
        clear: bool = True,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """
        Type *text* into the element matching *selector*.

        :param selector: CSS selector for the input element.
        :param text: Text to type.
        :param clear: Whether to clear existing content before typing.
        :param timeout: Override the default timeout (milliseconds).
        """
        self._ensure_started()
        try:
            kwargs: dict = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            if clear:
                self._page.fill(selector, text, **kwargs)
            else:
                self._page.type(selector, text, **kwargs)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def press_key(self, key: str, *, selector: Optional[str] = None) -> ActionResult:
        """
        Press a keyboard key, optionally focused on *selector*.

        :param key: Key name (e.g. ``"Enter"``, ``"Tab"``, ``"Escape"``).
        :param selector: If given, focus the element first.
        """
        self._ensure_started()
        try:
            if selector:
                self._page.press(selector, key)
            else:
                self._page.keyboard.press(key)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def hover(self, selector: str) -> ActionResult:
        """Hover over the element matching *selector*."""
        self._ensure_started()
        try:
            self._page.hover(selector)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def select_option(self, selector: str, value: str) -> ActionResult:
        """
        Select an option in a ``<select>`` element.

        :param selector: CSS selector for the ``<select>`` element.
        :param value: The value (or label) to select.
        """
        self._ensure_started()
        try:
            self._page.select_option(selector, value)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    def scroll(self, x: int = 0, y: int = 500) -> ActionResult:
        """
        Scroll the page by (*x*, *y*) pixels.

        :param x: Horizontal scroll distance in pixels.
        :param y: Vertical scroll distance in pixels.
        """
        self._ensure_started()
        try:
            self._page.evaluate(f"window.scrollBy({x}, {y})")
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def scroll_to_bottom(self) -> ActionResult:
        """Scroll to the bottom of the page."""
        self._ensure_started()
        try:
            self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def scroll_to_top(self) -> ActionResult:
        """Scroll to the top of the page."""
        self._ensure_started()
        try:
            self._page.evaluate("window.scrollTo(0, 0)")
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def scroll_element_into_view(self, selector: str) -> ActionResult:
        """Scroll so that the element matching *selector* is visible."""
        self._ensure_started()
        try:
            self._page.locator(selector).scroll_into_view_if_needed()
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def get_text(self, selector: str = "body") -> TextResult:
        """
        Return the visible text content of the element matching *selector*.

        :param selector: CSS selector (defaults to ``"body"`` for whole page).
        """
        self._ensure_started()
        try:
            text = self._page.inner_text(selector)
            return TextResult(
                success=True,
                text=text,
                url=self._page.url,
                title=self._page.title(),
            )
        except Exception as exc:
            return TextResult(success=False, error=str(exc))

    def get_html(self, selector: str = "html") -> TextResult:
        """
        Return the outer HTML of the element matching *selector*.

        :param selector: CSS selector (defaults to ``"html"`` for full page).
        """
        self._ensure_started()
        try:
            html = self._page.inner_html(selector)
            return TextResult(
                success=True,
                text=html,
                url=self._page.url,
                title=self._page.title(),
            )
        except Exception as exc:
            return TextResult(success=False, error=str(exc))

    def get_url(self) -> str:
        """Return the current page URL."""
        self._ensure_started()
        return self._page.url

    def get_title(self) -> str:
        """Return the current page title."""
        self._ensure_started()
        return self._page.title()

    # ------------------------------------------------------------------
    # Element querying
    # ------------------------------------------------------------------

    def find_elements(self, selector: str) -> ElementsResult:
        """
        Find all elements matching *selector* and return their metadata.

        :param selector: CSS selector.
        """
        self._ensure_started()
        try:
            locators = self._page.locator(selector).all()
            elements: List[Element] = []
            for loc in locators:
                tag = loc.evaluate("el => el.tagName.toLowerCase()")
                text = loc.inner_text() if loc.is_visible() else ""
                href = loc.get_attribute("href")
                el_id = loc.get_attribute("id")
                class_name = loc.get_attribute("class")
                elements.append(
                    Element(
                        tag=tag,
                        text=text.strip(),
                        href=href,
                        id=el_id,
                        class_name=class_name,
                        selector=selector,
                    )
                )
            return ElementsResult(success=True, elements=elements, count=len(elements))
        except Exception as exc:
            return ElementsResult(success=False, error=str(exc))

    def find_links(self) -> ElementsResult:
        """Return all hyperlinks (``<a>`` tags) on the current page."""
        return self.find_elements("a[href]")

    def find_buttons(self) -> ElementsResult:
        """Return all clickable buttons on the current page."""
        return self.find_elements("button, input[type='button'], input[type='submit']")

    def find_inputs(self) -> ElementsResult:
        """Return all text input fields on the current page."""
        return self.find_elements("input[type='text'], input[type='search'], textarea")

    # ------------------------------------------------------------------
    # Screenshots
    # ------------------------------------------------------------------

    def screenshot(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> ScreenshotResult:
        """
        Take a screenshot of the current page or a specific element.

        :param path: File path to save the screenshot. If omitted, the raw
            bytes are returned in :attr:`ScreenshotResult.data`.
        :param full_page: Whether to capture the full scrollable page.
        :param selector: CSS selector of a specific element to screenshot.
        """
        self._ensure_started()
        try:
            kwargs: dict = {"full_page": full_page}
            if path:
                kwargs["path"] = str(path)

            if selector:
                data = self._page.locator(selector).screenshot(**kwargs)
            else:
                data = self._page.screenshot(**kwargs)

            return ScreenshotResult(
                success=True,
                path=str(path) if path else "",
                data=data,
            )
        except Exception as exc:
            return ScreenshotResult(success=False, error=str(exc))

    def screenshot_base64(self, *, full_page: bool = False) -> str:
        """
        Take a screenshot and return it as a base64-encoded string.

        Useful for passing images directly to multimodal AI models.
        """
        result = self.screenshot(full_page=full_page)
        if not result.success or result.data is None:
            raise RuntimeError(f"Screenshot failed: {result.error}")
        return base64.b64encode(result.data).decode()

    # ------------------------------------------------------------------
    # JavaScript execution
    # ------------------------------------------------------------------

    def evaluate(self, script: str) -> ActionResult:
        """
        Execute *script* in the browser context and return the result.

        :param script: JavaScript expression or function body.
        """
        self._ensure_started()
        try:
            value = self._page.evaluate(script)
            return ActionResult(success=True, value=value)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    def wait_for_selector(
        self, selector: str, *, state: str = "visible", timeout: Optional[int] = None
    ) -> ActionResult:
        """
        Wait until *selector* reaches *state*.

        :param selector: CSS selector to wait for.
        :param state: One of ``"attached"``, ``"detached"``, ``"visible"``,
            ``"hidden"``.
        :param timeout: Override the default timeout (milliseconds).
        """
        self._ensure_started()
        try:
            kwargs: dict = {"state": state}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._page.wait_for_selector(selector, **kwargs)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def wait_for_load(self, *, timeout: Optional[int] = None) -> ActionResult:
        """Wait for the page ``load`` event."""
        self._ensure_started()
        try:
            kwargs: dict = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._page.wait_for_load_state("load", **kwargs)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def wait_for_network_idle(self, *, timeout: Optional[int] = None) -> ActionResult:
        """Wait until there are no pending network requests."""
        self._ensure_started()
        try:
            kwargs: dict = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._page.wait_for_load_state("networkidle", **kwargs)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Cookies & storage
    # ------------------------------------------------------------------

    def get_cookies(self) -> List[dict]:
        """Return all cookies for the current context."""
        self._ensure_started()
        return self._context.cookies()

    def set_cookie(self, name: str, value: str, **kwargs) -> ActionResult:
        """
        Add a cookie to the browser context.

        Additional keyword arguments are passed through to Playwright
        (e.g. ``domain``, ``path``, ``expires``).
        """
        self._ensure_started()
        try:
            self._context.add_cookies([{"name": name, "value": value, **kwargs}])
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    def clear_cookies(self) -> ActionResult:
        """Clear all cookies from the current context."""
        self._ensure_started()
        try:
            self._context.clear_cookies()
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))
