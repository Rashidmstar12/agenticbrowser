"""
Core BrowserAgent: Chromium-based agentic browser using Playwright.
Supports navigation, interaction, JS execution, and automatic popup handling.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Selector fallback chains for popular sites.
# Each entry: (primary_selector, [fallback_1, fallback_2, ...])
# resolve_selector() tries the primary selector first, then each fallback in
# order, returning the first one that matches an element on the current page.
# This insulates the agent against CSS churn on well-known sites (e.g. Google
# changed its search box from <input name='q'> to <textarea name='q'> in 2022).
# ---------------------------------------------------------------------------
# Selector prefixes that use Playwright's semantic engines instead of CSS.
# resolve_selector() handles these via page.locator() rather than query_selector(),
# so they survive full DOM rewrites as long as the visible text / ARIA role
# / label / placeholder stays the same.
_SEMANTIC_PREFIXES: tuple[str, ...] = (
    "text=",        # visible text content  e.g. text=Sign in
    "role=",        # ARIA role             e.g. role=button
    "label=",       # <label> text          e.g. label=Email address
    "placeholder=", # input placeholder     e.g. placeholder=Search
    "title=",       # title attribute       e.g. title=Close
    "alt=",         # img alt text          e.g. alt=Company logo
)

_SELECTOR_FALLBACKS: list[tuple[str, list[str]]] = [
    # Google search box – swapped from <input> to <textarea> in late 2022.
    # Primary is the current format; fallback covers old cached/proxied pages.
    (
        "textarea[name='q']",
        ["input[name='q']", "[aria-label='Search']", "[title='Search']"],
    ),
    # Bing search box
    (
        "input#sb_form_q",
        ["input[name='q']", "[aria-label='Enter your search term']", "input[type='search']"],
    ),
    # YouTube search box (id can change between desktop/mobile layouts)
    (
        "input#search",
        ["input[name='search_query']", "[aria-label='Search']", "input[placeholder='Search']"],
    ),
    # Wikipedia search box
    (
        "input#searchInput",
        ["input[name='search']", "[aria-label='Search Wikipedia']", "input[type='search']"],
    ),
]

# CSS selectors that commonly represent popup / overlay elements to auto-dismiss.
_POPUP_SELECTORS = [
    # Cookie consent buttons (accept / agree)
    "button[id*='accept']",
    "button[class*='accept']",
    "button[id*='agree']",
    "button[class*='agree']",
    "button[id*='cookie']",
    "button[class*='cookie']",
    "button[id*='consent']",
    "button[class*='consent']",
    "button[id*='allow']",
    "button[class*='allow']",
    # Generic close / dismiss buttons
    "button[aria-label='Close']",
    "button[aria-label='close']",
    "button[aria-label='Dismiss']",
    "button[aria-label='dismiss']",
    "button[class*='close']",
    "button[id*='close']",
    "[class*='modal-close']",
    "[class*='popup-close']",
    "[class*='dialog-close']",
    # GDPR / newsletter overlays
    "[class*='gdpr'] button",
    "[class*='newsletter'] button[class*='close']",
    "[class*='overlay'] button[class*='close']",
]


class BrowserAgent:
    """
    Chromium-based browser agent powered by Playwright.

    Parameters
    ----------
    headless : bool
        Run the browser without a visible window (default ``True``).
    slow_mo : int
        Milliseconds to slow down each Playwright operation by (useful for
        debugging); default ``0``.
    auto_close_popups : bool
        Automatically dismiss common popup / overlay patterns on every page
        load when ``True`` (default ``True``).
    default_timeout : int
        Default timeout in milliseconds for Playwright operations (default
        ``30000``).
    """

    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 0,
        auto_close_popups: bool = True,
        default_timeout: int = 30_000,
    ) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.auto_close_popups = auto_close_popups
        self.default_timeout = default_timeout

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "BrowserAgent":
        """Launch the Chromium browser and open a new page."""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        self._context.set_default_timeout(self.default_timeout)

        # Auto-handle native JS dialogs (alert / confirm / prompt).
        self._page = self._context.new_page()
        self._page.on("dialog", self._handle_dialog)

        logger.info("BrowserAgent started (headless=%s)", self.headless)
        return self

    def stop(self) -> None:
        """Close the browser and release all resources."""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        logger.info("BrowserAgent stopped")

    # Context manager support
    def __enter__(self) -> "BrowserAgent":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser is not started. Call start() first.")
        return self._page

    def _handle_dialog(self, dialog: Any) -> None:
        """Automatically dismiss native JS dialogs."""
        logger.debug("Auto-dismissing dialog: type=%s message=%s", dialog.type, dialog.message)
        dialog.dismiss()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> dict[str, Any]:
        """
        Navigate to *url*.

        Parameters
        ----------
        url:
            Target URL (e.g. ``"https://example.com"``).
        wait_until:
            Playwright load-state strategy: ``"load"``, ``"domcontentloaded"``,
            or ``"networkidle"``.

        Returns
        -------
        dict
            ``{"url": ..., "title": ...}``
        """
        logger.info("Navigating to %s", url)
        self.page.goto(url, wait_until=wait_until)
        if self.auto_close_popups:
            self.close_popups()
        return {"url": self.page.url, "title": self.page.title()}

    # ------------------------------------------------------------------
    # Popup handling
    # ------------------------------------------------------------------

    def close_popups(self) -> dict[str, Any]:
        """
        Attempt to close common popup / overlay elements.

        Tries each selector in ``_POPUP_SELECTORS`` and clicks it if found
        and visible.  Returns a summary of what was dismissed.
        """
        dismissed: list[str] = []
        for selector in _POPUP_SELECTORS:
            try:
                element = self.page.query_selector(selector)
                if element and element.is_visible():
                    element.click(timeout=2_000)
                    dismissed.append(selector)
                    logger.debug("Dismissed popup element: %s", selector)
            except Exception:
                pass  # Element may have disappeared or be unclickable — skip.

        # Also dismiss any visible <dialog> or [role=dialog] overlays by
        # pressing Escape (catches many SPA modals).
        try:
            has_dialog = self.page.evaluate(
                """() => {
                    const d = document.querySelector('dialog[open], [role="dialog"]');
                    return d !== null && d.offsetParent !== null;
                }"""
            )
            if has_dialog:
                self.page.keyboard.press("Escape")
                dismissed.append("Escape (dialog/role=dialog)")
        except Exception:
            pass

        return {"dismissed": dismissed, "count": len(dismissed)}

    # ------------------------------------------------------------------
    # Selector resolution
    # ------------------------------------------------------------------

    def resolve_selector(self, selector: str) -> str:
        """
        Resolve *selector* to one that actually matches an element on the
        current page.

        Tries *selector* first.  If no element is found, tries each fallback
        registered in ``_SELECTOR_FALLBACKS``.  Raises ``ValueError``
        immediately (no 30-second Playwright timeout) when nothing matches,
        with a message listing every selector that was attempted.

        Parameters
        ----------
        selector:
            The primary CSS selector to resolve.

        Returns
        -------
        str
            The first selector (primary or fallback) that matches an element.
        """
        candidates = [selector]
        for primary, fallbacks in _SELECTOR_FALLBACKS:
            if selector == primary:
                candidates.extend(fallbacks)
                break

        for candidate in candidates:
            try:
                el = self.page.query_selector(candidate)
            except Exception:
                continue
            if el is not None:
                if candidate != selector:
                    logger.info(
                        "Selector %r not found; resolved to fallback %r",
                        selector,
                        candidate,
                    )
                return candidate

        tried = ", ".join(repr(c) for c in candidates)
        raise ValueError(f"No element found on page. Tried selectors: {tried}")

    # ------------------------------------------------------------------
    # Element interaction
    # ------------------------------------------------------------------

    def click(self, selector: str, timeout: int | None = None) -> dict[str, Any]:
        """Click the first element matching *selector*."""
        selector = self.resolve_selector(selector)
        logger.info("Clicking '%s'", selector)
        self.page.click(selector, timeout=timeout or self.default_timeout)
        return {"clicked": selector, "url": self.page.url}

    def type_text(
        self,
        selector: str,
        text: str,
        clear_first: bool = True,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Type *text* into the element matching *selector*.

        Parameters
        ----------
        clear_first:
            When ``True`` (default) the field is cleared before typing.
        """
        logger.info("Typing into '%s'", selector)
        selector = self.resolve_selector(selector)
        self.page.click(selector, timeout=timeout or self.default_timeout)
        if clear_first:
            self.page.fill(selector, "", timeout=timeout or self.default_timeout)
        self.page.type(selector, text)
        return {"typed": text, "selector": selector}

    def fill(self, selector: str, value: str, timeout: int | None = None) -> dict[str, Any]:
        """Fill *selector* with *value* (faster than type_text for long text)."""
        selector = self.resolve_selector(selector)
        self.page.fill(selector, value, timeout=timeout or self.default_timeout)
        return {"filled": value, "selector": selector}

    def press_key(self, key: str) -> dict[str, Any]:
        """Press a keyboard key (e.g. ``"Enter"``, ``"Tab"``, ``"Escape"``)."""
        self.page.keyboard.press(key)
        return {"key": key}

    def hover(self, selector: str) -> dict[str, Any]:
        """Move the mouse over *selector*."""
        selector = self.resolve_selector(selector)
        self.page.hover(selector)
        return {"hovered": selector}

    def select_option(self, selector: str, value: str) -> dict[str, Any]:
        """Select an ``<option>`` in a ``<select>`` element by value or label."""
        self.page.select_option(selector, value)
        return {"selected": value, "selector": selector}

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    def scroll(self, x: int = 0, y: int = 500) -> dict[str, Any]:
        """Scroll the page by (*x*, *y*) pixels."""
        self.page.mouse.wheel(x, y)
        return {"scrolled": {"x": x, "y": y}}

    def scroll_to_element(self, selector: str) -> dict[str, Any]:
        """Scroll the element matching *selector* into view."""
        selector = self.resolve_selector(selector)
        el = self.page.query_selector(selector)
        if el is None:
            raise ValueError(f"Element disappeared after selector resolution: {selector!r}")
        el.scroll_into_view_if_needed()
        return {"scrolled_to": selector}

    # ------------------------------------------------------------------
    # Information extraction
    # ------------------------------------------------------------------

    def get_title(self) -> str:
        """Return the current page title."""
        return self.page.title()

    def get_url(self) -> str:
        """Return the current page URL."""
        return self.page.url

    def get_text(self, selector: str = "body") -> str:
        """Return the inner text of the element matching *selector*."""
        return self.page.inner_text(selector)

    def get_html(self, selector: str = "body") -> str:
        """Return the inner HTML of the element matching *selector*."""
        return self.page.inner_html(selector)

    def get_attribute(self, selector: str, attribute: str) -> str | None:
        """Return the value of *attribute* on the element matching *selector*."""
        return self.page.get_attribute(selector, attribute)

    def query_all(self, selector: str) -> list[dict[str, Any]]:
        """
        Return a list of ``{text, href}`` dicts for all elements matching
        *selector*.
        """
        elements = self.page.query_selector_all(selector)
        results = []
        for el in elements:
            results.append(
                {
                    "text": (el.inner_text() or "").strip(),
                    "href": el.get_attribute("href"),
                }
            )
        return results

    # ------------------------------------------------------------------
    # JavaScript execution
    # ------------------------------------------------------------------

    def evaluate(self, script: str) -> Any:
        """Evaluate arbitrary JavaScript *script* in the page context."""
        return self.page.evaluate(script)

    # ------------------------------------------------------------------
    # Screenshots
    # ------------------------------------------------------------------

    def screenshot(
        self,
        path: str | None = None,
        full_page: bool = False,
        as_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Capture a screenshot.

        Parameters
        ----------
        path:
            If given, save the PNG to this file path.
        full_page:
            Capture the full scrollable page when ``True``.
        as_base64:
            Return the image as a base-64 encoded string in the result dict.

        Returns
        -------
        dict
            ``{"path": ..., "base64": ..., "width": ..., "height": ...}``
        """
        kwargs: dict[str, Any] = {"full_page": full_page}
        if path:
            kwargs["path"] = path

        raw = self.page.screenshot(**kwargs)
        result: dict[str, Any] = {"path": path}
        if as_base64 or not path:
            result["base64"] = base64.b64encode(raw).decode()
        return result

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------

    def wait_for_selector(self, selector: str, timeout: int | None = None) -> dict[str, Any]:
        """Wait until the element matching *selector* appears in the DOM."""
        self.page.wait_for_selector(selector, timeout=timeout or self.default_timeout)
        return {"visible": selector}

    def wait_for_navigation(self, url: str | None = None) -> dict[str, Any]:
        """Wait for any navigation to complete (optionally matching *url*)."""
        with self.page.expect_navigation(url=url):
            pass
        return {"url": self.page.url}

    def wait_for_load_state(self, state: str = "networkidle") -> dict[str, Any]:
        """Wait for a specific load state (``"load"``, ``"domcontentloaded"``, ``"networkidle"``)."""
        self.page.wait_for_load_state(state)
        return {"state": state}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_page_info(self) -> dict[str, Any]:
        """Return a summary of the current page state."""
        return {
            "url": self.page.url,
            "title": self.page.title(),
            "viewport": self.page.viewport_size,
        }
