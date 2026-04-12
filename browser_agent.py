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
        self._pages: list[Page] = []  # all open tabs; self._page is the active one
        self._active_frame: Any = None  # set by iframe_switch; None = use active page
        self._intercept_handlers: list[tuple[str, Any]] = []  # (url_pattern, handler)

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
        self._pages = [self._page]  # track all open tabs

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
        self._pages = []
        self._active_frame = None
        self._intercept_handlers = []
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

    @property
    def _frame(self) -> Any:
        """Return the active frame (or the active page if no iframe is active)."""
        if self._active_frame is not None:
            return self._active_frame
        return self.page

    def _handle_dialog(self, dialog: Any) -> None:
        """Automatically dismiss native JS dialogs."""
        logger.debug("Auto-dismissing dialog: type=%s message=%s", dialog.type, dialog.message)
        dialog.dismiss()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _try_networkidle(self, timeout: int = 5_000) -> bool:
        """
        Attempt to wait for the ``networkidle`` load state.

        Unlike :meth:`wait_for_load_state`, this method catches
        ``TimeoutError`` and falls back to ``"load"`` so that SPAs with
        persistent background network activity (WebSockets, analytics, etc.)
        never cause the agent to hang indefinitely.

        Parameters
        ----------
        timeout:
            Milliseconds to wait for networkidle before giving up (default
            5 000 ms — enough for normal pages, short enough to fail fast on
            SPAs).

        Returns
        -------
        bool
            ``True`` if networkidle was reached, ``False`` if the fallback was
            used.
        """
        try:
            self.page.wait_for_load_state("networkidle", timeout=timeout)
            return True
        except Exception:
            logger.debug(
                "networkidle not reached within %d ms; falling back to 'load'", timeout
            )
            try:
                self.page.wait_for_load_state("load", timeout=timeout)
            except Exception:
                pass
            return False

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> dict[str, Any]:
        """
        Navigate to *url*.

        Parameters
        ----------
        url:
            Target URL (e.g. ``"https://example.com"``).
        wait_until:
            Playwright load-state strategy: ``"load"``, ``"domcontentloaded"``,
            or ``"networkidle"``.  When ``"networkidle"`` is requested the page
            is loaded with ``"domcontentloaded"`` first and then a best-effort
            networkidle wait is attempted (see :meth:`_try_networkidle`).  This
            prevents SPAs with background requests from hanging forever.

        Returns
        -------
        dict
            ``{"url": ..., "title": ...}``
        """
        logger.info("Navigating to %s", url)
        # Navigate with a reliable load event; networkidle is attempted separately
        # to avoid infinite hangs on SPAs that never stop making network requests.
        actual_wait = "domcontentloaded" if wait_until == "networkidle" else wait_until
        self.page.goto(url, wait_until=actual_wait)
        if wait_until == "networkidle":
            self._try_networkidle()
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
        current page (or the active iframe frame).

        Semantic selectors (``text=``, ``role=``, ``label=``, ``placeholder=``,
        ``title=``, ``alt=``) are handled via Playwright's locator engine which
        understands ARIA semantics and survives DOM rewrites.  CSS selectors are
        tried first, then each fallback registered in ``_SELECTOR_FALLBACKS``.

        Raises ``ValueError`` immediately (no 30-second Playwright timeout)
        when nothing matches, with a message listing every selector tried.

        Parameters
        ----------
        selector:
            The primary CSS / semantic selector to resolve.

        Returns
        -------
        str
            The first selector (primary or fallback) that matches an element.
        """
        frame = self._frame
        # Semantic selectors use locator(), not query_selector().
        if selector.startswith(_SEMANTIC_PREFIXES):
            try:
                if frame.locator(selector).count() > 0:
                    return selector
            except Exception:
                pass
            raise ValueError(f"No element found for semantic selector: {selector!r}")

        candidates = [selector]
        for primary, fallbacks in _SELECTOR_FALLBACKS:
            if selector == primary:
                candidates.extend(fallbacks)
                break

        for candidate in candidates:
            try:
                el = frame.query_selector(candidate)
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
        self._frame.click(selector, timeout=timeout or self.default_timeout)
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
        self._frame.click(selector, timeout=timeout or self.default_timeout)
        if clear_first:
            self._frame.fill(selector, "", timeout=timeout or self.default_timeout)
        self._frame.type(selector, text)
        return {"typed": text, "selector": selector}

    def fill(self, selector: str, value: str, timeout: int | None = None) -> dict[str, Any]:
        """Fill *selector* with *value* (faster than type_text for long text)."""
        selector = self.resolve_selector(selector)
        self._frame.fill(selector, value, timeout=timeout or self.default_timeout)
        return {"filled": value, "selector": selector}

    def press_key(self, key: str) -> dict[str, Any]:
        """Press a keyboard key (e.g. ``"Enter"``, ``"Tab"``, ``"Escape"``)."""
        self.page.keyboard.press(key)
        return {"key": key}

    def hover(self, selector: str) -> dict[str, Any]:
        """Move the mouse over *selector*."""
        selector = self.resolve_selector(selector)
        self._frame.hover(selector)
        return {"hovered": selector}

    def select_option(self, selector: str, value: str) -> dict[str, Any]:
        """Select an ``<option>`` in a ``<select>`` element by value or label."""
        selector = self.resolve_selector(selector)
        self._frame.select_option(selector, value)
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
        return self._frame.inner_text(selector)

    def get_html(self, selector: str = "body") -> str:
        """Return the inner HTML of the element matching *selector*."""
        return self._frame.inner_html(selector)

    def get_attribute(self, selector: str, attribute: str) -> str | None:
        """Return the value of *attribute* on the element matching *selector*."""
        return self._frame.get_attribute(selector, attribute)

    def query_all(self, selector: str) -> list[dict[str, Any]]:
        """
        Return a list of ``{text, href}`` dicts for all elements matching
        *selector*.
        """
        elements = self._frame.query_selector_all(selector)
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
        """Wait for a specific load state (``"load"``, ``"domcontentloaded"``, ``"networkidle"``).

        When *state* is ``"networkidle"`` a best-effort wait with a short
        timeout is used so that SPAs with persistent background requests do not
        hang the agent.
        """
        if state == "networkidle":
            reached = self._try_networkidle()
            return {"state": "networkidle", "reached": reached}
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

    # ------------------------------------------------------------------
    # Smart extraction
    # ------------------------------------------------------------------

    def extract_links(self, selector: str = "a", limit: int = 100) -> dict[str, Any]:
        """
        Return all hyperlinks on the page matching *selector*.

        Parameters
        ----------
        selector:
            CSS selector used to find anchor (or other) elements (default ``"a"``).
        limit:
            Maximum number of links to return (default ``100``).

        Returns
        -------
        dict
            ``{"links": [{"text": ..., "href": ...}, ...], "count": N}``
        """
        links: list[dict[str, Any]] = self.page.evaluate(
            """
            ([sel, lim]) => {
                const els = Array.from(document.querySelectorAll(sel)).slice(0, lim);
                return els.map(el => ({
                    text: (el.textContent || "").trim(),
                    href: el.href || el.getAttribute("href") || "",
                }));
            }
            """,
            [selector, limit],
        )
        return {"links": links, "count": len(links), "selector": selector}

    def extract_table(self, selector: str = "table", table_index: int = 0) -> dict[str, Any]:
        """
        Extract an HTML ``<table>`` as a list of row dicts keyed by header text.

        The first row is treated as the header row.  If the table has no
        ``<thead>`` the first ``<tr>`` is used instead.

        Parameters
        ----------
        selector:
            CSS selector for the table element (default ``"table"``).
        table_index:
            Zero-based index when the selector matches multiple tables.

        Returns
        -------
        dict
            ``{"rows": [{col: value, ...}, ...], "count": N, "headers": [...]}``
        """
        data: dict[str, Any] | None = self.page.evaluate(
            """
            ([sel, idx]) => {
                const tables = document.querySelectorAll(sel);
                const table = tables[idx];
                if (!table) return null;
                const rows = Array.from(table.rows);
                if (rows.length === 0) return {headers: [], rows: [], count: 0};
                const headers = Array.from(rows[0].cells).map(c => c.textContent.trim());
                const dataRows = rows.slice(1).map(row => {
                    const cells = Array.from(row.cells).map(c => c.textContent.trim());
                    const obj = {};
                    headers.forEach((h, i) => { obj[h || String(i)] = cells[i] ?? ""; });
                    return obj;
                });
                return {headers, rows: dataRows, count: dataRows.length};
            }
            """,
            [selector, table_index],
        )
        if data is None:
            raise ValueError(
                f"No table found for selector {selector!r} at index {table_index}. "
                "Check that the page has a <table> element."
            )
        return data

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def assert_text(
        self,
        text: str,
        selector: str = "body",
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Assert that *text* is present in the text content of *selector*.

        Raises ``AssertionError`` (which causes the step to fail) when the
        text is not found.  This is useful for verification steps in a task
        plan — e.g. confirm a login succeeded before proceeding.

        Returns
        -------
        dict
            ``{"found": True, "text": ..., "selector": ...}``
        """
        content = self.get_text(selector)
        haystack = content if case_sensitive else content.lower()
        needle   = text    if case_sensitive else text.lower()
        if needle not in haystack:
            raise AssertionError(
                f"Expected text {text!r} not found in element {selector!r}."
            )
        return {"found": True, "text": text, "selector": selector}

    def assert_url(self, pattern: str) -> dict[str, Any]:
        """
        Assert that the current URL contains *pattern* as a literal substring.

        Raises ``AssertionError`` when the URL does not contain the pattern.

        Parameters
        ----------
        pattern:
            Literal substring that must be present in the current URL
            (e.g. ``"/dashboard"``, ``"?tab=profile"``).

        Returns
        -------
        dict
            ``{"url": ..., "pattern": ..., "matched": True}``
        """
        current = self.page.url
        if pattern not in current:
            raise AssertionError(
                f"URL {current!r} does not contain {pattern!r}."
            )
        return {"url": current, "pattern": pattern, "matched": True}

    # ------------------------------------------------------------------
    # Wait for dynamic content
    # ------------------------------------------------------------------

    def wait_text(
        self,
        text: str,
        selector: str = "body",
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Wait until *text* appears in the text content of *selector*.

        Useful when content is loaded asynchronously (e.g. after an XHR
        completes) and a CSS-selector-based wait would be too broad.

        Parameters
        ----------
        text:
            The string to wait for.
        selector:
            CSS selector of the container element (default ``"body"``).
        timeout:
            Override the default timeout (milliseconds).

        Returns
        -------
        dict
            ``{"found": True, "text": ..., "selector": ...}``
        """
        timeout_ms = timeout or self.default_timeout
        self.page.wait_for_function(
            "([sel, txt]) => {"
            "  const el = document.querySelector(sel);"
            "  return el && el.textContent.includes(txt);"
            "}",
            arg=[selector, text],
            timeout=timeout_ms,
        )
        return {"found": True, "text": text, "selector": selector}

    # ------------------------------------------------------------------
    # Cookie persistence
    # ------------------------------------------------------------------

    def get_cookies(self) -> list[dict[str, Any]]:
        """Return all cookies for the current browser context."""
        return self._context.cookies()  # type: ignore[union-attr]

    def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
        """Add *cookies* to the current browser context."""
        self._context.add_cookies(cookies)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Multi-tab management
    # ------------------------------------------------------------------

    def new_tab(self, url: str | None = None) -> dict[str, Any]:
        """
        Open a new browser tab and make it the active page.

        Parameters
        ----------
        url:
            Optional URL to navigate to immediately after opening the tab.

        Returns
        -------
        dict
            ``{"tab_index": N, "url": ..., "title": ...}``
        """
        if self._context is None:
            raise RuntimeError("Browser is not started. Call start() first.")
        page = self._context.new_page()
        page.on("dialog", self._handle_dialog)
        self._pages.append(page)
        self._page = page
        index = len(self._pages) - 1
        if url:
            self.navigate(url)
        return {"tab_index": index, "url": self._page.url, "title": self._page.title()}

    def switch_tab(self, index: int) -> dict[str, Any]:
        """
        Switch the active page to the tab at *index*.

        Returns
        -------
        dict
            ``{"tab_index": N, "url": ..., "title": ...}``
        """
        if index < 0 or index >= len(self._pages):
            raise ValueError(
                f"Tab index {index} is out of range. "
                f"Open tabs: 0–{len(self._pages) - 1}."
            )
        self._page = self._pages[index]
        return {"tab_index": index, "url": self._page.url, "title": self._page.title()}

    def close_tab(self, index: int | None = None) -> dict[str, Any]:
        """
        Close a browser tab.

        Parameters
        ----------
        index:
            Tab index to close.  When ``None`` the currently active tab is
            closed.  Raises ``RuntimeError`` if there is only one tab open.

        Returns
        -------
        dict
            ``{"closed_index": N, "remaining_tabs": M}``
        """
        if len(self._pages) <= 1:
            raise RuntimeError(
                "Cannot close the last open tab. "
                "Use stop() to shut down the browser session."
            )
        if index is None:
            index = self._pages.index(self._page)
        if index < 0 or index >= len(self._pages):
            raise ValueError(
                f"Tab index {index} is out of range. "
                f"Open tabs: 0–{len(self._pages) - 1}."
            )
        target = self._pages.pop(index)
        target.close()
        if self._page is target:
            # Switch to the last remaining tab.
            self._page = self._pages[-1]
        return {"closed_index": index, "remaining_tabs": len(self._pages)}

    def list_tabs(self) -> dict[str, Any]:
        """
        Return information about all open tabs.

        Returns
        -------
        dict
            ``{"tabs": [{"index": N, "url": ..., "title": ..., "active": bool}], "count": N}``
        """
        tabs = []
        for i, page in enumerate(self._pages):
            try:
                tabs.append({
                    "index":  i,
                    "url":    page.url,
                    "title":  page.title(),
                    "active": page is self._page,
                })
            except Exception:
                tabs.append({"index": i, "url": "unknown", "title": "unknown", "active": page is self._page})
        return {"tabs": tabs, "count": len(tabs)}

    # ------------------------------------------------------------------
    # New browser interactions (Category 1)
    # ------------------------------------------------------------------

    def drag_drop(self, source: str, target: str) -> dict[str, Any]:
        """
        Drag the element matching *source* and drop it onto *target*.

        Parameters
        ----------
        source:
            CSS selector of the element to drag.
        target:
            CSS selector of the drop destination.

        Returns
        -------
        dict
            ``{"source": ..., "target": ...}``
        """
        source = self.resolve_selector(source)
        target = self.resolve_selector(target)
        logger.info("Dragging '%s' → '%s'", source, target)
        self.page.drag_and_drop(source, target)
        return {"source": source, "target": target}

    def right_click(self, selector: str) -> dict[str, Any]:
        """
        Right-click the element matching *selector* to open its context menu.

        Returns
        -------
        dict
            ``{"right_clicked": selector}``
        """
        selector = self.resolve_selector(selector)
        logger.info("Right-clicking '%s'", selector)
        self._frame.click(selector, button="right")
        return {"right_clicked": selector}

    def double_click(self, selector: str) -> dict[str, Any]:
        """
        Double-click the element matching *selector*.

        Returns
        -------
        dict
            ``{"double_clicked": selector}``
        """
        selector = self.resolve_selector(selector)
        logger.info("Double-clicking '%s'", selector)
        self._frame.dblclick(selector)
        return {"double_clicked": selector}

    def upload_file(self, selector: str, path: str) -> dict[str, Any]:
        """
        Attach a local file to an ``<input type="file">`` element.

        Parameters
        ----------
        selector:
            CSS selector of the file input element.
        path:
            Absolute or workspace-relative path to the file to attach.

        Returns
        -------
        dict
            ``{"uploaded": path, "selector": selector}``
        """
        selector = self.resolve_selector(selector)
        logger.info("Uploading '%s' to '%s'", path, selector)
        self._frame.set_input_files(selector, path)
        return {"uploaded": path, "selector": selector}

    def set_viewport(self, width: int, height: int) -> dict[str, Any]:
        """
        Resize the browser viewport.

        Parameters
        ----------
        width:
            Viewport width in pixels.
        height:
            Viewport height in pixels.

        Returns
        -------
        dict
            ``{"width": ..., "height": ...}``
        """
        logger.info("Setting viewport to %dx%d", width, height)
        self.page.set_viewport_size({"width": width, "height": height})
        return {"width": width, "height": height}

    def block_resource(self, types: list[str] | None = None) -> dict[str, Any]:
        """
        Block requests for specified resource types to speed up page loads.

        Common types: ``"image"``, ``"stylesheet"``, ``"font"``, ``"media"``,
        ``"script"``.  When *types* is empty or omitted, ``["image", "stylesheet",
        "font"]`` is used as the default.

        Parameters
        ----------
        types:
            List of Playwright resource-type strings to abort.

        Returns
        -------
        dict
            ``{"blocked_types": [...]}``

        Notes
        -----
        This installs a Playwright route handler on the active page.  It only
        affects requests made *after* this call.
        """
        if not types:
            types = ["image", "stylesheet", "font"]
        blocked = list(types)
        logger.info("Blocking resource types: %s", blocked)

        def _abort_if_blocked(route: Any) -> None:
            if route.request.resource_type in blocked:
                route.abort()
            else:
                route.continue_()

        self.page.route("**/*", _abort_if_blocked)
        return {"blocked_types": blocked}

    def iframe_switch(self, selector: str) -> dict[str, Any]:
        """
        Switch the active interaction context to the ``<iframe>`` element
        matching *selector*.

        All subsequent browser interactions (click, fill, get_text, etc.) will
        operate within this iframe until :meth:`iframe_exit` is called.

        Parameters
        ----------
        selector:
            CSS selector of the ``<iframe>`` element.

        Returns
        -------
        dict
            ``{"frame_url": ..., "selector": selector}``
        """
        el = self.page.query_selector(selector)
        if el is None:
            raise ValueError(f"No iframe element found for selector: {selector!r}")
        frame = el.content_frame()
        if frame is None:
            raise ValueError(f"Element {selector!r} is not an iframe or has no content frame.")
        self._active_frame = frame
        logger.info("Switched to iframe '%s' (url=%s)", selector, frame.url)
        return {"frame_url": frame.url, "selector": selector}

    def iframe_exit(self) -> dict[str, Any]:
        """
        Return to the top-level page context after a previous :meth:`iframe_switch`.

        Returns
        -------
        dict
            ``{"frame_url": current_page_url}``
        """
        self._active_frame = None
        logger.info("Exited iframe context; now on top-level page")
        return {"frame_url": self.page.url}

    # ------------------------------------------------------------------
    # Data extraction (Category 2)
    # ------------------------------------------------------------------

    def extract_json_ld(self) -> dict[str, Any]:
        """
        Extract all Schema.org JSON-LD metadata blocks from the page.

        Returns
        -------
        dict
            ``{"items": [...], "count": N}``
        """
        items: list[Any] = self.page.evaluate(
            """() => {
                const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                return Array.from(scripts).map(s => {
                    try { return JSON.parse(s.textContent || ""); }
                    catch { return null; }
                }).filter(x => x !== null);
            }"""
        )
        return {"items": items, "count": len(items)}

    def extract_headings(self) -> dict[str, Any]:
        """
        Extract all headings (h1–h6) from the page as a structured outline.

        Returns
        -------
        dict
            ``{"headings": [{"level": N, "text": "..."}, ...], "count": N}``
        """
        headings: list[dict[str, Any]] = self.page.evaluate(
            """() => {
                const els = document.querySelectorAll("h1,h2,h3,h4,h5,h6");
                return Array.from(els).map(el => ({
                    level: parseInt(el.tagName.slice(1), 10),
                    text: (el.textContent || "").trim(),
                }));
            }"""
        )
        return {"headings": headings, "count": len(headings)}

    def extract_images(self, selector: str = "img", limit: int = 100) -> dict[str, Any]:
        """
        Extract all images from the page.

        Parameters
        ----------
        selector:
            CSS selector used to find image elements (default ``"img"``).
        limit:
            Maximum number of images to return (default ``100``).

        Returns
        -------
        dict
            ``{"images": [{"src": ..., "alt": ..., "width": ..., "height": ...}], "count": N}``
        """
        images: list[dict[str, Any]] = self.page.evaluate(
            """([sel, lim]) => {
                const els = Array.from(document.querySelectorAll(sel)).slice(0, lim);
                return els.map(el => ({
                    src:    el.src || el.getAttribute("src") || "",
                    alt:    (el.alt || "").trim(),
                    width:  el.naturalWidth || el.width || null,
                    height: el.naturalHeight || el.height || null,
                }));
            }""",
            [selector, limit],
        )
        return {"images": images, "count": len(images)}

    def extract_form_fields(self, selector: str = "form") -> dict[str, Any]:
        """
        Describe all interactive form fields within the first ``<form>``
        (or element) matching *selector*.

        Returns
        -------
        dict
            ``{"fields": [{"name": ..., "type": ..., "id": ..., "placeholder": ...,
            "value": ..., "required": bool}], "count": N}``
        """
        fields: list[dict[str, Any]] = self.page.evaluate(
            """([sel]) => {
                const form = document.querySelector(sel) || document.body;
                const inputs = form.querySelectorAll("input,select,textarea,button");
                return Array.from(inputs).map(el => ({
                    tag:         el.tagName.toLowerCase(),
                    name:        el.name || "",
                    id:          el.id || "",
                    type:        el.type || el.tagName.toLowerCase(),
                    placeholder: el.placeholder || "",
                    value:       el.value || "",
                    required:    el.required || false,
                }));
            }""",
            [selector],
        )
        return {"fields": fields, "count": len(fields)}

    def extract_meta(self) -> dict[str, Any]:
        """
        Extract ``<meta>`` tag values including title, description, and
        Open Graph / Twitter card tags.

        Returns
        -------
        dict
            ``{"title": ..., "description": ..., "tags": [{name, property, content}], "count": N}``
        """
        result: dict[str, Any] = self.page.evaluate(
            """() => {
                const tags = Array.from(document.querySelectorAll("meta")).map(m => ({
                    name:     m.name || "",
                    property: m.getAttribute("property") || "",
                    content:  m.content || "",
                }));
                const title = document.title || "";
                const desc = (document.querySelector('meta[name="description"]') || {}).content || "";
                return {title, description: desc, tags, count: tags.length};
            }"""
        )
        return result

    # ------------------------------------------------------------------
    # Network control (Category 1 — remaining)
    # ------------------------------------------------------------------

    def download_file(self, url: str, save_path: str) -> dict[str, Any]:
        """
        Navigate to *url* and save the resulting download to *save_path*.

        The method registers a one-shot ``download`` event listener before
        triggering the navigation so that the file is written to disk without
        requiring a manual "Save As" dialog.

        Parameters
        ----------
        url:
            The URL that triggers a file download (e.g. a direct link to a PDF
            or ZIP file).
        save_path:
            Workspace-relative or absolute path where the downloaded file
            should be saved.

        Returns
        -------
        dict
            ``{"url": ..., "save_path": ..., "size_bytes": N}``
        """
        import os as _os
        # Validate that save_path stays within the workspace (cwd) to prevent
        # path traversal attacks.
        workspace_root = _os.path.realpath(_os.getcwd())
        resolved_path  = _os.path.realpath(save_path)
        if not (resolved_path.startswith(workspace_root + _os.sep) or resolved_path == workspace_root):
            raise ValueError(
                f"save_path must be inside the workspace directory ({workspace_root!r})."
            )
        # Rebuild from the trusted base component to ensure no tainted data
        # flows into subsequent file-system calls.
        safe_path = _os.path.join(workspace_root, _os.path.relpath(resolved_path, workspace_root))
        logger.info("Downloading '%s' → '%s'", url, safe_path)
        with self.page.expect_download() as dl_info:
            self.page.evaluate(
                "([u]) => { const a = document.createElement('a'); a.href = u; a.download = ''; document.body.appendChild(a); a.click(); document.body.removeChild(a); }",
                [url],
            )
        download = dl_info.value
        # Capture file size from Playwright's temporary download path (not
        # user-supplied) before moving the file to the requested location.
        _tmp = download.path()
        size = _os.path.getsize(_tmp) if (_tmp and _os.path.isfile(_tmp)) else 0
        download.save_as(safe_path)
        return {"url": url, "save_path": safe_path, "size_bytes": size}

    def emulate_device(self, device_name: str) -> dict[str, Any]:
        """
        Emulate a named device, updating viewport size, user-agent, and
        device scale factor to match the device's real-world profile.

        Supported device names are those recognised by Playwright's built-in
        device descriptor list (e.g. ``"iPhone 14"``, ``"Pixel 7"``,
        ``"iPad Pro 11"``, ``"Galaxy S9+"``, ``"Desktop Chrome"``).

        Parameters
        ----------
        device_name:
            A string matching a Playwright device descriptor name.

        Returns
        -------
        dict
            ``{"device": ..., "viewport": {"width": ..., "height": ...},
              "user_agent": ...}``

        Raises
        ------
        ValueError
            When *device_name* is not found in Playwright's device list.
        """
        if self._playwright is None:
            raise RuntimeError("Browser is not started. Call start() first.")
        devices = self._playwright.devices
        if device_name not in devices:
            available = sorted(devices.keys())
            raise ValueError(
                f"Unknown device {device_name!r}. "
                f"Available: {', '.join(available[:20])}{'...' if len(available) > 20 else ''}"
            )
        descriptor = devices[device_name]
        vp = descriptor.get("viewport", {})
        width  = vp.get("width",  1280)
        height = vp.get("height", 720)
        ua     = descriptor.get("user_agent", "")
        # Apply viewport immediately to the active page.
        self.page.set_viewport_size({"width": width, "height": height})
        logger.info("Emulating device %r (%dx%d)", device_name, width, height)
        return {
            "device":     device_name,
            "viewport":   {"width": width, "height": height},
            "user_agent": ua,
        }

    def intercept_request(
        self,
        url_pattern: str,
        action: str = "block",
    ) -> dict[str, Any]:
        """
        Install a Playwright route handler that intercepts all requests
        matching *url_pattern*.

        Parameters
        ----------
        url_pattern:
            A glob pattern (e.g. ``"**/api/v1/**"``) or exact URL to match.
        action:
            What to do with matched requests:

            * ``"block"`` — abort the request (default).
            * ``"passthrough"`` — allow the request to continue unchanged.

        Returns
        -------
        dict
            ``{"url_pattern": ..., "action": ...}``

        Notes
        -----
        Call :meth:`mock_response` instead when you want to return a fake
        response body rather than simply blocking.
        """
        if action not in ("block", "passthrough"):
            raise ValueError(f"action must be 'block' or 'passthrough', got {action!r}.")

        if action == "block":
            def _handler(route: Any) -> None:
                route.abort()
        else:
            def _handler(route: Any) -> None:
                route.continue_()

        self.page.route(url_pattern, _handler)
        self._intercept_handlers.append((url_pattern, _handler))
        logger.info("Intercept installed: pattern=%r action=%r", url_pattern, action)
        return {"url_pattern": url_pattern, "action": action}

    def mock_response(
        self,
        url_pattern: str,
        body: str = "",
        status: int = 200,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        """
        Install a Playwright route handler that intercepts requests matching
        *url_pattern* and replies with a synthetic HTTP response.

        Parameters
        ----------
        url_pattern:
            A glob pattern or exact URL to intercept.
        body:
            Response body string (default: ``""``).
        status:
            HTTP status code (default: ``200``).
        content_type:
            ``Content-Type`` header value (default: ``"application/json"``).

        Returns
        -------
        dict
            ``{"url_pattern": ..., "status": ..., "content_type": ...}``
        """
        def _mock_handler(route: Any) -> None:
            route.fulfill(
                status=status,
                content_type=content_type,
                body=body,
            )

        self.page.route(url_pattern, _mock_handler)
        self._intercept_handlers.append((url_pattern, _mock_handler))
        logger.info(
            "Mock response installed: pattern=%r status=%d", url_pattern, status
        )
        return {"url_pattern": url_pattern, "status": status, "content_type": content_type}

    # ------------------------------------------------------------------
    # Authentication & Session Management (Category 3)
    # ------------------------------------------------------------------

    def set_extra_headers(self, headers: dict[str, str]) -> dict[str, Any]:
        """
        Inject additional HTTP request headers for all subsequent requests in
        the current browser context.

        Parameters
        ----------
        headers:
            Mapping of header name → value, e.g.
            ``{"Authorization": "Bearer <token>"}``.

        Returns
        -------
        dict
            ``{"headers_set": ["Header-Name", ...]}``
        """
        if self._context is None:
            raise RuntimeError("Browser is not started. Call start() first.")
        self._context.set_extra_http_headers(headers)
        logger.info("Extra HTTP headers set (%d header(s))", len(headers))
        return {"headers_set": list(headers.keys())}

    def http_auth(self, username: str, password: str) -> dict[str, Any]:
        """
        Set HTTP Basic Authentication credentials for all subsequent requests.

        This encodes the credentials as a Base64 ``Authorization: Basic ...``
        header and injects it via :meth:`set_extra_headers`.

        Parameters
        ----------
        username:
            HTTP Basic Auth username.
        password:
            HTTP Basic Auth password.

        Returns
        -------
        dict
            ``{"auth_set": True, "username": ...}``
        """
        import base64 as _b64
        token = _b64.b64encode(f"{username}:{password}".encode()).decode()
        self.set_extra_headers({"Authorization": f"Basic {token}"})
        logger.info("HTTP Basic Auth set")
        return {"auth_set": True, "username": username}

    def local_storage_set(self, key: str, value: str) -> dict[str, Any]:
        """
        Write a key–value pair into the page's ``localStorage``.

        Parameters
        ----------
        key:
            Storage key.
        value:
            String value to store.

        Returns
        -------
        dict
            ``{"key": ..., "value": ...}``
        """
        self.page.evaluate(
            "([k, v]) => localStorage.setItem(k, v)",
            [key, value],
        )
        return {"key": key, "value": value}

    def local_storage_get(self, key: str) -> dict[str, Any]:
        """
        Read a value from the page's ``localStorage``.

        Parameters
        ----------
        key:
            Storage key to read.

        Returns
        -------
        dict
            ``{"key": ..., "value": ...}``  (``value`` is ``None`` if not set)
        """
        value: str | None = self.page.evaluate(
            "([k]) => localStorage.getItem(k)",
            [key],
        )
        return {"key": key, "value": value}

    def session_storage_set(self, key: str, value: str) -> dict[str, Any]:
        """
        Write a key–value pair into the page's ``sessionStorage``.

        Parameters
        ----------
        key:
            Storage key.
        value:
            String value to store.

        Returns
        -------
        dict
            ``{"key": ..., "value": ...}``
        """
        self.page.evaluate(
            "([k, v]) => sessionStorage.setItem(k, v)",
            [key, value],
        )
        return {"key": key, "value": value}

    def session_storage_get(self, key: str) -> dict[str, Any]:
        """
        Read a value from the page's ``sessionStorage``.

        Parameters
        ----------
        key:
            Storage key to read.

        Returns
        -------
        dict
            ``{"key": ..., "value": ...}``  (``value`` is ``None`` if not set)
        """
        value: str | None = self.page.evaluate(
            "([k]) => sessionStorage.getItem(k)",
            [key],
        )
        return {"key": key, "value": value}

    # ------------------------------------------------------------------
    # Assertions & Verification (Category 4)
    # ------------------------------------------------------------------

    def assert_element_count(
        self,
        selector: str,
        count: int,
        *,
        operator: str = "eq",
    ) -> dict[str, Any]:
        """
        Assert the number of elements matching *selector*.

        Parameters
        ----------
        selector:
            CSS selector to count.
        count:
            Expected element count.
        operator:
            Comparison operator: ``"eq"`` (default), ``"gte"``, ``"lte"``,
            ``"gt"``, ``"lt"``.

        Returns
        -------
        dict
            ``{"selector": ..., "expected": ..., "actual": ..., "operator": ...}``

        Raises
        ------
        AssertionError
            When the count does not satisfy the expected condition.
        """
        actual = len(self.page.query_selector_all(selector))
        ops = {
            "eq":  actual == count,
            "gte": actual >= count,
            "lte": actual <= count,
            "gt":  actual >  count,
            "lt":  actual <  count,
        }
        if operator not in ops:
            raise ValueError(f"Unknown operator {operator!r}. Use: eq, gte, lte, gt, lt.")
        if not ops[operator]:
            raise AssertionError(
                f"assert_element_count failed: selector={selector!r} "
                f"expected {operator} {count} but got {actual}."
            )
        return {"selector": selector, "expected": count, "actual": actual, "operator": operator}

    def assert_attribute(
        self,
        selector: str,
        attribute: str,
        value: str,
        *,
        case_sensitive: bool = True,
    ) -> dict[str, Any]:
        """
        Assert that the HTML *attribute* of the element matching *selector*
        equals *value*.

        Parameters
        ----------
        selector:
            CSS selector of the element.
        attribute:
            HTML attribute name (e.g. ``"href"``, ``"class"``, ``"data-id"``).
        value:
            Expected attribute value.
        case_sensitive:
            When ``False``, comparison is case-insensitive.

        Returns
        -------
        dict
            ``{"selector": ..., "attribute": ..., "expected": ..., "actual": ...}``

        Raises
        ------
        AssertionError
            When the attribute value does not match.
        """
        actual = self.page.get_attribute(selector, attribute)
        haystack = actual if case_sensitive else (actual or "").lower()
        needle   = value  if case_sensitive else value.lower()
        if haystack != needle:
            raise AssertionError(
                f"assert_attribute failed: {selector!r}[{attribute}]="
                f"{actual!r} ≠ {value!r}."
            )
        return {"selector": selector, "attribute": attribute, "expected": value, "actual": actual}

    def assert_title(
        self,
        pattern: str,
        *,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Assert that the page title contains *pattern* as a substring.

        Parameters
        ----------
        pattern:
            Literal substring that must appear in the page title.
        case_sensitive:
            When ``False`` (default), comparison is case-insensitive.

        Returns
        -------
        dict
            ``{"title": ..., "pattern": ..., "matched": True}``

        Raises
        ------
        AssertionError
            When the title does not contain the pattern.
        """
        title = self.page.title()
        haystack = title   if case_sensitive else title.lower()
        needle   = pattern if case_sensitive else pattern.lower()
        if needle not in haystack:
            raise AssertionError(
                f"assert_title failed: title={title!r} does not contain {pattern!r}."
            )
        return {"title": title, "pattern": pattern, "matched": True}

    def assert_visible(self, selector: str) -> dict[str, Any]:
        """
        Assert that the element matching *selector* is visible on the page.

        Returns
        -------
        dict
            ``{"selector": ..., "visible": True}``

        Raises
        ------
        AssertionError
            When the element is not found or not visible.
        """
        el = self.page.query_selector(selector)
        if el is None or not el.is_visible():
            raise AssertionError(
                f"assert_visible failed: element {selector!r} is not visible."
            )
        return {"selector": selector, "visible": True}

    def assert_hidden(self, selector: str) -> dict[str, Any]:
        """
        Assert that the element matching *selector* is NOT visible (or absent).

        Returns
        -------
        dict
            ``{"selector": ..., "hidden": True}``

        Raises
        ------
        AssertionError
            When the element is found and visible.
        """
        el = self.page.query_selector(selector)
        if el is not None and el.is_visible():
            raise AssertionError(
                f"assert_hidden failed: element {selector!r} is visible."
            )
        return {"selector": selector, "hidden": True}
