"""
Core BrowserAgent: Chromium-based agentic browser using Playwright.
Supports navigation, interaction, JS execution, and automatic popup handling.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
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

def _generate_dynamic_fallbacks(selector: str) -> list[str]:
    """Derive Playwright semantic fallback selectors from a CSS selector string.

    When all static fallbacks are exhausted, this function inspects common
    attribute patterns in the selector (``aria-label``, ``placeholder``,
    ``data-testid``, ``name``, ``title``) and generates equivalent semantic
    selectors that survive DOM rewrites as long as the visible text / ARIA
    attributes remain stable.  Class-based fallbacks using ``[class*=...]``
    are appended last as a broad catch-all.

    Parameters
    ----------
    selector:
        The original CSS selector that failed to match.

    Returns
    -------
    list[str]
        Zero or more semantic selectors to try (in priority order).
    """
    candidates: list[str] = []

    # [aria-label="X"] → label=X, text=X
    m = re.search(r'\[aria-label=["\']([^"\']+)["\']', selector)
    if m:
        val = m.group(1)
        candidates.append(f"label={val}")
        candidates.append(f"text={val}")

    # [placeholder="X"] → placeholder=X
    m = re.search(r'\[placeholder=["\']([^"\']+)["\']', selector)
    if m:
        candidates.append(f"placeholder={m.group(1)}")

    # input/textarea/select[name="X"] → label=X, placeholder=X
    m = re.search(r'(?:input|textarea|select)\[name=["\']([^"\']+)["\']', selector)
    if m:
        val = m.group(1)
        candidates.append(f"label={val}")
        candidates.append(f"placeholder={val}")

    # [data-testid="X"] → text=X
    m = re.search(r'\[data-testid=["\']([^"\']+)["\']', selector)
    if m:
        candidates.append(f"text={m.group(1)}")

    # [title="X"] → title=X
    m = re.search(r'\[title=["\']([^"\']+)["\']', selector)
    if m:
        candidates.append(f"title={m.group(1)}")

    # .some-class → [class*="some-class"] for the most specific (last) classes
    classes = re.findall(r'\.([a-zA-Z][\w-]*)', selector)
    for cls in classes[-2:]:  # at most 2 classes (most specific suffix)
        candidates.append(f"[class*='{cls}']")

    return candidates


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

# Allowed URL schemes for navigation (prevents javascript:, file:, data: injection).
_ALLOWED_URL_SCHEMES: tuple[str, ...] = ("http://", "https://")


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
        self._intercept_patterns: list[str] = []  # patterns registered via set_network_intercept
        self._recording: bool = False          # True while video recording is active
        self._recording_path: str | None = None  # desired output path for active recording

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
        self._make_context()
        logger.info("BrowserAgent started (headless=%s)", self.headless)
        return self

    def stop(self) -> None:
        """Close the browser and release all resources."""
        if self._recording:
            logger.warning(
                "Video recording was active when stop() was called; "
                "the saved video may be incomplete."
            )
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._pages = []
        self._recording = False
        self._recording_path = None
        logger.info("BrowserAgent stopped")

    # ------------------------------------------------------------------
    # Private context helpers
    # ------------------------------------------------------------------

    def _make_context(self, record_video_dir: str | None = None) -> None:
        """Create (or recreate) the BrowserContext and a fresh first page.

        When *record_video_dir* is given, Playwright will save a WebM video of
        everything that happens inside this context to that directory.
        """
        if self._browser is None:
            raise RuntimeError("Browser is not launched. Call start() first.")
        ctx_kwargs: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 800},
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        }
        if record_video_dir is not None:
            ctx_kwargs["record_video_dir"] = record_video_dir
            ctx_kwargs["record_video_size"] = {"width": 1280, "height": 800}
        self._context = self._browser.new_context(**ctx_kwargs)
        self._context.set_default_timeout(self.default_timeout)
        self._page = self._context.new_page()
        self._page.on("dialog", self._handle_dialog)
        self._pages = [self._page]

    def _close_context(self) -> None:
        """Close all pages and the current context without touching the browser."""
        for page in list(self._pages):
            try:
                page.close()
            except Exception:
                pass
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
        self._page = None
        self._context = None
        self._pages = []

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
            Target URL (e.g. ``"https://example.com"``).  Only ``http://`` and
            ``https://`` schemes are accepted; ``javascript:``, ``file://``, and
            ``data:`` URIs are rejected with a ``ValueError``.
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
        if not url.startswith(_ALLOWED_URL_SCHEMES):
            raise ValueError(
                f"Unsafe URL scheme in {url!r}. "
                "Only http:// and https:// are allowed."
            )
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
        current page.

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
        # Semantic selectors use page.locator(), not query_selector().
        if selector.startswith(_SEMANTIC_PREFIXES):
            try:
                if self.page.locator(selector).count() > 0:
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

        # --- Dynamic fallbacks generated from selector attribute hints ---
        # When all static fallbacks are exhausted, derive semantic alternatives
        # from the CSS selector's own attributes (aria-label, placeholder, …).
        dynamic = _generate_dynamic_fallbacks(selector)
        for candidate in dynamic:
            try:
                if candidate.startswith(_SEMANTIC_PREFIXES):
                    if self.page.locator(candidate).count() > 0:
                        logger.info(
                            "Selector not found; resolved to dynamic fallback (type: %s)",
                            candidate.split("=")[0] if "=" in candidate else "css",
                        )
                        return candidate
                else:
                    el = self.page.query_selector(candidate)
                    if el is not None:
                        logger.info(
                            "Selector not found; resolved to dynamic fallback (type: %s)",
                            candidate.split("=")[0] if "=" in candidate else "css",
                        )
                        return candidate
            except Exception:
                continue

        tried = ", ".join(repr(c) for c in candidates + dynamic)
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
        selector = self.resolve_selector(selector)
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
    # Video recording
    # ------------------------------------------------------------------

    def start_video_recording(self, save_path: str) -> dict[str, Any]:
        """Start recording the browser session as a WebM video.

        Playwright records the browser context as ``.webm`` video.  The file is
        only finalised when :meth:`stop_video_recording` is called.

        Parameters
        ----------
        save_path:
            Absolute path for the output ``.webm`` file (e.g.
            ``/workspace/recordings/session.webm``).  The parent directory is
            created if it does not exist.  Must use only ``http://`` / ``https://``
            URLs when navigating during recording.

        Returns
        -------
        dict
            ``{"recording": True, "save_path": ...}``
        """
        if self._recording:
            raise RuntimeError(
                "Video recording is already active. "
                "Call stop_video_recording() before starting a new one."
            )
        save_dir = str(Path(save_path).parent)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        current_url = self._page.url if self._page else None
        self._close_context()
        self._make_context(record_video_dir=save_dir)

        self._recording = True
        self._recording_path = save_path

        if current_url and current_url.startswith(_ALLOWED_URL_SCHEMES):
            self.navigate(current_url)

        logger.info("Video recording started")
        return {"recording": True, "save_path": save_path}

    def stop_video_recording(self) -> dict[str, Any]:
        """Stop recording and save the finalised WebM video.

        Closes the current browser context (which tells Playwright to finalise
        the video), renames the auto-generated file to the path that was
        supplied to :meth:`start_video_recording`, then reopens a clean
        context.

        Returns
        -------
        dict
            ``{"saved_to": ..., "ok": True}``
        """
        if not self._recording:
            raise RuntimeError(
                "No video recording is in progress. "
                "Call start_video_recording() first."
            )

        # Capture the raw video path BEFORE closing (Playwright sets it on
        # record; the file is finalised only after context.close()).
        raw_video_path: str | None = None
        if self._page is not None and self._page.video is not None:
            try:
                raw_video_path = self._page.video.path()
            except Exception:
                pass

        current_url = self._page.url if self._page else None
        desired_path = self._recording_path

        self._recording = False
        self._recording_path = None

        # Closing the context finalises the video file.
        self._close_context()

        # Move the auto-named file to the caller's desired path.
        saved_to: str | None = None
        if raw_video_path and desired_path and raw_video_path != desired_path:
            import shutil as _shutil
            try:
                _shutil.move(raw_video_path, desired_path)
                saved_to = desired_path
            except Exception as exc:
                logger.warning("Could not rename recorded video file: %s", exc)
                saved_to = raw_video_path
        else:
            saved_to = raw_video_path or desired_path

        # Reinitialise without recording so the agent remains usable.
        self._make_context()
        if current_url and current_url.startswith(_ALLOWED_URL_SCHEMES):
            self.navigate(current_url)

        logger.info("Video recording stopped")
        return {"saved_to": saved_to, "ok": True}

    # ------------------------------------------------------------------
    # GIF recording
    # ------------------------------------------------------------------

    def record_gif(
        self,
        path: str,
        duration: float = 3.0,
        fps: int = 2,
    ) -> dict[str, Any]:
        """Capture an animated GIF of the current page over *duration* seconds.

        Takes ``ceil(duration * fps)`` screenshots at equal intervals and
        stitches them into an animated GIF using Pillow (``pillow>=10.3.0``
        is already listed in ``requirements.txt``).

        Parameters
        ----------
        path:
            Absolute path for the output ``.gif`` file.
        duration:
            Total recording time in seconds (default ``3.0``).
        fps:
            Frames per second (default ``2``; max recommended ``10``).

        Returns
        -------
        dict
            ``{"saved_to": ..., "frames": N, "fps": N, "duration": N, "ok": True}``
        """
        import io
        import time

        from PIL import Image  # type: ignore[import-untyped]

        n_frames = max(1, int(duration * fps))
        interval = duration / n_frames

        frames: list[Any] = []
        for i in range(n_frames):
            raw = self.page.screenshot()
            frames.append(Image.open(io.BytesIO(raw)))
            if i < n_frames - 1:
                time.sleep(interval)

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            dest,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            loop=0,
            duration=int(interval * 1000),
        )
        logger.info("GIF saved (%d frames @ %d fps)", n_frames, fps)
        return {
            "saved_to": str(dest),
            "frames": n_frames,
            "fps": fps,
            "duration": duration,
            "ok": True,
        }

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------

    def wait_for_selector(self, selector: str, timeout: int | None = None) -> dict[str, Any]:
        """Wait until the element matching *selector* appears in the DOM."""
        self.page.wait_for_selector(selector, timeout=timeout or self.default_timeout)
        return {"visible": selector}

    def wait_for_navigation(self, url: str | None = None) -> dict[str, Any]:
        """
        Wait for the current page to reach a settled load state.

        This method waits for ``domcontentloaded`` (falling back gracefully if
        the page is already loaded) so it is safe to call after an action that
        triggers navigation has already been started.

        .. note::
            To wait for navigation that is *about to be triggered* by a click or
            form submission, use ``wait_for_load_state()`` immediately after the
            triggering action instead — Playwright's ``expect_navigation``
            context manager requires the triggering action to occur *inside* the
            ``with`` block, which is not possible from this convenience wrapper.

        Parameters
        ----------
        url:
            Ignored (kept for backwards-compatibility).  Use ``assert_url()``
            after this call if you need to verify the destination URL.

        Returns
        -------
        dict
            ``{"url": <current_url>}``
        """
        try:
            self.page.wait_for_load_state("domcontentloaded")
        except Exception:
            pass  # already loaded — not an error
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
    # High-priority advanced interactions
    # ------------------------------------------------------------------

    def upload_file(self, selector: str, path: str) -> dict[str, Any]:
        """
        Set file(s) on a file ``<input type="file">`` element.

        Parameters
        ----------
        selector:
            CSS selector targeting the file input element.
        path:
            Absolute path to the file to upload.  Multiple files can be
            supplied as a pipe-separated string (``"a.txt|b.txt"``).

        Returns
        -------
        dict
            ``{"selector": ..., "uploaded": ..., "ok": True}``
        """
        files: Any = path.split("|") if "|" in path else path
        self.page.locator(selector).set_input_files(files)
        return {"selector": selector, "uploaded": path, "ok": True}

    def download_file(self, url: str, save_path: str) -> dict[str, Any]:
        """
        Navigate to *url* and save the triggered download to *save_path*.

        This relies on the server responding with a ``Content-Disposition:
        attachment`` header (or an equivalent download-triggering mechanism).
        The browser navigates to *url* inside a ``expect_download`` context so
        that the file is captured before the page changes.

        Parameters
        ----------
        url:
            Direct download URL.
        save_path:
            Absolute path where the downloaded file will be saved.

        Returns
        -------
        dict
            ``{"saved_to": ..., "filename": ..., "ok": True}``
        """
        if not url.startswith(_ALLOWED_URL_SCHEMES):
            raise ValueError(
                f"Unsafe URL scheme in {url!r}. "
                f"Only {_ALLOWED_URL_SCHEMES} are allowed."
            )
        with self.page.expect_download() as download_info:
            self.page.goto(url, wait_until="domcontentloaded")
        download = download_info.value
        download.save_as(save_path)
        return {"saved_to": save_path, "filename": download.suggested_filename, "ok": True}

    def drag_and_drop(self, source_selector: str, target_selector: str) -> dict[str, Any]:
        """
        Drag the element matching *source_selector* and drop it on
        *target_selector*.

        Parameters
        ----------
        source_selector:
            CSS selector of the element to drag.
        target_selector:
            CSS selector of the drop target.

        Returns
        -------
        dict
            ``{"source": ..., "target": ..., "ok": True}``
        """
        self.page.drag_and_drop(source_selector, target_selector)
        return {"source": source_selector, "target": target_selector, "ok": True}

    def right_click(self, selector: str) -> dict[str, Any]:
        """
        Right-click (secondary/context-menu click) on the element matching
        *selector*.

        Parameters
        ----------
        selector:
            CSS selector of the target element.

        Returns
        -------
        dict
            ``{"selector": ..., "ok": True}``
        """
        self.page.click(selector, button="right")
        return {"selector": selector, "ok": True}

    def double_click(self, selector: str) -> dict[str, Any]:
        """
        Double-click on the element matching *selector*.

        Parameters
        ----------
        selector:
            CSS selector of the target element.

        Returns
        -------
        dict
            ``{"selector": ..., "ok": True}``
        """
        self.page.dblclick(selector)
        return {"selector": selector, "ok": True}

    def get_element_rect(self, selector: str) -> dict[str, Any]:
        """
        Return the bounding-box (position + dimensions) of the first element
        matching *selector*.

        Parameters
        ----------
        selector:
            CSS selector of the target element.

        Returns
        -------
        dict
            ``{"x": ..., "y": ..., "width": ..., "height": ..., "selector": ...}``

        Raises
        ------
        RuntimeError
            If the element is not found or not currently visible.
        """
        box = self.page.locator(selector).first.bounding_box()
        if box is None:
            raise RuntimeError(
                f"Element not found or not visible: {selector!r}. "
                "Ensure the element is in the viewport."
            )
        return {
            "x":        box["x"],
            "y":        box["y"],
            "width":    box["width"],
            "height":   box["height"],
            "selector": selector,
        }

    def set_network_intercept(self, url_pattern: str, action: str = "abort") -> dict[str, Any]:
        """
        Intercept all future requests whose URL matches *url_pattern*.

        Parameters
        ----------
        url_pattern:
            Glob pattern (e.g. ``"**/*.png"``), URL substring, or full URL.
            Playwright glob syntax: ``*`` matches any characters except ``/``;
            ``**`` matches any characters including ``/``.
        action:
            What to do with matching requests:
            - ``"abort"`` — block the request entirely (default).
            - ``"continue"`` — let the request through unchanged (useful to
              remove a previously set abort rule while leaving routing active).

        Returns
        -------
        dict
            ``{"url_pattern": ..., "action": ..., "ok": True}``
        """
        if action not in ("abort", "continue"):
            raise ValueError(f"action must be 'abort' or 'continue', got {action!r}")

        if action == "abort":
            def _abort_handler(route: Any) -> None:
                route.abort()
            self.page.route(url_pattern, _abort_handler)
        else:
            def _continue_handler(route: Any) -> None:
                route.continue_()
            self.page.route(url_pattern, _continue_handler)

        self._intercept_patterns.append(url_pattern)
        return {"url_pattern": url_pattern, "action": action, "ok": True}

    def clear_network_intercepts(self) -> dict[str, Any]:
        """
        Remove all network intercept routes previously set via
        :meth:`set_network_intercept`.

        Returns
        -------
        dict
            ``{"cleared": N, "ok": True}``
        """
        count = len(self._intercept_patterns)
        try:
            self.page.unroute_all()
        except AttributeError:
            # Playwright < 1.32 does not have unroute_all(); fall back to
            # unrouting each pattern individually.
            for pattern in self._intercept_patterns:
                try:
                    self.page.unroute(pattern)
                except Exception:
                    pass
        self._intercept_patterns.clear()
        return {"cleared": count, "ok": True}

    def set_viewport(self, width: int, height: int) -> dict[str, Any]:
        """
        Resize the browser viewport.

        Parameters
        ----------
        width:
            New viewport width in pixels.
        height:
            New viewport height in pixels.

        Returns
        -------
        dict
            ``{"width": ..., "height": ..., "ok": True}``
        """
        self.page.set_viewport_size({"width": width, "height": height})
        return {"width": width, "height": height, "ok": True}

    def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> dict[str, Any]:
        """
        Override the browser's geolocation with *latitude* / *longitude*.

        The geolocation permission is automatically granted for the current
        origin so that ``navigator.geolocation.getCurrentPosition()`` returns
        the spoofed coordinates immediately.

        Parameters
        ----------
        latitude:
            Latitude in decimal degrees (``-90`` to ``90``).
        longitude:
            Longitude in decimal degrees (``-180`` to ``180``).
        accuracy:
            Accuracy radius in metres (default ``10``).

        Returns
        -------
        dict
            ``{"latitude": ..., "longitude": ..., "accuracy": ..., "ok": True}``
        """
        if not -90 <= latitude <= 90:
            raise ValueError(f"latitude must be in [-90, 90], got {latitude}")
        if not -180 <= longitude <= 180:
            raise ValueError(f"longitude must be in [-180, 180], got {longitude}")
        if self._context is None:
            raise RuntimeError("Browser is not started. Call start() first.")
        self._context.set_geolocation({"latitude": latitude, "longitude": longitude, "accuracy": accuracy})
        self._context.grant_permissions(["geolocation"])
        return {"latitude": latitude, "longitude": longitude, "accuracy": accuracy, "ok": True}
