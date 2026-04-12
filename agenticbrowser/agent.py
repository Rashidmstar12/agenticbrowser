"""
Agent-friendly interface for agenticbrowser.

:class:`AgentBrowser` wraps :class:`~agenticbrowser.browser.Browser` and
provides a single :meth:`run_action` method that accepts a plain dictionary
(action name + parameters) so that language-model agents can drive the
browser without importing individual methods.

Supported action names
----------------------
``navigate``        – Navigate to a URL
``back``            – Go back in history
``forward``         – Go forward in history
``reload``          – Reload the current page
``click``           – Click an element
``type``            – Type text into an element
``press``           – Press a keyboard key
``hover``           – Hover over an element
``select``          – Select an option in a <select>
``scroll``          – Scroll the page
``scroll_bottom``   – Scroll to bottom
``scroll_top``      – Scroll to top
``get_text``        – Get visible text (whole page or element)
``get_html``        – Get HTML source
``get_url``         – Get current URL
``get_title``       – Get current page title
``find``            – Find elements by CSS selector
``find_links``      – Find all links
``find_buttons``    – Find all buttons
``find_inputs``     – Find all inputs
``screenshot``      – Take a screenshot
``screenshot_b64``  – Take a screenshot (base64-encoded)
``wait_for``        – Wait for a selector to be visible
``evaluate``        – Execute arbitrary JavaScript
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agenticbrowser.browser import Browser
from agenticbrowser.models import ActionResult, BrowserOptions


class AgentBrowser:
    """
    A simplified, action-dictionary-driven interface for AI agents.

    Example::

        agent = AgentBrowser()
        agent.start()
        result = agent.run_action({"action": "navigate", "url": "https://example.com"})
        text   = agent.run_action({"action": "get_text"})
        agent.close()
    """

    def __init__(self, options: Optional[BrowserOptions] = None) -> None:
        self._browser = Browser(options)

    def start(self) -> None:
        self._browser.start()

    def close(self) -> None:
        self._browser.close()

    def __enter__(self) -> "AgentBrowser":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------

    def run_action(self, action: Dict[str, Any]) -> Any:
        """
        Execute a browser action described by *action*.

        :param action: A dictionary with at least an ``"action"`` key.
            The remaining keys are parameters specific to that action.
        :returns: An appropriate result object.
        :raises ValueError: If the action name is unknown.
        """
        name = action.get("action", "")
        params = {k: v for k, v in action.items() if k != "action"}

        dispatch = {
            "navigate": self._navigate,
            "back": lambda **_: self._browser.go_back(),
            "forward": lambda **_: self._browser.go_forward(),
            "reload": lambda **_: self._browser.reload(),
            "click": self._click,
            "type": self._type,
            "press": self._press,
            "hover": self._hover,
            "select": self._select,
            "scroll": self._scroll,
            "scroll_bottom": lambda **_: self._browser.scroll_to_bottom(),
            "scroll_top": lambda **_: self._browser.scroll_to_top(),
            "get_text": self._get_text,
            "get_html": self._get_html,
            "get_url": lambda **_: self._browser.get_url(),
            "get_title": lambda **_: self._browser.get_title(),
            "find": self._find,
            "find_links": lambda **_: self._browser.find_links(),
            "find_buttons": lambda **_: self._browser.find_buttons(),
            "find_inputs": lambda **_: self._browser.find_inputs(),
            "screenshot": self._screenshot,
            "screenshot_b64": lambda **_: self._browser.screenshot_base64(),
            "wait_for": self._wait_for,
            "evaluate": self._evaluate,
        }

        handler = dispatch.get(name)
        if handler is None:
            raise ValueError(
                f"Unknown action '{name}'. "
                f"Supported actions: {sorted(dispatch.keys())}"
            )
        return handler(**params)

    # ------------------------------------------------------------------
    # Private helpers – thin wrappers that accept keyword params
    # ------------------------------------------------------------------

    def _navigate(self, url: str, wait_until: str = "domcontentloaded", **_):
        return self._browser.navigate(url, wait_until=wait_until)

    def _click(self, selector: str, timeout: Optional[int] = None, **_):
        return self._browser.click(selector, timeout=timeout)

    def _type(
        self,
        selector: str,
        text: str,
        clear: bool = True,
        timeout: Optional[int] = None,
        **_,
    ):
        return self._browser.type_text(selector, text, clear=clear, timeout=timeout)

    def _press(self, key: str, selector: Optional[str] = None, **_):
        return self._browser.press_key(key, selector=selector)

    def _hover(self, selector: str, **_):
        return self._browser.hover(selector)

    def _select(self, selector: str, value: str, **_):
        return self._browser.select_option(selector, value)

    def _scroll(self, x: int = 0, y: int = 500, **_):
        return self._browser.scroll(x, y)

    def _get_text(self, selector: str = "body", **_):
        return self._browser.get_text(selector)

    def _get_html(self, selector: str = "html", **_):
        return self._browser.get_html(selector)

    def _find(self, selector: str, **_):
        return self._browser.find_elements(selector)

    def _screenshot(self, path: Optional[str] = None, full_page: bool = False, **_):
        return self._browser.screenshot(path=path, full_page=full_page)

    def _wait_for(
        self,
        selector: str,
        state: str = "visible",
        timeout: Optional[int] = None,
        **_,
    ):
        return self._browser.wait_for_selector(selector, state=state, timeout=timeout)

    def _evaluate(self, script: str, **_):
        return self._browser.evaluate(script)
