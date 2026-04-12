"""
agenticbrowser - An agentic browser tool for AI agents to control web browsers.

Basic usage::

    from agenticbrowser import Browser

    with Browser() as browser:
        result = browser.navigate("https://example.com")
        print(result.title)
        content = browser.get_text()
        print(content.text)
"""

from agenticbrowser.browser import Browser
from agenticbrowser.models import (
    ActionResult,
    BrowserOptions,
    Element,
    ElementsResult,
    NavigateResult,
    ScreenshotResult,
    TextResult,
)

__all__ = [
    "Browser",
    "ActionResult",
    "NavigateResult",
    "ScreenshotResult",
    "TextResult",
    "ElementsResult",
    "Element",
    "BrowserOptions",
]

__version__ = "0.1.0"
