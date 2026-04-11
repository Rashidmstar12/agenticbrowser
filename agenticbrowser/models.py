"""Data models for agenticbrowser action results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ActionResult:
    """Base result for any browser action."""

    success: bool
    error: Optional[str] = None
    value: Optional[object] = None

    def __bool__(self) -> bool:
        return self.success


@dataclass
class NavigateResult(ActionResult):
    """Result of a page navigation."""

    url: str = ""
    title: str = ""
    status: Optional[int] = None


@dataclass
class ScreenshotResult(ActionResult):
    """Result of a screenshot action."""

    path: str = ""
    data: Optional[bytes] = None


@dataclass
class TextResult(ActionResult):
    """Result of extracting text from the page."""

    text: str = ""
    url: str = ""
    title: str = ""


@dataclass
class Element:
    """Represents a DOM element found on the page."""

    tag: str
    text: str
    href: Optional[str] = None
    id: Optional[str] = None
    class_name: Optional[str] = None
    selector: Optional[str] = None
    attributes: dict = field(default_factory=dict)


@dataclass
class ElementsResult(ActionResult):
    """Result of querying elements from the page."""

    elements: List[Element] = field(default_factory=list)
    count: int = 0


@dataclass
class BrowserOptions:
    """Configuration options for the Browser."""

    headless: bool = True
    browser_type: str = "chromium"
    timeout: int = 30_000
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: Optional[str] = None
    slow_mo: int = 0
    proxy: Optional[str] = None
    ignore_https_errors: bool = False
