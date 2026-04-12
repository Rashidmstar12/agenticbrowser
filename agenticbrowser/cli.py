"""Command-line interface for agenticbrowser."""

from __future__ import annotations

import json
import sys

import click

from agenticbrowser import Browser, BrowserOptions
from agenticbrowser.agent import AgentBrowser


@click.group()
@click.version_option(package_name="agenticbrowser")
def main() -> None:
    """agenticbrowser – drive a web browser from the command line."""


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_browser_options = [
    click.option("--browser", default="chromium", show_default=True,
                 type=click.Choice(["chromium", "firefox", "webkit"]),
                 help="Browser engine to use."),
    click.option("--headless/--no-headless", default=True, show_default=True,
                 help="Run in headless mode."),
    click.option("--timeout", default=30000, show_default=True,
                 help="Default timeout in milliseconds."),
]


def _add_options(options):
    def decorator(func):
        for option in reversed(options):
            func = option(func)
        return func
    return decorator


def _make_options(browser: str, headless: bool, timeout: int) -> BrowserOptions:
    return BrowserOptions(
        browser_type=browser,
        headless=headless,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# navigate
# ---------------------------------------------------------------------------

@main.command()
@click.argument("url")
@_add_options(_browser_options)
@click.option("--screenshot", "screenshot_path", default=None,
              help="Save a screenshot to this file after navigating.")
@click.option("--text", "show_text", is_flag=True,
              help="Print the page text after navigating.")
def navigate(url: str, browser: str, headless: bool, timeout: int,
             screenshot_path: str, show_text: bool) -> None:
    """Navigate to URL and optionally capture text or a screenshot."""
    opts = _make_options(browser, headless, timeout)
    with Browser(opts) as b:
        result = b.navigate(url)
        if not result.success:
            click.echo(f"Error: {result.error}", err=True)
            sys.exit(1)
        click.echo(f"Title : {result.title}")
        click.echo(f"URL   : {result.url}")
        click.echo(f"Status: {result.status}")
        if show_text:
            text_result = b.get_text()
            if text_result.success:
                click.echo("\n--- Page text ---")
                click.echo(text_result.text)
        if screenshot_path:
            sr = b.screenshot(screenshot_path, full_page=True)
            if sr.success:
                click.echo(f"Screenshot saved to {screenshot_path}")
            else:
                click.echo(f"Screenshot failed: {sr.error}", err=True)


# ---------------------------------------------------------------------------
# screenshot
# ---------------------------------------------------------------------------

@main.command()
@click.argument("url")
@click.argument("output", default="screenshot.png")
@_add_options(_browser_options)
@click.option("--full-page", is_flag=True, help="Capture the full page.")
def screenshot(url: str, output: str, browser: str, headless: bool,
               timeout: int, full_page: bool) -> None:
    """Take a screenshot of URL and save to OUTPUT (default: screenshot.png)."""
    opts = _make_options(browser, headless, timeout)
    with Browser(opts) as b:
        nav = b.navigate(url)
        if not nav.success:
            click.echo(f"Navigation error: {nav.error}", err=True)
            sys.exit(1)
        sr = b.screenshot(output, full_page=full_page)
        if sr.success:
            click.echo(f"Saved screenshot to {output}")
        else:
            click.echo(f"Error: {sr.error}", err=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

@main.command()
@click.argument("url")
@_add_options(_browser_options)
@click.option("--selector", default="body", show_default=True,
              help="CSS selector of the element to extract text from.")
def text(url: str, browser: str, headless: bool, timeout: int,
         selector: str) -> None:
    """Print the visible text of URL to stdout."""
    opts = _make_options(browser, headless, timeout)
    with Browser(opts) as b:
        nav = b.navigate(url)
        if not nav.success:
            click.echo(f"Navigation error: {nav.error}", err=True)
            sys.exit(1)
        result = b.get_text(selector)
        if result.success:
            click.echo(result.text)
        else:
            click.echo(f"Error: {result.error}", err=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# links
# ---------------------------------------------------------------------------

@main.command()
@click.argument("url")
@_add_options(_browser_options)
@click.option("--json", "as_json", is_flag=True,
              help="Output as JSON.")
def links(url: str, browser: str, headless: bool, timeout: int,
          as_json: bool) -> None:
    """List all hyperlinks found on URL."""
    opts = _make_options(browser, headless, timeout)
    with Browser(opts) as b:
        nav = b.navigate(url)
        if not nav.success:
            click.echo(f"Navigation error: {nav.error}", err=True)
            sys.exit(1)
        result = b.find_links()
        if not result.success:
            click.echo(f"Error: {result.error}", err=True)
            sys.exit(1)
        if as_json:
            data = [
                {"text": el.text, "href": el.href}
                for el in result.elements
            ]
            click.echo(json.dumps(data, indent=2))
        else:
            for el in result.elements:
                click.echo(f"{el.href}  {el.text}")


# ---------------------------------------------------------------------------
# run (agent action runner)
# ---------------------------------------------------------------------------

@main.command("run")
@click.argument("actions_file", type=click.Path(exists=True))
@_add_options(_browser_options)
def run_actions(actions_file: str, browser: str, headless: bool,
                timeout: int) -> None:
    """
    Execute a sequence of browser actions from a JSON file.

    ACTIONS_FILE must be a JSON file containing a list of action objects, e.g.:

    \b
    [
      {"action": "navigate", "url": "https://example.com"},
      {"action": "get_text"}
    ]
    """
    with open(actions_file) as fh:
        actions = json.load(fh)

    opts = _make_options(browser, headless, timeout)
    with AgentBrowser(opts) as agent:
        for i, action in enumerate(actions, 1):
            click.echo(f"[{i}/{len(actions)}] {action.get('action')}")
            try:
                result = agent.run_action(action)
                click.echo(f"  → {result}")
            except ValueError as exc:
                click.echo(f"  Error: {exc}", err=True)
                sys.exit(1)


if __name__ == "__main__":
    main()
