"""
Local interactive runner for the Agentic Browser.

Usage
-----
Interactive REPL:
    python local_runner.py

Run a JSON task file:
    python local_runner.py --task tasks/example.json

Run a single command:
    python local_runner.py --cmd navigate --url https://example.com

Headed mode (visible browser window):
    python local_runner.py --no-headless
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from browser_agent import BrowserAgent
from task_planner import TaskPlanner, StepValidationError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

COMMANDS: dict[str, str] = {
    "navigate":           "navigate <url> [wait_until=domcontentloaded]",
    "click":              "click <selector>",
    "type":               "type <selector> <text>",
    "fill":               "fill <selector> <value>",
    "press":              "press <key>",
    "hover":              "hover <selector>",
    "scroll":             "scroll [y=500] [x=0]",
    "scroll_to":          "scroll_to <selector>",
    "close_popups":       "close_popups",
    "screenshot":         "screenshot [path=screenshot.png] [full=false]",
    "text":               "text [selector=body]",
    "html":               "html [selector=body]",
    "attr":               "attr <selector> <attribute>",
    "query":              "query <selector>",
    "eval":               "eval <javascript>",
    "info":               "info",
    "wait":               "wait <selector>",
    "wait_state":         "wait_state [networkidle|load|domcontentloaded]",
    "help":               "help",
    "quit":               "quit",
}


def _print_help() -> None:
    print("\nAvailable commands:")
    for cmd, usage in COMMANDS.items():
        print(f"  {usage}")
    print()


def _dispatch(agent: BrowserAgent, line: str) -> Any:
    """Parse *line* and execute the corresponding BrowserAgent method."""
    parts = line.strip().split(None, 2)
    if not parts:
        return None
    cmd = parts[0].lower()

    try:
        if cmd == "navigate":
            url = parts[1] if len(parts) > 1 else input("  URL: ").strip()
            wait_until = parts[2] if len(parts) > 2 else "domcontentloaded"
            return agent.navigate(url, wait_until=wait_until)

        elif cmd == "click":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            return agent.click(selector)

        elif cmd == "type":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            text = parts[2] if len(parts) > 2 else input("  Text: ").strip()
            return agent.type_text(selector, text)

        elif cmd == "fill":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            value = parts[2] if len(parts) > 2 else input("  Value: ").strip()
            return agent.fill(selector, value)

        elif cmd == "press":
            key = parts[1] if len(parts) > 1 else input("  Key: ").strip()
            return agent.press_key(key)

        elif cmd == "hover":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            return agent.hover(selector)

        elif cmd == "scroll":
            y = int(parts[1]) if len(parts) > 1 else 500
            x = int(parts[2]) if len(parts) > 2 else 0
            return agent.scroll(x=x, y=y)

        elif cmd == "scroll_to":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            return agent.scroll_to_element(selector)

        elif cmd == "close_popups":
            return agent.close_popups()

        elif cmd == "screenshot":
            path = parts[1] if len(parts) > 1 else "screenshot.png"
            full = (parts[2].lower() == "true") if len(parts) > 2 else False
            result = agent.screenshot(path=path, full_page=full, as_base64=False)
            print(f"  Saved to: {path}")
            return result

        elif cmd == "text":
            selector = parts[1] if len(parts) > 1 else "body"
            text = agent.get_text(selector)
            print(text[:2000] + ("..." if len(text) > 2000 else ""))
            return {"text_length": len(text)}

        elif cmd == "html":
            selector = parts[1] if len(parts) > 1 else "body"
            html = agent.get_html(selector)
            print(html[:2000] + ("..." if len(html) > 2000 else ""))
            return {"html_length": len(html)}

        elif cmd == "attr":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            attribute = parts[2] if len(parts) > 2 else input("  Attribute: ").strip()
            return {"value": agent.get_attribute(selector, attribute)}

        elif cmd == "query":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            elements = agent.query_all(selector)
            for i, el in enumerate(elements[:20]):
                print(f"  [{i}] text={el['text']!r}  href={el['href']!r}")
            if len(elements) > 20:
                print(f"  ... and {len(elements) - 20} more")
            return {"count": len(elements)}

        elif cmd == "eval":
            script = parts[1] if len(parts) > 1 else input("  JS: ").strip()
            result = agent.evaluate(script)
            print(f"  => {result!r}")
            return {"result": result}

        elif cmd == "info":
            return agent.get_page_info()

        elif cmd == "wait":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            return agent.wait_for_selector(selector)

        elif cmd == "wait_state":
            state = parts[1] if len(parts) > 1 else "networkidle"
            return agent.wait_for_load_state(state)

        elif cmd in ("help", "?"):
            _print_help()
            return None

        elif cmd in ("quit", "exit", "q"):
            return "QUIT"

        else:
            print(f"  Unknown command: {cmd!r}. Type 'help' for a list of commands.")
            return None

    except Exception as exc:
        print(f"  Error: {exc}")
        logger.debug("Command error", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Task file runner
# ---------------------------------------------------------------------------

def run_task_file(agent: BrowserAgent, path: str) -> None:
    """
    Execute a JSON task file.

    The file should be a JSON array of step objects, each with a ``"cmd"``
    key and optional parameters, e.g.::

        [
            {"cmd": "navigate", "url": "https://example.com"},
            {"cmd": "close_popups"},
            {"cmd": "click", "selector": "a.more"},
            {"cmd": "screenshot", "path": "result.png"}
        ]
    """
    with open(path) as fh:
        steps: list[dict[str, Any]] = json.load(fh)

    logger.info("Running task file: %s (%d steps)", path, len(steps))

    for i, step in enumerate(steps):
        cmd = step.pop("cmd", "").lower()
        if not cmd:
            logger.warning("Step %d has no 'cmd' field — skipping", i)
            continue

        # Build a command-line string from the step dict so we can reuse _dispatch.
        if cmd == "navigate":
            line = f"navigate {step.get('url', '')} {step.get('wait_until', 'domcontentloaded')}"
        elif cmd in ("click", "hover", "scroll_to", "wait"):
            line = f"{cmd} {step.get('selector', '')}"
        elif cmd == "type":
            line = f"type {step.get('selector', '')} {step.get('text', '')}"
        elif cmd == "fill":
            line = f"fill {step.get('selector', '')} {step.get('value', '')}"
        elif cmd == "press":
            line = f"press {step.get('key', '')}"
        elif cmd == "scroll":
            line = f"scroll {step.get('y', 500)} {step.get('x', 0)}"
        elif cmd == "screenshot":
            line = f"screenshot {step.get('path', 'screenshot.png')} {step.get('full', False)}"
        elif cmd in ("text", "html"):
            line = f"{cmd} {step.get('selector', 'body')}"
        elif cmd == "attr":
            line = f"attr {step.get('selector', '')} {step.get('attribute', '')}"
        elif cmd == "query":
            line = f"query {step.get('selector', '')}"
        elif cmd == "eval":
            line = f"eval {step.get('script', '')}"
        elif cmd in ("close_popups", "info"):
            line = cmd
        elif cmd == "wait_state":
            line = f"wait_state {step.get('state', 'networkidle')}"
        else:
            logger.warning("Step %d: unknown cmd=%r — skipping", i, cmd)
            continue

        logger.info("Step %d: %s", i, line)
        result = _dispatch(agent, line)
        if result:
            logger.info("         => %s", result)
        if result == "QUIT":
            break


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_repl(agent: BrowserAgent) -> None:
    """Start an interactive read-eval-print loop."""
    print("Agentic Browser — interactive mode")
    print("Type 'help' for available commands, 'quit' to exit.\n")

    while True:
        try:
            line = input("browser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue

        result = _dispatch(agent, line)
        if result == "QUIT":
            break
        if result is not None:
            print(f"  OK: {result}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Agentic Browser runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show the browser window (headed mode)",
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        metavar="MS",
        help="Slow down browser operations by MS milliseconds",
    )
    parser.add_argument(
        "--no-auto-popups",
        action="store_true",
        help="Disable automatic popup dismissal",
    )
    parser.add_argument(
        "--task",
        metavar="FILE",
        help="Path to a JSON task file to execute",
    )
    parser.add_argument(
        "--cmd",
        metavar="CMD",
        help="Single command to run (e.g. 'navigate https://example.com')",
    )
    args = parser.parse_args()

    agent = BrowserAgent(
        headless=not args.no_headless,
        slow_mo=args.slow_mo,
        auto_close_popups=not args.no_auto_popups,
    )

    with agent:
        if args.task:
            run_task_file(agent, args.task)
        elif args.cmd:
            result = _dispatch(agent, args.cmd)
            if result and result != "QUIT":
                print(result)
        else:
            run_repl(agent)


if __name__ == "__main__":
    main()
