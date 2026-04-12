"""
Local interactive runner for the Agentic Browser.

Usage
-----
Interactive REPL:
    python local_runner.py

Run a JSON task file (steps use "action" key, same format as /task/execute):
    python local_runner.py --task tasks/example.json

Run a natural-language task and exit:
    python local_runner.py --intent "go to google and search python asyncio"

Run a single low-level command and exit:
    python local_runner.py --cmd "navigate https://example.com"

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
from system_tools import SystemTools
from task_planner import TaskPlanner, validate_steps

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ANSI colour helpers — respects NO_COLOR (https://no-color.org) and FORCE_COLOR
def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR", ""):
        return False
    if os.environ.get("FORCE_COLOR", ""):
        return True
    return sys.stdout.isatty()

_USE_COLOR = _color_enabled()
_GREEN  = "\033[32m" if _USE_COLOR else ""
_RED    = "\033[31m" if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_BOLD   = "\033[1m"  if _USE_COLOR else ""
_RESET  = "\033[0m"  if _USE_COLOR else ""


# ---------------------------------------------------------------------------
# Doctor helpers
# ---------------------------------------------------------------------------

def run_doctor(workspace: str = "workspace", fix: bool = False) -> int:
    """
    Run environment health checks, print coloured results, and return an
    exit code (0 = all pass/warn, 1 = at least one failure).
    """
    from doctor import run_checks

    print(f"\n{_BOLD}{'─' * 55}{_RESET}")
    print(f"  🩺  {_BOLD}Agentic Browser — Environment Doctor{_RESET}")
    print(f"{_BOLD}{'─' * 55}{_RESET}")
    print(f"  Mode: {'diagnose + auto-fix' if fix else 'diagnose only  (re-run with --fix to auto-fix)'}\n")

    checks = run_checks(workspace=workspace, fix=fix)
    any_fail = False
    for c in checks:
        if c.status == "ok":
            icon = f"{_GREEN}✓{_RESET}"
        elif c.status == "warn":
            icon = f"{_YELLOW}⚠{_RESET}"
        else:
            icon = f"{_RED}✗{_RESET}"
            any_fail = True
        fixed_tag = f" {_GREEN}[fixed]{_RESET}" if c.fixed else ""
        print(f"  {icon}  {c.name:<28s}  {c.message}{fixed_tag}")

    print(f"\n{_BOLD}{'─' * 55}{_RESET}")
    if any_fail:
        print(f"  {_RED}Some checks failed.{_RESET}  Re-run with --doctor --fix to auto-fix.")
    else:
        print(f"  {_GREEN}All checks passed.{_RESET}")
    print(f"{_BOLD}{'─' * 55}{_RESET}\n")
    return 1 if any_fail else 0


def _print_failure_report(
    raw_error: str,
    *,
    action: str | None = None,
    step_index: int | None = None,
    intent: str | None = None,
    rerun_cmd: str | None = None,
) -> None:
    """Print a structured failure report with reason, suggestions, and re-run hint."""
    from doctor import explain_failure

    report = explain_failure(
        raw_error,
        action=action,
        step_index=step_index,
        intent=intent,
        rerun_cmd=rerun_cmd,
    )

    print(f"\n  {_RED}● Failure Report{_RESET}")
    print(f"  Reason : {report.reason}")
    if raw_error and raw_error not in report.reason:
        print(f"  Error  : {raw_error[:300]}")
    print(f"\n  {_YELLOW}Troubleshoot:{_RESET}")
    for tip in report.suggestions:
        print(f"    • {tip}")
    if report.rerun_hint:
        print(f"\n  {_GREEN}Re-run  :{_RESET}  {report.rerun_hint}")
    print()


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

COMMANDS: dict[str, str] = {
    # Browser commands
    "navigate":           "navigate <url> [wait_until=domcontentloaded]",
    "click":              "click <selector>",
    "type":               "type <selector> <text>",
    "fill":               "fill <selector> <value>",
    "press":              "press <key>",
    "hover":              "hover <selector>",
    "select_option":      "select_option <selector> <value>",
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
    # New browser interactions
    "drag_drop":          "drag_drop <source_selector> <target_selector>",
    "right_click":        "right_click <selector>",
    "double_click":       "double_click <selector>",
    "upload_file":        "upload_file <selector> <path>",
    "set_viewport":       "set_viewport <width> <height>",
    "block_resource":     "block_resource [image,stylesheet,font]",
    "iframe_switch":      "iframe_switch <selector>",
    "iframe_exit":        "iframe_exit",
    "download_file":      "download_file <url> <save_path>",
    "emulate_device":     "emulate_device <device_name>  (e.g. 'iPhone 14', 'Pixel 7')",
    "intercept_request":  "intercept_request <url_pattern> [block|passthrough]",
    "mock_response":      "mock_response <url_pattern> [body={}] [status=200]",
    # Smart extraction / assertion / wait_text
    "extract_links":      "extract_links [selector=a] [limit=100]",
    "extract_table":      "extract_table [selector=table] [table_index=0]",
    "extract_json_ld":    "extract_json_ld",
    "extract_headings":   "extract_headings",
    "extract_images":     "extract_images [selector=img] [limit=100]",
    "extract_form_fields": "extract_form_fields [selector=form]",
    "extract_meta":       "extract_meta",
    "assert_text":        "assert_text <text> [selector=body]",
    "assert_url":         "assert_url <pattern>",
    "assert_title":       "assert_title <pattern>",
    "assert_element_count": "assert_element_count <selector> <count> [operator=eq]",
    "assert_attribute":   "assert_attribute <selector> <attribute> <value>",
    "assert_visible":     "assert_visible <selector>",
    "assert_hidden":      "assert_hidden <selector>",
    "wait_text":          "wait_text <text> [selector=body]",
    # Cookie persistence
    "save_cookies":       "save_cookies <path>",
    "load_cookies":       "load_cookies <path>",
    # Auth & session
    "set_extra_headers":  "set_extra_headers <header>=<value> [<header>=<value> ...]",
    "http_auth":          "http_auth <username> <password>",
    "local_storage_set":  "local_storage_set <key> <value>",
    "local_storage_get":  "local_storage_get <key>",
    "session_storage_set": "session_storage_set <key> <value>",
    "session_storage_get": "session_storage_get <key>",
    # Multi-tab
    "new_tab":            "new_tab [url]",
    "switch_tab":         "switch_tab <index>",
    "close_tab":          "close_tab [index]",
    "list_tabs":          "list_tabs",
    # Task planner commands
    "task":               "task <intent>  — plan + execute a natural-language task",
    "task_plan":          "task_plan <intent>  — preview the plan without executing",
    # System tool commands
    "write_file":         "write_file <path> <content>",
    "append_file":        "append_file <path> <content>",
    "read_file":          "read_file <path>",
    "list_dir":           "list_dir [path=.]",
    "run_python":         "run_python <code>",
    "run_shell":          "run_shell <command>",
    # Meta
    "help":               "help",
    "rerun":              "rerun  — repeat the last command",
    "skill":              "skill list | skill load <source> | skill info <name>",
    "doctor":             "doctor [fix]  — run environment health checks",
    "quit":               "quit",
}


def _print_help() -> None:
    print("\nAvailable commands:")
    for cmd, usage in COMMANDS.items():
        print(f"  {usage}")
    print()


def _dispatch(
    agent: BrowserAgent,
    line: str,
    *,
    planner: TaskPlanner | None = None,
    tools: SystemTools | None = None,
) -> Any:
    """Parse *line* and execute the corresponding action."""
    parts = line.strip().split(None, 2)
    if not parts:
        return None
    cmd = parts[0].lower()

    try:
        # ---- Browser commands ----

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

        elif cmd == "select_option":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            value    = parts[2] if len(parts) > 2 else input("  Value: ").strip()
            return agent.select_option(selector, value)

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

        # ---- New browser interactions ----

        elif cmd == "drag_drop":
            source = parts[1] if len(parts) > 1 else input("  Source selector: ").strip()
            target = parts[2] if len(parts) > 2 else input("  Target selector: ").strip()
            return agent.drag_drop(source, target)

        elif cmd == "right_click":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            return agent.right_click(selector)

        elif cmd == "double_click":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            return agent.double_click(selector)

        elif cmd == "upload_file":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            path     = parts[2] if len(parts) > 2 else input("  File path: ").strip()
            return agent.upload_file(selector, path)

        elif cmd == "set_viewport":
            width  = int(parts[1]) if len(parts) > 1 else int(input("  Width: ").strip())
            height = int(parts[2]) if len(parts) > 2 else int(input("  Height: ").strip())
            result = agent.set_viewport(width, height)
            print(f"  Viewport set to {width}x{height}")
            return result

        elif cmd == "block_resource":
            types_raw = parts[1] if len(parts) > 1 else "image,stylesheet,font"
            types = [t.strip() for t in types_raw.split(",") if t.strip()]
            result = agent.block_resource(types=types)
            print(f"  Blocking resource types: {result['blocked_types']}")
            return result

        elif cmd == "iframe_switch":
            selector = parts[1] if len(parts) > 1 else input("  iframe selector: ").strip()
            result = agent.iframe_switch(selector)
            print(f"  Switched to iframe: {result['frame_url']}")
            return result

        elif cmd == "iframe_exit":
            result = agent.iframe_exit()
            print(f"  Exited iframe; now on: {result['frame_url']}")
            return result

        elif cmd == "download_file":
            url       = parts[1] if len(parts) > 1 else input("  URL: ").strip()
            save_path = parts[2] if len(parts) > 2 else input("  Save path: ").strip()
            result    = agent.download_file(url, save_path)
            print(f"  Downloaded {result['size_bytes']} bytes → {result['save_path']}")
            return result

        elif cmd == "emulate_device":
            device_name = " ".join(parts[1:]) if len(parts) > 1 else input("  Device name: ").strip()
            result      = agent.emulate_device(device_name)
            vp = result["viewport"]
            print(f"  Emulating {result['device']!r} ({vp['width']}x{vp['height']})")
            return result

        elif cmd == "intercept_request":
            url_pattern = parts[1] if len(parts) > 1 else input("  URL pattern: ").strip()
            action      = parts[2] if len(parts) > 2 else "block"
            result      = agent.intercept_request(url_pattern, action=action)
            print(f"  Intercept installed: {result['url_pattern']!r} → {result['action']}")
            return result

        elif cmd == "mock_response":
            url_pattern  = parts[1] if len(parts) > 1 else input("  URL pattern: ").strip()
            body         = parts[2] if len(parts) > 2 else input("  Body (JSON string): ").strip()
            status       = int(parts[3]) if len(parts) > 3 else 200
            content_type = parts[4] if len(parts) > 4 else "application/json"
            result       = agent.mock_response(url_pattern, body=body, status=status, content_type=content_type)
            print(f"  Mock installed: {result['url_pattern']!r} → HTTP {result['status']}")
            return result

        # ---- Smart extraction / assertion / wait_text ----

        elif cmd == "extract_links":
            selector = parts[1] if len(parts) > 1 else "a"
            limit    = int(parts[2]) if len(parts) > 2 else 100
            result   = agent.extract_links(selector=selector, limit=limit)
            for lnk in result["links"][:20]:
                print(f"  {lnk['text']!r:40s} → {lnk['href']}")
            if result["count"] > 20:
                print(f"  ... and {result['count'] - 20} more")
            return result

        elif cmd == "extract_table":
            selector    = parts[1] if len(parts) > 1 else "table"
            table_index = int(parts[2]) if len(parts) > 2 else 0
            result      = agent.extract_table(selector=selector, table_index=table_index)
            for row in result["rows"][:10]:
                print(f"  {row}")
            if result["count"] > 10:
                print(f"  ... and {result['count'] - 10} more rows")
            return result

        elif cmd == "extract_json_ld":
            result = agent.extract_json_ld()
            for i, item in enumerate(result["items"][:5]):
                print(f"  [{i}] {json.dumps(item)[:120]}")
            if result["count"] > 5:
                print(f"  ... and {result['count'] - 5} more")
            return result

        elif cmd == "extract_headings":
            result = agent.extract_headings()
            for h in result["headings"][:20]:
                print(f"  {'  ' * (h['level'] - 1)}H{h['level']}: {h['text']}")
            if result["count"] > 20:
                print(f"  ... and {result['count'] - 20} more")
            return result

        elif cmd == "extract_images":
            selector = parts[1] if len(parts) > 1 else "img"
            limit    = int(parts[2]) if len(parts) > 2 else 100
            result   = agent.extract_images(selector=selector, limit=limit)
            for img in result["images"][:10]:
                print(f"  {img['src']!r:60s} alt={img['alt']!r}")
            if result["count"] > 10:
                print(f"  ... and {result['count'] - 10} more")
            return result

        elif cmd == "extract_form_fields":
            selector = parts[1] if len(parts) > 1 else "form"
            result   = agent.extract_form_fields(selector=selector)
            for f in result["fields"][:20]:
                print(f"  [{f['tag']}] name={f['name']!r} type={f['type']!r} placeholder={f['placeholder']!r}")
            if result["count"] > 20:
                print(f"  ... and {result['count'] - 20} more")
            return result

        elif cmd == "extract_meta":
            result = agent.extract_meta()
            print(f"  title: {result.get('title', '')!r}")
            print(f"  description: {result.get('description', '')!r}")
            for tag in result.get("tags", [])[:10]:
                name = tag.get("name") or tag.get("property") or ""
                if name:
                    print(f"  {name}: {tag.get('content', '')!r}")
            return result

        elif cmd == "assert_text":
            text     = parts[1] if len(parts) > 1 else input("  Text: ").strip()
            selector = parts[2] if len(parts) > 2 else "body"
            return agent.assert_text(text, selector=selector)

        elif cmd == "assert_url":
            pattern = parts[1] if len(parts) > 1 else input("  Pattern: ").strip()
            return agent.assert_url(pattern)

        elif cmd == "assert_title":
            pattern = parts[1] if len(parts) > 1 else input("  Pattern: ").strip()
            return agent.assert_title(pattern)

        elif cmd == "assert_element_count":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            count    = int(parts[2]) if len(parts) > 2 else int(input("  Count: ").strip())
            operator = parts[3] if len(parts) > 3 else "eq"
            return agent.assert_element_count(selector, count, operator=operator)

        elif cmd == "assert_attribute":
            selector  = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            attribute = parts[2] if len(parts) > 2 else input("  Attribute: ").strip()
            value     = parts[3] if len(parts) > 3 else input("  Value: ").strip()
            return agent.assert_attribute(selector, attribute, value)

        elif cmd == "assert_visible":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            return agent.assert_visible(selector)

        elif cmd == "assert_hidden":
            selector = parts[1] if len(parts) > 1 else input("  Selector: ").strip()
            return agent.assert_hidden(selector)

        elif cmd == "wait_text":
            text     = parts[1] if len(parts) > 1 else input("  Text: ").strip()
            selector = parts[2] if len(parts) > 2 else "body"
            return agent.wait_text(text, selector=selector)

        # ---- Cookie persistence ----

        elif cmd == "save_cookies":
            path = parts[1] if len(parts) > 1 else input("  Path: ").strip()
            cookies = agent.get_cookies()
            if tools is None:
                print("  Error: workspace not configured. Use --workspace.")
                return None
            tools.write_file(path, json.dumps(cookies, indent=2))
            print(f"  Saved {len(cookies)} cookies to {path}")
            return {"cookies_saved": len(cookies), "path": path}

        elif cmd == "load_cookies":
            path = parts[1] if len(parts) > 1 else input("  Path: ").strip()
            if tools is None:
                print("  Error: workspace not configured. Use --workspace.")
                return None
            r = tools.read_file(path)
            cookies = json.loads(r["content"])
            agent.add_cookies(cookies)
            print(f"  Loaded {len(cookies)} cookies from {path}")
            return {"cookies_loaded": len(cookies), "path": path}

        # ---- Auth & session ----

        elif cmd == "set_extra_headers":
            # Accept pairs like  Authorization=Bearer abc  X-Custom=value
            headers: dict[str, str] = {}
            for token in parts[1:]:
                if "=" in token:
                    k, _, v = token.partition("=")
                    headers[k.strip()] = v.strip()
            if not headers:
                raw = input("  Header (Name=Value): ").strip()
                k, _, v = raw.partition("=")
                headers[k.strip()] = v.strip()
            result = agent.set_extra_headers(headers)
            print(f"  Headers set: {result['headers_set']}")
            return result

        elif cmd == "http_auth":
            username = parts[1] if len(parts) > 1 else input("  Username: ").strip()
            password = parts[2] if len(parts) > 2 else input("  Password: ").strip()
            result = agent.http_auth(username, password)
            print(f"  HTTP Basic Auth set for {result['username']!r}")
            return result

        elif cmd == "local_storage_set":
            key   = parts[1] if len(parts) > 1 else input("  Key: ").strip()
            value = parts[2] if len(parts) > 2 else input("  Value: ").strip()
            return agent.local_storage_set(key, value)

        elif cmd == "local_storage_get":
            key    = parts[1] if len(parts) > 1 else input("  Key: ").strip()
            result = agent.local_storage_get(key)
            print(f"  {result['key']!r} = {result['value']!r}")
            return result

        elif cmd == "session_storage_set":
            key   = parts[1] if len(parts) > 1 else input("  Key: ").strip()
            value = parts[2] if len(parts) > 2 else input("  Value: ").strip()
            return agent.session_storage_set(key, value)

        elif cmd == "session_storage_get":
            key    = parts[1] if len(parts) > 1 else input("  Key: ").strip()
            result = agent.session_storage_get(key)
            print(f"  {result['key']!r} = {result['value']!r}")
            return result

        # ---- Multi-tab ----

        elif cmd == "new_tab":
            url    = parts[1] if len(parts) > 1 else None
            result = agent.new_tab(url=url)
            print(f"  Opened tab {result['tab_index']}: {result['url']}")
            return result

        elif cmd == "switch_tab":
            index  = int(parts[1]) if len(parts) > 1 else int(input("  Tab index: ").strip())
            result = agent.switch_tab(index)
            print(f"  Switched to tab {result['tab_index']}: {result['url']}")
            return result

        elif cmd == "close_tab":
            index  = int(parts[1]) if len(parts) > 1 else None
            result = agent.close_tab(index=index)
            print(f"  Closed tab {result['closed_index']}. Remaining: {result['remaining_tabs']}")
            return result

        elif cmd == "list_tabs":
            result = agent.list_tabs()
            for tab in result["tabs"]:
                marker = " ← active" if tab["active"] else ""
                print(f"  [{tab['index']}] {tab['url']!r:50s}  {tab['title']!r}{marker}")
            return result

        # ---- Task planner ----

        elif cmd == "task":
            intent = parts[1] if len(parts) > 1 else input("  Intent: ").strip()
            if len(parts) > 2:
                intent = intent + " " + parts[2]
            if planner is None:
                planner = TaskPlanner()
            summary = planner.run(intent, agent)
            if summary["success"]:
                print(f"  Done — {len(summary['steps'])} steps executed successfully.")
            else:
                failed = [r for r in summary["results"] if r["status"] == "error"]
                print(f"  Task failed ({len(failed)} step(s) errored).")
                for r in failed:
                    err = r.get("error", "unknown error")
                    print(f"    Step {r['step']} ({r['action']}): {err}")
                # Show structured failure report for the first failed step
                if failed:
                    first = failed[0]
                    _print_failure_report(
                        first.get("error", ""),
                        action=first.get("action"),
                        step_index=first.get("step"),
                        intent=intent,
                        rerun_cmd=f'task {intent}',
                    )
            return summary

        elif cmd == "task_plan":
            intent = parts[1] if len(parts) > 1 else input("  Intent: ").strip()
            if len(parts) > 2:
                intent = intent + " " + parts[2]
            if planner is None:
                planner = TaskPlanner()
            steps = planner.plan(intent)
            print(json.dumps(steps, indent=2))
            return {"steps": steps, "count": len(steps)}

        # ---- System tools ----

        elif cmd == "write_file":
            path    = parts[1] if len(parts) > 1 else input("  Path: ").strip()
            content = parts[2] if len(parts) > 2 else input("  Content: ").strip()
            if tools is None:
                print("  Error: workspace not configured. Use --workspace.")
                return None
            result = tools.write_file(path, content)
            print(f"  Written {result['bytes_written']} bytes to {path}")
            return result

        elif cmd == "append_file":
            path    = parts[1] if len(parts) > 1 else input("  Path: ").strip()
            content = parts[2] if len(parts) > 2 else input("  Content: ").strip()
            if tools is None:
                print("  Error: workspace not configured. Use --workspace.")
                return None
            result = tools.append_file(path, content)
            print(f"  Appended {result['bytes_written']} bytes to {path}")
            return result

        elif cmd == "read_file":
            path = parts[1] if len(parts) > 1 else input("  Path: ").strip()
            if tools is None:
                print("  Error: workspace not configured. Use --workspace.")
                return None
            result = tools.read_file(path)
            print(result["content"][:2000] + ("..." if result["size"] > 2000 else ""))
            return result

        elif cmd == "list_dir":
            path = parts[1] if len(parts) > 1 else "."
            if tools is None:
                print("  Error: workspace not configured. Use --workspace.")
                return None
            result = tools.list_dir(path)
            for entry in result["entries"]:
                size = f"{entry['size']} bytes" if entry["size"] is not None else ""
                print(f"  [{entry['type']:4s}] {entry['name']:<30s}  {size}")
            return result

        elif cmd == "run_python":
            code = parts[1] if len(parts) > 1 else input("  Code: ").strip()
            if len(parts) > 2:
                code = code + " " + parts[2]
            if tools is None:
                print("  Error: workspace not configured. Use --workspace.")
                return None
            result = tools.run_python(code)
            if result["stdout"]:
                print(result["stdout"])
            if result["stderr"]:
                print(f"  stderr: {result['stderr']}")
            return result

        elif cmd == "run_shell":
            command = parts[1] if len(parts) > 1 else input("  Command: ").strip()
            if len(parts) > 2:
                command = command + " " + parts[2]
            if tools is None:
                print("  Error: workspace not configured. Use --workspace.")
                return None
            result = tools.run_shell(command)
            if result["stdout"]:
                print(result["stdout"])
            if result["stderr"]:
                print(f"  stderr: {result['stderr']}")
            return result

        # ---- Meta ----

        elif cmd in ("help", "?"):
            _print_help()
            return None

        elif cmd == "doctor":
            fix = len(parts) > 1 and parts[1] == "fix"
            run_doctor(fix=fix)
            return None

        elif cmd == "skill":
            sub = parts[1] if len(parts) > 1 else "list"
            from skills import SkillLoadError, get_default_registry
            reg = get_default_registry()
            if sub == "list":
                skills = reg.list_skills()
                if not skills:
                    print("  No skills loaded. Use:  skill load <source>")
                else:
                    print(f"  {len(skills)} skill(s) loaded:")
                    for s in skills:
                        print(f"    • {s.name:<25s} v{s.version}  {s.description[:60]}")
            elif sub == "load":
                source = " ".join(parts[2:]) if len(parts) > 2 else input("  Source (file/url/gh:owner/repo): ").strip()
                try:
                    loaded = reg.load_from_source(source)
                    print(f"  Loaded {len(loaded)} skill(s): {', '.join(s.name for s in loaded)}")
                except (SkillLoadError, FileNotFoundError, OSError) as exc:
                    print(f"  Error loading skills: {exc}")
            elif sub == "info":
                name = parts[2] if len(parts) > 2 else input("  Skill name: ").strip()
                skill = reg.get(name)
                if skill is None:
                    print(f"  Skill {name!r} not found. Use 'skill list' to see loaded skills.")
                else:
                    print(json.dumps(skill.to_dict(), indent=2))
            elif sub == "unload":
                name = parts[2] if len(parts) > 2 else input("  Skill name: ").strip()
                removed = reg.unregister(name)
                print(f"  {'Unloaded' if removed else 'Skill not found:'} {name!r}")
            else:
                print("  Usage:  skill list | skill load <source> | skill info <name> | skill unload <name>")
            return None

        elif cmd in ("quit", "exit", "q"):
            return "QUIT"

        else:
            print(f"  Unknown command: {cmd!r}. Type 'help' for a list of commands.")
            return None

    except Exception as exc:
        print(f"  Error: {exc}")
        logger.debug("Command error", exc_info=True)
        _print_failure_report(str(exc), action=cmd if "cmd" in dir() else None)
        return None


# ---------------------------------------------------------------------------
# Task file runner  (uses TaskPlanner.execute — supports "action" format,
#                    {{last}} interpolation, schema validation, system actions)
# ---------------------------------------------------------------------------

def run_task_file(
    agent: BrowserAgent,
    path: str,
    *,
    workspace: str | None = None,
) -> None:
    """
    Execute a JSON task file through the TaskPlanner pipeline.

    The file must be a JSON array of step objects, each with an ``"action"``
    key (NOT ``"cmd"``).  Example::

        [
            {"action": "navigate",   "url": "https://example.com"},
            {"action": "close_popups"},
            {"action": "get_text",   "selector": "body"},
            {"action": "write_file", "path": "out.txt", "content": "{{last}}"}
        ]
    """
    with open(path) as fh:
        raw_steps: list[dict[str, Any]] = json.load(fh)

    logger.info("Running task file: %s (%d steps)", path, len(raw_steps))

    try:
        steps = validate_steps(raw_steps)
    except Exception as exc:
        logger.error("Task file validation failed: %s", exc)
        print(f"  Validation error: {exc}")
        return

    planner = TaskPlanner()
    if workspace:
        from system_tools import SystemTools
        planner._system_tools = SystemTools(workspace=workspace)

    results = planner.execute(steps, agent)
    for r in results:
        status = "✓" if r["status"] == "ok" else "✗"
        logger.info("[%s] Step %d (%s): %s", status, r["step"], r["action"],
                    r.get("result") or r.get("error"))

    failed = [r for r in results if r["status"] == "error"]
    if failed:
        print(f"\n  {len(failed)} step(s) failed:")
        for r in failed:
            print(f"    Step {r['step']} ({r['action']}): {r.get('error', '')}")
        first = failed[0]
        _print_failure_report(
            first.get("error", ""),
            action=first.get("action"),
            step_index=first.get("step"),
            rerun_cmd=f"python local_runner.py --task {path}",
        )
    else:
        print(f"\n  All {len(results)} steps completed successfully.")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_repl(
    agent: BrowserAgent,
    *,
    planner: TaskPlanner | None = None,
    tools: SystemTools | None = None,
) -> None:
    """Start an interactive read-eval-print loop."""
    print("Agentic Browser — interactive mode")
    print("Type 'help' for available commands, 'rerun' to repeat last command, 'quit' to exit.\n")

    last_line: str = ""

    while True:
        try:
            line = input("browser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue

        # Handle rerun before updating last_line
        if line in ("rerun", "r") and last_line:
            print(f"  ↩ Rerunning: {last_line}")
            line = last_line
        elif line not in ("rerun", "r"):
            last_line = line

        result = _dispatch(agent, line, planner=planner, tools=tools)
        if result == "QUIT":
            break
        if result is not None and not isinstance(result, dict):
            print(f"  OK: {result}")
        elif isinstance(result, dict) and "error" not in result:
            pass  # _dispatch already printed task/step results


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
        "--intent",
        metavar="TEXT",
        help="Run a natural-language task and exit (e.g. 'go to google and search X')",
    )
    parser.add_argument(
        "--task",
        metavar="FILE",
        help="Path to a JSON task file to execute",
    )
    parser.add_argument(
        "--cmd",
        metavar="CMD",
        help="Single low-level command to run and exit (e.g. 'navigate https://example.com')",
    )
    parser.add_argument(
        "--workspace",
        metavar="DIR",
        default=os.environ.get("BROWSER_WORKSPACE", "workspace"),
        help="Directory for file I/O operations (default: ./workspace or BROWSER_WORKSPACE env var)",
    )
    parser.add_argument(
        "--proxy",
        metavar="URL",
        default=None,
        help=(
            "Proxy server URL, e.g. 'http://user:pass@host:port' or 'socks5://host:port'. "
            "When omitted no proxy is used."
        ),
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run environment health checks and exit",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues found by --doctor (install missing packages, browsers)",
    )
    parser.add_argument(
        "--skills",
        metavar="SOURCE",
        action="append",
        default=[],
        help=(
            "Load skills from a source before running "
            "(file, directory, URL, or gh:owner/repo). "
            "Can be specified multiple times."
        ),
    )
    args = parser.parse_args()

    # --doctor is standalone — run and exit, no browser needed
    if args.doctor:
        sys.exit(run_doctor(workspace=args.workspace, fix=args.fix))

    # Load any external skills into the default registry
    if args.skills:
        from skills import SkillLoadError, get_default_registry
        reg = get_default_registry()
        for source in args.skills:
            try:
                loaded = reg.load_from_source(source)
                print(f"  Loaded {len(loaded)} skill(s) from {source!r}")
            except (SkillLoadError, FileNotFoundError, OSError) as exc:
                print(f"  Warning: could not load skills from {source!r}: {exc}")

    workspace = args.workspace
    tools   = SystemTools(workspace=workspace)
    planner = TaskPlanner()
    planner._system_tools = tools  # share the same workspace instance

    agent = BrowserAgent(
        headless=not args.no_headless,
        slow_mo=args.slow_mo,
        auto_close_popups=not args.no_auto_popups,
        proxy=args.proxy,
    )

    with agent:
        if args.intent:
            summary = planner.run(args.intent, agent)
            if not summary.get("success"):
                failed = [r for r in summary.get("results", []) if r.get("status") == "error"]
                if failed:
                    first = failed[0]
                    _print_failure_report(
                        first.get("error", ""),
                        action=first.get("action"),
                        step_index=first.get("step"),
                        intent=args.intent,
                        rerun_cmd=f'python {sys.argv[0]} --intent "{args.intent}"',
                    )
            print(json.dumps(summary, indent=2, default=str))
        elif args.task:
            run_task_file(agent, args.task, workspace=workspace)
        elif args.cmd:
            result = _dispatch(agent, args.cmd, planner=planner, tools=tools)
            if result and result != "QUIT":
                print(result)
        else:
            run_repl(agent, planner=planner, tools=tools)


if __name__ == "__main__":
    main()
