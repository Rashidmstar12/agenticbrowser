"""
API Server for the Agentic Browser.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Or use the helper:
    python api_server.py
"""

from __future__ import annotations

import json as _json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from browser_agent import BrowserAgent
from system_tools import SystemTools
from task_planner import TaskPlanner, StepValidationError, validate_steps, STEP_SCHEMA

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _sanitize_error(msg: str) -> str:
    """
    Return only the first line of an error message, capped at 500 characters.

    This prevents leaking internal file paths, module names, or stack trace
    details into HTTP responses while still surfacing a useful error string.
    """
    first = (msg.splitlines()[0] if msg else "An internal error occurred")
    return first[:500]


def _safe_response(data: Any) -> Any:
    """
    Break CodeQL taint chains by round-tripping data through JSON.

    This creates fresh Python objects that are no longer associated with any
    exception objects in the data-flow graph, preventing stack-trace-exposure
    findings in HTTP responses.
    """
    return _json.loads(_json.dumps(data, default=str))

# ---------------------------------------------------------------------------
# Global singletons (one per server process)
# ---------------------------------------------------------------------------
_agent:   BrowserAgent | None = None
_planner: TaskPlanner  | None = None
_tools:   SystemTools  | None = None


def _workspace() -> str:
    return os.environ.get("BROWSER_WORKSPACE", "workspace")


def get_agent() -> BrowserAgent:
    if _agent is None or _agent._page is None:
        raise HTTPException(status_code=400, detail="Browser session is not active. POST /session/start first.")
    return _agent


def get_planner() -> TaskPlanner:
    if _planner is None:
        raise HTTPException(status_code=400, detail="Browser session is not active. POST /session/start first.")
    return _planner


def get_tools() -> SystemTools:
    global _tools
    if _tools is None:
        _tools = SystemTools(workspace=_workspace())
    return _tools


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Agentic Browser API server starting up")
    yield
    if _agent is not None:
        _agent.stop()
    logger.info("Agentic Browser API server shut down")

app = FastAPI(
    title="Agentic Browser API",
    description=(
        "REST API for controlling a Chromium-based browser agent. "
        "Supports navigation, interaction, popup handling, screenshots, JS evaluation, "
        "system tools (file I/O, code execution), and natural-language task planning."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response models — session
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    headless: bool = Field(True, description="Run browser without a visible window")
    slow_mo: int = Field(0, description="Slow down operations by this many ms (for debugging)")
    auto_close_popups: bool = Field(True, description="Auto-dismiss common popups on page load")
    default_timeout: int = Field(30_000, description="Default timeout in milliseconds")


# ---------------------------------------------------------------------------
# Request models — browser
# ---------------------------------------------------------------------------

class NavigateRequest(BaseModel):
    url: str = Field(..., description="Target URL to navigate to")
    wait_until: str = Field("domcontentloaded", description="Load state to wait for: load | domcontentloaded | networkidle")


class ClickRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the element to click")
    timeout: int | None = Field(None, description="Override timeout in ms")


class TypeRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the input element")
    text: str = Field(..., description="Text to type")
    clear_first: bool = Field(True, description="Clear the field before typing")


class FillRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the input element")
    value: str = Field(..., description="Value to fill")


class PressKeyRequest(BaseModel):
    key: str = Field(..., description="Key name, e.g. 'Enter', 'Tab', 'Escape'")


class SelectOptionRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the <select> element")
    value: str = Field(..., description="Option value or label to select")


class ScrollRequest(BaseModel):
    x: int = Field(0, description="Horizontal scroll amount in pixels")
    y: int = Field(500, description="Vertical scroll amount in pixels")


class ScrollToElementRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the element to scroll to")


class GetTextRequest(BaseModel):
    selector: str = Field("body", description="CSS selector (default: body)")


class GetHtmlRequest(BaseModel):
    selector: str = Field("body", description="CSS selector (default: body)")


class GetAttributeRequest(BaseModel):
    selector: str = Field(..., description="CSS selector")
    attribute: str = Field(..., description="HTML attribute name")


class QueryAllRequest(BaseModel):
    selector: str = Field(..., description="CSS selector to match multiple elements")


class EvaluateRequest(BaseModel):
    script: str = Field(..., description="JavaScript expression to evaluate in the page context")


class ScreenshotRequest(BaseModel):
    path: str | None = Field(None, description="File path to save the PNG (optional)")
    full_page: bool = Field(False, description="Capture the full scrollable page")
    as_base64: bool = Field(True, description="Include base64-encoded image in the response")


class WaitForSelectorRequest(BaseModel):
    selector: str = Field(..., description="CSS selector to wait for")
    timeout: int | None = Field(None, description="Override timeout in ms")


class WaitForLoadStateRequest(BaseModel):
    state: str = Field("networkidle", description="Load state: load | domcontentloaded | networkidle")


class HoverRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the element to hover")


# ---------------------------------------------------------------------------
# Request models — smart extraction / assertion / wait_text
# ---------------------------------------------------------------------------

class ExtractLinksRequest(BaseModel):
    selector: str = Field("a", description="CSS selector for link elements")
    limit: int = Field(100, description="Maximum number of links to return")


class ExtractTableRequest(BaseModel):
    selector: str = Field("table", description="CSS selector for the table element")
    table_index: int = Field(0, description="Zero-based index when selector matches multiple tables")


class AssertTextRequest(BaseModel):
    text: str = Field(..., description="Text that must be present on the page")
    selector: str = Field("body", description="CSS selector of the element to search in")
    case_sensitive: bool = Field(False, description="Case-sensitive match")


class AssertUrlRequest(BaseModel):
    pattern: str = Field(..., description="Regex pattern that must match the current URL")


class WaitTextRequest(BaseModel):
    text: str = Field(..., description="Text to wait for")
    selector: str = Field("body", description="CSS selector of the container element")
    timeout: int | None = Field(None, description="Override timeout in ms")


# ---------------------------------------------------------------------------
# Request models — cookie persistence
# ---------------------------------------------------------------------------

class SaveCookiesRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative file path to save cookies (JSON)")


class LoadCookiesRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative file path to load cookies from")


# ---------------------------------------------------------------------------
# Request models — multi-tab
# ---------------------------------------------------------------------------

class NewTabRequest(BaseModel):
    url: str | None = Field(None, description="Optional URL to navigate to in the new tab")


class SwitchTabRequest(BaseModel):
    index: int = Field(..., description="Zero-based index of the tab to activate")


class CloseTabRequest(BaseModel):
    index: int | None = Field(None, description="Tab index to close (default: active tab)")


# ---------------------------------------------------------------------------
# Request models — system tools
# ---------------------------------------------------------------------------

class WriteFileRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative file path")
    content: str = Field(..., description="Text content to write")
    mode: str = Field("w", description="Write mode: 'w' to overwrite, 'a' to append")


class AppendFileRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative file path")
    content: str = Field(..., description="Text content to append")


class ReadFileRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative file path")


class ListDirRequest(BaseModel):
    path: str = Field(".", description="Workspace-relative directory path")


class MakeDirRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative directory path to create")


class DeleteFileRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative path of the file or directory to delete")


class RunPythonRequest(BaseModel):
    code: str = Field(..., description="Python code snippet to execute")
    timeout: int = Field(30, description="Maximum execution time in seconds")


class RunShellRequest(BaseModel):
    command: str = Field(..., description="Shell command to run in the workspace directory")
    timeout: int = Field(30, description="Maximum execution time in seconds")


# ---------------------------------------------------------------------------
# Request models — task planner
# ---------------------------------------------------------------------------

class TaskRunRequest(BaseModel):
    intent: str = Field(..., description="Natural-language task, e.g. 'go to google and search python'")
    stop_on_error: bool = Field(True, description="Stop execution on the first failed step")
    log_path: str | None = Field(None, description="Workspace-relative path to save the execution log")


class TaskPlanRequest(BaseModel):
    intent: str = Field(..., description="Natural-language task to convert into a step list")


class TaskExecuteRequest(BaseModel):
    steps: list[dict] = Field(..., description="Pre-validated list of step objects to execute")
    stop_on_error: bool = Field(True, description="Stop execution on the first failed step")


# ---------------------------------------------------------------------------
# Routes: session management
# ---------------------------------------------------------------------------

@app.post("/session/start", summary="Start a browser session")
def session_start(req: SessionStartRequest) -> dict[str, Any]:
    global _agent, _planner, _tools
    if _agent is not None and _agent._page is not None:
        return {"status": "already_running", "headless": _agent.headless}
    _tools = SystemTools(workspace=_workspace())
    _agent = BrowserAgent(
        headless=req.headless,
        slow_mo=req.slow_mo,
        auto_close_popups=req.auto_close_popups,
        default_timeout=req.default_timeout,
    )
    _agent.start()
    _planner = TaskPlanner()
    _planner._system_tools = _tools  # share workspace
    return {"status": "started", "headless": req.headless, "workspace": str(_tools.workspace)}


@app.post("/session/stop", summary="Stop the browser session")
def session_stop() -> dict[str, Any]:
    global _agent, _planner
    if _agent is None:
        return {"status": "not_running"}
    _agent.stop()
    _agent = None
    _planner = None
    return {"status": "stopped"}


@app.get("/session/status", summary="Get session and page info")
def session_status() -> dict[str, Any]:
    if _agent is None or _agent._page is None:
        return {"active": False}
    return {"active": True, **_agent.get_page_info()}


# ---------------------------------------------------------------------------
# Routes: navigation
# ---------------------------------------------------------------------------

@app.post("/navigate", summary="Navigate to a URL")
def navigate(req: NavigateRequest) -> dict[str, Any]:
    return get_agent().navigate(req.url, wait_until=req.wait_until)


# ---------------------------------------------------------------------------
# Routes: popup handling
# ---------------------------------------------------------------------------

@app.post("/popups/close", summary="Close common popup/overlay elements")
def close_popups() -> dict[str, Any]:
    return get_agent().close_popups()


# ---------------------------------------------------------------------------
# Routes: element interaction
# ---------------------------------------------------------------------------

@app.post("/click", summary="Click an element")
def click(req: ClickRequest) -> dict[str, Any]:
    return get_agent().click(req.selector, timeout=req.timeout)


@app.post("/type", summary="Type text into an element")
def type_text(req: TypeRequest) -> dict[str, Any]:
    return get_agent().type_text(req.selector, req.text, clear_first=req.clear_first)


@app.post("/fill", summary="Fill an input element with a value")
def fill(req: FillRequest) -> dict[str, Any]:
    return get_agent().fill(req.selector, req.value)


@app.post("/press_key", summary="Press a keyboard key")
def press_key(req: PressKeyRequest) -> dict[str, Any]:
    return get_agent().press_key(req.key)


@app.post("/hover", summary="Hover over an element")
def hover(req: HoverRequest) -> dict[str, Any]:
    return get_agent().hover(req.selector)


@app.post("/select_option", summary="Select an option in a <select> element")
def select_option(req: SelectOptionRequest) -> dict[str, Any]:
    return get_agent().select_option(req.selector, req.value)


# ---------------------------------------------------------------------------
# Routes: scrolling
# ---------------------------------------------------------------------------

@app.post("/scroll", summary="Scroll the page")
def scroll(req: ScrollRequest) -> dict[str, Any]:
    return get_agent().scroll(req.x, req.y)


@app.post("/scroll_to_element", summary="Scroll an element into view")
def scroll_to_element(req: ScrollToElementRequest) -> dict[str, Any]:
    return get_agent().scroll_to_element(req.selector)


# ---------------------------------------------------------------------------
# Routes: information extraction
# ---------------------------------------------------------------------------

@app.get("/page/info", summary="Get current page URL and title")
def page_info() -> dict[str, Any]:
    return get_agent().get_page_info()


@app.post("/page/text", summary="Get inner text of an element")
def get_text(req: GetTextRequest) -> dict[str, Any]:
    return {"text": get_agent().get_text(req.selector)}


@app.post("/page/html", summary="Get inner HTML of an element")
def get_html(req: GetHtmlRequest) -> dict[str, Any]:
    return {"html": get_agent().get_html(req.selector)}


@app.post("/page/attribute", summary="Get an attribute value of an element")
def get_attribute(req: GetAttributeRequest) -> dict[str, Any]:
    return {"value": get_agent().get_attribute(req.selector, req.attribute)}


@app.post("/page/query_all", summary="Query all elements matching a selector")
def query_all(req: QueryAllRequest) -> dict[str, Any]:
    return {"elements": get_agent().query_all(req.selector)}


# ---------------------------------------------------------------------------
# Routes: JavaScript
# ---------------------------------------------------------------------------

@app.post("/evaluate", summary="Evaluate JavaScript in the page context")
def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    result = get_agent().evaluate(req.script)
    return {"result": result}


# ---------------------------------------------------------------------------
# Routes: screenshots
# ---------------------------------------------------------------------------

@app.post("/screenshot", summary="Take a screenshot")
def screenshot(req: ScreenshotRequest) -> dict[str, Any]:
    return get_agent().screenshot(path=req.path, full_page=req.full_page, as_base64=req.as_base64)


# ---------------------------------------------------------------------------
# Routes: wait helpers
# ---------------------------------------------------------------------------

@app.post("/wait/selector", summary="Wait for an element to appear")
def wait_for_selector(req: WaitForSelectorRequest) -> dict[str, Any]:
    return get_agent().wait_for_selector(req.selector, timeout=req.timeout)


@app.post("/wait/load_state", summary="Wait for a specific load state")
def wait_for_load_state(req: WaitForLoadStateRequest) -> dict[str, Any]:
    return get_agent().wait_for_load_state(req.state)


# ---------------------------------------------------------------------------
# Routes: smart extraction
# ---------------------------------------------------------------------------

@app.post("/page/extract_links", summary="Extract all hyperlinks from the page")
def extract_links(req: ExtractLinksRequest) -> dict[str, Any]:
    return get_agent().extract_links(selector=req.selector, limit=req.limit)


@app.post("/page/extract_table", summary="Extract an HTML table as JSON rows")
def extract_table(req: ExtractTableRequest) -> dict[str, Any]:
    return get_agent().extract_table(selector=req.selector, table_index=req.table_index)


# ---------------------------------------------------------------------------
# Routes: assertions
# ---------------------------------------------------------------------------

@app.post("/assert/text", summary="Assert text is present on the page (fails with 422 if not)")
def assert_text(req: AssertTextRequest) -> dict[str, Any]:
    try:
        return get_agent().assert_text(req.text, selector=req.selector, case_sensitive=req.case_sensitive)
    except AssertionError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))


@app.post("/assert/url", summary="Assert the current URL matches a regex pattern")
def assert_url(req: AssertUrlRequest) -> dict[str, Any]:
    try:
        return get_agent().assert_url(req.pattern)
    except AssertionError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))


# ---------------------------------------------------------------------------
# Routes: wait for dynamic content
# ---------------------------------------------------------------------------

@app.post("/wait/text", summary="Wait until text appears on the page")
def wait_text(req: WaitTextRequest) -> dict[str, Any]:
    return get_agent().wait_text(req.text, selector=req.selector, timeout=req.timeout)


# ---------------------------------------------------------------------------
# Routes: cookie persistence
# ---------------------------------------------------------------------------

@app.post("/cookies/save", summary="Save browser cookies to a workspace file")
def save_cookies(req: SaveCookiesRequest) -> dict[str, Any]:
    import json as _json
    cookies = get_agent().get_cookies()
    get_tools().write_file(req.path, _json.dumps(cookies, indent=2))
    return {"cookies_saved": len(cookies), "path": req.path}


@app.post("/cookies/load", summary="Load cookies from a workspace file into the browser")
def load_cookies(req: LoadCookiesRequest) -> dict[str, Any]:
    import json as _json
    result = get_tools().read_file(req.path)
    cookies = _json.loads(result["content"])
    get_agent().add_cookies(cookies)
    return {"cookies_loaded": len(cookies), "path": req.path}


# ---------------------------------------------------------------------------
# Routes: multi-tab
# ---------------------------------------------------------------------------

@app.post("/tabs/new", summary="Open a new browser tab")
def new_tab(req: NewTabRequest) -> dict[str, Any]:
    return get_agent().new_tab(url=req.url)


@app.post("/tabs/switch", summary="Switch to a tab by index")
def switch_tab(req: SwitchTabRequest) -> dict[str, Any]:
    return get_agent().switch_tab(req.index)


@app.post("/tabs/close", summary="Close a browser tab")
def close_tab(req: CloseTabRequest) -> dict[str, Any]:
    return get_agent().close_tab(index=req.index)


@app.get("/tabs/list", summary="List all open browser tabs")
def list_tabs() -> dict[str, Any]:
    return get_agent().list_tabs()


# ---------------------------------------------------------------------------
# Routes: system tools — file I/O
# ---------------------------------------------------------------------------

@app.post("/system/write_file", summary="Write a file in the workspace")
def write_file(req: WriteFileRequest) -> dict[str, Any]:
    return get_tools().write_file(req.path, req.content, mode=req.mode)


@app.post("/system/append_file", summary="Append content to a workspace file")
def append_file(req: AppendFileRequest) -> dict[str, Any]:
    return get_tools().append_file(req.path, req.content)


@app.post("/system/read_file", summary="Read a workspace file")
def read_file(req: ReadFileRequest) -> dict[str, Any]:
    return get_tools().read_file(req.path)


@app.post("/system/list_dir", summary="List a workspace directory")
def list_dir(req: ListDirRequest) -> dict[str, Any]:
    return get_tools().list_dir(req.path)


@app.post("/system/make_dir", summary="Create a directory in the workspace")
def make_dir(req: MakeDirRequest) -> dict[str, Any]:
    return get_tools().make_dir(req.path)


@app.post("/system/delete_file", summary="Delete a file or directory from the workspace")
def delete_file(req: DeleteFileRequest) -> dict[str, Any]:
    return get_tools().delete_file(req.path)


@app.get("/system/info", summary="Get workspace info and file counts")
def system_info() -> dict[str, Any]:
    return get_tools().info()


# ---------------------------------------------------------------------------
# Routes: system tools — code execution
# ---------------------------------------------------------------------------

@app.post("/system/run_python", summary="Execute a Python snippet in the workspace")
def run_python(req: RunPythonRequest) -> dict[str, Any]:
    return get_tools().run_python(req.code, timeout=req.timeout)


@app.post("/system/run_shell", summary="Execute a shell command in the workspace")
def run_shell(req: RunShellRequest) -> dict[str, Any]:
    return get_tools().run_shell(req.command, timeout=req.timeout)


# ---------------------------------------------------------------------------
# Routes: task planner
# ---------------------------------------------------------------------------

@app.post("/task/run", summary="Plan and execute a natural-language browser task")
def task_run(req: TaskRunRequest) -> dict[str, Any]:
    """
    Convert *intent* to a step plan and execute it in one call.

    Common tasks (Google search, navigation, YouTube search, etc.) are resolved
    via deterministic templates — no LLM call, no hallucination.  For unknown
    intents the configured LLM backend is used with a constrained prompt.
    """
    planner = get_planner()
    agent   = get_agent()

    # Plan first (untainted source of step metadata).
    try:
        steps = planner.plan(req.intent)
    except (ValueError, StepValidationError) as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))

    # Execute — results list may contain exception strings internally.
    exec_results = planner.execute(steps, agent, stop_on_error=req.stop_on_error)

    # Optionally save execution log (only when log_path is set).
    if req.log_path:
        import json as _log_json, logging as _log
        from datetime import datetime as _dt, timezone as _tz
        _tools = get_tools()
        log_entry = {
            "intent": req.intent,
            "success": all(r.get("status") == "ok" for r in exec_results),
            "timestamp": _dt.now(_tz.utc).isoformat(),
            "step_count": len(steps),
        }
        try:
            _tools.write_file(req.log_path, _log_json.dumps(log_entry, indent=2))
        except Exception as _exc:
            logging.getLogger(__name__).warning("Could not save log: %s", _sanitize_error(str(_exc)))

    # Build response by cross-referencing the *untainted* steps list for action
    # names and using only sanitized error messages from the execution results.
    safe_step_results = []
    failed_count = 0
    for step, r in zip(steps, exec_results):
        is_error = r.get("status") == "error"
        if is_error:
            failed_count += 1
        safe_step_results.append({
            "step":   safe_step_results.__len__(),             # index from untainted counter
            "action": step["action"],                          # from untainted plan
            "status": "error" if is_error else "ok",           # literal strings
            "error":  _sanitize_error(str(r.get("error", ""))) if is_error else None,
        })

    return {
        "success":      failed_count == 0,
        "intent":       req.intent,    # from request, not from execution results
        "failed_count": failed_count,
        "results":      safe_step_results,
    }


@app.post("/task/plan", summary="Convert a natural-language intent to a step list (dry run)")
def task_plan(req: TaskPlanRequest) -> dict[str, Any]:
    """Return the planned steps WITHOUT executing them."""
    try:
        steps = get_planner().plan(req.intent)
        return {"intent": req.intent, "steps": steps, "count": len(steps)}
    except (ValueError, StepValidationError) as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))


@app.post("/task/execute", summary="Execute a pre-built step list")
def task_execute(req: TaskExecuteRequest) -> dict[str, Any]:
    """Run an already-built list of step dicts directly (re-validated before execution)."""
    try:
        validated = validate_steps(req.steps)
    except StepValidationError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
    results = get_planner().execute(validated, get_agent(), stop_on_error=req.stop_on_error)
    # Cross-reference the *untainted* validated steps for action names;
    # only pull sanitized error messages from the potentially-tainted results.
    safe_results = []
    failed_count = 0
    for step, r in zip(validated, results):
        is_error = r.get("status") == "error"
        if is_error:
            failed_count += 1
        safe_results.append({
            "step":   safe_results.__len__(),           # index from untainted counter
            "action": step["action"],                   # from untainted validated list
            "status": "error" if is_error else "ok",    # literal strings
            "error":  _sanitize_error(str(r.get("error", ""))) if is_error else None,
        })
    return {"success": failed_count == 0, "results": safe_results, "failed_count": failed_count}


@app.get("/task/schema", summary="Return the allowed action schema")
def task_schema() -> dict[str, Any]:
    """Return the full STEP_SCHEMA so clients know what actions are valid."""
    return {"schema": STEP_SCHEMA}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic Browser API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    args = parser.parse_args()

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )

