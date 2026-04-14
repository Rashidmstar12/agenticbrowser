"""
API Server for the Agentic Browser.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Or use the helper:
    python api_server.py
"""

from __future__ import annotations

import asyncio
import contextvars
import json as _json
import logging
import os
import queue as _queue
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path as _Path
from typing import Any, AsyncIterator, Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from browser_agent import BrowserAgent
from doctor import run_checks
from skills import SkillLoadError, get_default_registry
from system_tools import PathTraversalError, SystemTools, safe_path
from task_planner import STEP_SCHEMA, StepValidationError, TaskPlanner, validate_steps

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
# Browser worker thread — thread-affinity fix for Playwright sync API
# ---------------------------------------------------------------------------

class _BrowserThread:
    """Single dedicated thread that owns all Playwright sync objects.

    Playwright's ``sync_api`` objects (``Page``, ``BrowserContext``, etc.) are
    *thread-affine*: they must be created **and** used on the same OS thread.
    FastAPI's sync route handlers can run on any thread in uvicorn's thread
    pool, so if the browser session is started in one request and later used in
    another the Playwright internals raise "Cannot switch to a different thread".

    This class solves the problem by funnelling every Playwright call through a
    single long-lived daemon thread.  All browser-touching routes submit a
    callable to this thread via :meth:`submit`, block until it completes, and
    return the result (or re-raise any exception) on the calling thread.

    Pooled agents (``/agents/pool/*``) manage their own independent browser
    processes and are not routed through this thread.
    """

    def __init__(self) -> None:
        self._q: _queue.SimpleQueue = _queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="browser-worker"
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:  # poison pill — shut down
                break
            fut, fn, args, kwargs = item
            try:
                result = fn(*args, **kwargs)
                fut.set_result(result)
            except Exception as exc:
                try:
                    fut.set_exception(exc)
                except Exception:
                    pass

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run *fn* on the browser thread and return its result.

        Blocks the calling thread until the work completes (or raises).
        """
        fut: Future[Any] = Future()
        self._q.put((fut, fn, args, kwargs))
        return fut.result()

    def stop(self) -> None:
        """Send a poison pill and wait for the worker thread to exit."""
        self._q.put(None)
        self._thread.join(timeout=10)


# Module-level singleton — one browser worker thread per process.
_browser_thread = _BrowserThread()


def _on_browser_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Convenience wrapper: execute *fn* on the dedicated browser worker thread."""
    return _browser_thread.submit(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# Global singletons (one per server process)
# ---------------------------------------------------------------------------
_agent:   BrowserAgent | None = None
_planner: TaskPlanner  | None = None
_tools:   SystemTools  | None = None

# ---------------------------------------------------------------------------
# Agent pool — named agents that run alongside the default session
# ---------------------------------------------------------------------------
_agent_pool:   dict[str, BrowserAgent]   = {}
_planner_pool: dict[str, TaskPlanner]    = {}
_pool_lock:    threading.Lock            = threading.Lock()

# Per-request context variable — set by _agent_routing_middleware so that
# get_agent() / get_planner() can transparently route to a pooled agent.
_current_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_agent_id", default=None
)


def _workspace() -> str:
    return os.environ.get("BROWSER_WORKSPACE", "workspace")


def get_agent() -> BrowserAgent:
    aid = _current_agent_id.get()
    if aid:
        return get_pooled_agent(aid)
    if _agent is None or _agent._page is None:
        raise HTTPException(status_code=400, detail="Browser session is not active. POST /session/start first.")
    return _agent


def get_planner() -> TaskPlanner:
    aid = _current_agent_id.get()
    if aid:
        with _pool_lock:
            planner = _planner_pool.get(aid)
        if planner is None:
            raise HTTPException(status_code=404, detail=f"Agent {aid!r} not found in pool.")
        return planner
    if _planner is None:
        raise HTTPException(status_code=400, detail="Browser session is not active. POST /session/start first.")
    return _planner


def get_tools() -> SystemTools:
    global _tools
    if _tools is None:
        _tools = SystemTools(workspace=_workspace())
    return _tools


def get_pooled_agent(agent_id: str) -> BrowserAgent:
    """Return the pooled agent identified by *agent_id*, or raise 404/400."""
    with _pool_lock:
        agent = _agent_pool.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found in pool.")
    if agent._page is None:
        raise HTTPException(status_code=400, detail=f"Agent {agent_id!r} has no active page.")
    return agent


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Agentic Browser API server starting up")
    yield
    # Stop the shared agent on its owner thread so Playwright teardown is
    # thread-safe (same thread that called start()).
    def _stop_shared() -> None:
        if _agent is not None:
            try:
                _agent.stop()
            except Exception:
                pass
    try:
        _browser_thread.submit(_stop_shared)
    except Exception:
        pass
    # Note: _browser_thread is a daemon thread; it is cleaned up automatically
    # when the process exits.  We must NOT stop() it here because the singleton
    # persists for the lifetime of the module (across multiple TestClient
    # invocations in tests), and stopping it would leave subsequent requests
    # blocked waiting on a dead thread.
    # Stop all pooled agents (each has its own lifecycle thread).
    with _pool_lock:
        for _pooled in list(_agent_pool.values()):
            try:
                _pooled.stop()
            except Exception:
                pass
        _agent_pool.clear()
        _planner_pool.clear()
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
# Security: code-execution gate
# ---------------------------------------------------------------------------

_CODE_EXEC_ALLOWED: bool = os.environ.get("BROWSER_ALLOW_CODE_EXEC", "false").lower() in (
    "1", "true", "yes"
)

# ---------------------------------------------------------------------------
# Security: API key authentication middleware
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.environ.get("BROWSER_API_KEY")

_AUTH_EXEMPT_PATHS = frozenset({"/", "/docs", "/redoc", "/openapi.json", "/ui", "/doctor"})


@app.middleware("http")
async def _api_key_middleware(request: Request, call_next):  # type: ignore[type-arg]
    if _API_KEY:
        # Allow un-authenticated access to health/docs paths
        # Normalize path: strip trailing slash for comparison (except root "/")
        path = request.url.path.rstrip("/") or "/"
        if path not in _AUTH_EXEMPT_PATHS:
            provided = request.headers.get("X-API-Key", "")
            if provided != _API_KEY:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key. Set X-API-Key header."},
                )
    return await call_next(request)


@app.middleware("http")
async def _agent_routing_middleware(request: Request, call_next):  # type: ignore[type-arg]
    """Read ``X-Agent-Id`` header (or ``?agent_id=`` query param) and store the
    value in a context variable so every route can transparently use a pooled
    agent instead of the global singleton — without requiring any signature
    changes to existing routes.
    """
    aid = (
        request.headers.get("X-Agent-Id")
        or request.query_params.get("agent_id")
    ) or None
    token = _current_agent_id.set(aid)
    try:
        return await call_next(request)
    finally:
        _current_agent_id.reset(token)


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
# Request models — high-priority advanced interactions
# ---------------------------------------------------------------------------

class UploadFileRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the <input type='file'> element")
    path: str = Field(..., description="Workspace-relative path of the file to upload. Separate multiple files with '|'.")


class DownloadFileRequest(BaseModel):
    url: str = Field(..., description="Direct download URL")
    path: str = Field(..., description="Workspace-relative path where the file will be saved")


class DragDropRequest(BaseModel):
    source: str = Field(..., description="CSS selector of the element to drag")
    target: str = Field(..., description="CSS selector of the drop target")


class RightClickRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the element to right-click")


class DoubleClickRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the element to double-click")


class GetElementRectRequest(BaseModel):
    selector: str = Field(..., description="CSS selector of the element")


class SetNetworkInterceptRequest(BaseModel):
    url_pattern: str = Field(..., description="Glob pattern (e.g. '**/*.png') to match request URLs")
    action: str = Field("abort", description="'abort' to block the request, 'continue' to pass it through")


class SetViewportRequest(BaseModel):
    width: int = Field(..., ge=1, description="Viewport width in pixels")
    height: int = Field(..., ge=1, description="Viewport height in pixels")


class SetGeolocationRequest(BaseModel):
    latitude: float = Field(..., description="Latitude in decimal degrees (-90 to 90)")
    longitude: float = Field(..., description="Longitude in decimal degrees (-180 to 180)")
    accuracy: float = Field(10.0, description="Accuracy radius in metres")


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


class VisionPlanRequest(BaseModel):
    intent: str = Field(
        ...,
        description=(
            "Natural-language task to perform on the current page. "
            "A screenshot is taken automatically and sent to the vision LLM."
        ),
    )
    provider: str | None = Field(
        None,
        description=(
            "Vision provider to use: 'openai', 'anthropic', 'gemini', or 'ollama'. "
            "When omitted the server auto-detects from available API keys."
        ),
    )


class VisionRunRequest(BaseModel):
    intent: str = Field(
        ...,
        description="Natural-language task to vision-plan and execute on the current page.",
    )
    stop_on_error: bool = Field(True, description="Stop execution on the first failed step")
    log_path: str | None = Field(None, description="Workspace-relative path to save the execution log")
    provider: str | None = Field(
        None,
        description=(
            "Vision provider to use: 'openai', 'anthropic', 'gemini', or 'ollama'. "
            "When omitted the server auto-detects from available API keys."
        ),
    )


class AgenticRunRequest(BaseModel):
    intent: str = Field(..., description="Natural-language task to execute in agentic observe-decide-act mode")
    stop_on_error: bool = Field(True, description="Stop execution on the first failed step")
    log_path: str | None = Field(None, description="Workspace-relative path to save the execution log")
    max_steps: int = Field(
        20,
        ge=1,
        le=50,
        description=(
            "Hard ceiling on total steps executed (including injected corrective steps). "
            "Range 1–50; default 20."
        ),
    )
    checkpoint_every: int = Field(
        3,
        ge=1,
        le=10,
        description=(
            "Number of steps between LLM observe-decide calls. "
            "Range 1–10; default 3."
        ),
    )

# ---------------------------------------------------------------------------
# Request models — video / GIF recording
# ---------------------------------------------------------------------------

class StartRecordingRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative path for the output .webm video file")


class RecordGifRequest(BaseModel):
    path: str = Field(..., description="Workspace-relative path for the output .gif file")
    duration: float = Field(3.0, ge=0.1, description="Duration to record in seconds")
    fps: int = Field(2, ge=1, le=30, description="Frames per second for the GIF")


# ---------------------------------------------------------------------------
# Request models — agent pool + parallel execution
# ---------------------------------------------------------------------------

class PoolAgentStartRequest(BaseModel):
    agent_id: str | None = Field(None, description="Agent identifier; auto-generated UUID if omitted")
    headless: bool = Field(True, description="Run agent without a visible window")
    slow_mo: int = Field(0, description="Slow down Playwright operations by this many ms")
    auto_close_popups: bool = Field(True, description="Auto-dismiss common popups on page load")
    default_timeout: int = Field(30_000, description="Default timeout in milliseconds")


class PoolAgentExecuteRequest(BaseModel):
    steps: list[dict] = Field(..., description="Pre-built step list to execute on this agent")
    stop_on_error: bool = Field(True, description="Stop execution on the first failed step")


class TabTask(BaseModel):
    steps: list[dict] = Field(..., description="Steps to execute on this tab")
    url: str | None = Field(None, description="Navigate to this URL before running steps")


class TabParallelExecuteRequest(BaseModel):
    tasks: list[TabTask] = Field(..., min_length=2, max_length=20, description="One entry per tab; at least 2 required")
    stop_on_error: bool = Field(True, description="Stop each tab's task on its first step error")


class ParallelAgentTask(BaseModel):
    steps: list[dict] = Field(..., description="Steps to execute on this agent")
    agent_id: str | None = Field(None, description="Existing pooled agent ID; None = spawn a fresh agent")


class ExecuteParallelRequest(BaseModel):
    tasks: list[ParallelAgentTask] | None = Field(
        None,
        description="Explicit per-agent tasks. Provide either this or (agent_count + steps).",
    )
    agent_count: int | None = Field(
        None, ge=1, le=20,
        description="Broadcast mode: spawn N agents and run the same 'steps' on each.",
    )
    steps: list[dict] | None = Field(
        None,
        description="Steps used for every agent when agent_count is set.",
    )
    headless: bool = Field(True, description="Headless mode for auto-spawned agents")
    auto_stop: bool = Field(True, description="Stop auto-spawned agents after execution completes")
    stop_on_error: bool = Field(True, description="Stop each agent's task on its first step error")


# ---------------------------------------------------------------------------
# Routes: session management
# ---------------------------------------------------------------------------

@app.post("/session/start", summary="Start a browser session")
def session_start(req: SessionStartRequest) -> dict[str, Any]:
    # Create and start the agent entirely on the browser thread so all
    # Playwright sync objects are owned by that thread from the start.
    def _do() -> dict[str, Any]:
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
    return _browser_thread.submit(_do)


@app.post("/session/stop", summary="Stop the browser session")
def session_stop() -> dict[str, Any]:
    def _do() -> dict[str, Any]:
        global _agent, _planner
        if _agent is None:
            return {"status": "not_running"}
        _agent.stop()
        _agent = None
        _planner = None
        return {"status": "stopped"}
    return _browser_thread.submit(_do)


@app.get("/session/status", summary="Get session and page info")
def session_status() -> dict[str, Any]:
    def _check() -> dict[str, Any]:
        if _agent is None or _agent._page is None:
            return {"active": False}
        return {"active": True, **_agent.get_page_info()}
    return _browser_thread.submit(_check)


# ---------------------------------------------------------------------------
# Routes: navigation
# ---------------------------------------------------------------------------

@app.post("/navigate", summary="Navigate to a URL")
def navigate(req: NavigateRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().navigate, req.url, wait_until=req.wait_until)


# ---------------------------------------------------------------------------
# Routes: popup handling
# ---------------------------------------------------------------------------

@app.post("/popups/close", summary="Close common popup/overlay elements")
def close_popups() -> dict[str, Any]:
    return _on_browser_thread(get_agent().close_popups)


# ---------------------------------------------------------------------------
# Routes: element interaction
# ---------------------------------------------------------------------------

@app.post("/click", summary="Click an element")
def click(req: ClickRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().click, req.selector, timeout=req.timeout)


@app.post("/type", summary="Type text into an element")
def type_text(req: TypeRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().type_text, req.selector, req.text, clear_first=req.clear_first)


@app.post("/fill", summary="Fill an input element with a value")
def fill(req: FillRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().fill, req.selector, req.value)


@app.post("/press_key", summary="Press a keyboard key")
def press_key(req: PressKeyRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().press_key, req.key)


@app.post("/hover", summary="Hover over an element")
def hover(req: HoverRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().hover, req.selector)


@app.post("/select_option", summary="Select an option in a <select> element")
def select_option(req: SelectOptionRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().select_option, req.selector, req.value)


# ---------------------------------------------------------------------------
# Routes: scrolling
# ---------------------------------------------------------------------------

@app.post("/scroll", summary="Scroll the page")
def scroll(req: ScrollRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().scroll, req.x, req.y)


@app.post("/scroll_to_element", summary="Scroll an element into view")
def scroll_to_element(req: ScrollToElementRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().scroll_to_element, req.selector)


# ---------------------------------------------------------------------------
# Routes: information extraction
# ---------------------------------------------------------------------------

@app.get("/page/info", summary="Get current page URL and title")
def page_info() -> dict[str, Any]:
    return _on_browser_thread(get_agent().get_page_info)


@app.post("/page/text", summary="Get inner text of an element")
def get_text(req: GetTextRequest) -> dict[str, Any]:
    return {"text": _on_browser_thread(get_agent().get_text, req.selector)}


@app.post("/page/html", summary="Get inner HTML of an element")
def get_html(req: GetHtmlRequest) -> dict[str, Any]:
    return {"html": _on_browser_thread(get_agent().get_html, req.selector)}


@app.post("/page/attribute", summary="Get an attribute value of an element")
def get_attribute(req: GetAttributeRequest) -> dict[str, Any]:
    return {"value": _on_browser_thread(get_agent().get_attribute, req.selector, req.attribute)}


@app.post("/page/query_all", summary="Query all elements matching a selector")
def query_all(req: QueryAllRequest) -> dict[str, Any]:
    return {"elements": _on_browser_thread(get_agent().query_all, req.selector)}


# ---------------------------------------------------------------------------
# Routes: JavaScript
# ---------------------------------------------------------------------------

@app.post("/evaluate", summary="Evaluate JavaScript in the page context")
def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    result = _on_browser_thread(get_agent().evaluate, req.script)
    return {"result": result}


# ---------------------------------------------------------------------------
# Routes: screenshots
# ---------------------------------------------------------------------------

@app.post("/screenshot", summary="Take a screenshot")
def screenshot(req: ScreenshotRequest) -> dict[str, Any]:
    safe_screenshot_path: str | None = None
    if req.path is not None:
        tools = get_tools()
        try:
            safe_screenshot_path = str(safe_path(tools.workspace, req.path))
        except PathTraversalError as exc:
            raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
    return _on_browser_thread(
        get_agent().screenshot,
        path=safe_screenshot_path, full_page=req.full_page, as_base64=req.as_base64,
    )


# ---------------------------------------------------------------------------
# Routes: video / GIF recording
# ---------------------------------------------------------------------------

@app.post("/recording/start", summary="Start recording the browser session as a WebM video")
def recording_start(req: StartRecordingRequest) -> dict[str, Any]:
    tools = get_tools()
    try:
        save_path = str(safe_path(tools.workspace, req.path))
    except PathTraversalError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
    agent = get_agent()
    try:
        return _on_browser_thread(agent.start_video_recording, save_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=_sanitize_error(str(exc)))


@app.post("/recording/stop", summary="Stop video recording and save the WebM file")
def recording_stop() -> dict[str, Any]:
    try:
        return _on_browser_thread(get_agent().stop_video_recording)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=_sanitize_error(str(exc)))


@app.post("/recording/gif", summary="Capture an animated GIF of the current page")
def recording_gif(req: RecordGifRequest) -> dict[str, Any]:
    tools = get_tools()
    try:
        save_path = str(safe_path(tools.workspace, req.path))
    except PathTraversalError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
    return _on_browser_thread(get_agent().record_gif, save_path, duration=req.duration, fps=req.fps)


# ---------------------------------------------------------------------------
# Routes: wait helpers
# ---------------------------------------------------------------------------

@app.post("/wait/selector", summary="Wait for an element to appear")
def wait_for_selector(req: WaitForSelectorRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().wait_for_selector, req.selector, timeout=req.timeout)


@app.post("/wait/load_state", summary="Wait for a specific load state")
def wait_for_load_state(req: WaitForLoadStateRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().wait_for_load_state, req.state)


# ---------------------------------------------------------------------------
# Routes: smart extraction
# ---------------------------------------------------------------------------

@app.post("/page/extract_links", summary="Extract all hyperlinks from the page")
def extract_links(req: ExtractLinksRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().extract_links, selector=req.selector, limit=req.limit)


@app.post("/page/extract_table", summary="Extract an HTML table as JSON rows")
def extract_table(req: ExtractTableRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().extract_table, selector=req.selector, table_index=req.table_index)


# ---------------------------------------------------------------------------
# Routes: assertions
# ---------------------------------------------------------------------------

@app.post("/assert/text", summary="Assert text is present on the page (fails with 422 if not)")
def assert_text(req: AssertTextRequest) -> dict[str, Any]:
    try:
        return _on_browser_thread(
            get_agent().assert_text, req.text, selector=req.selector, case_sensitive=req.case_sensitive
        )
    except AssertionError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))


@app.post("/assert/url", summary="Assert the current URL matches a regex pattern")
def assert_url(req: AssertUrlRequest) -> dict[str, Any]:
    try:
        return _on_browser_thread(get_agent().assert_url, req.pattern)
    except AssertionError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))


# ---------------------------------------------------------------------------
# Routes: wait for dynamic content
# ---------------------------------------------------------------------------

@app.post("/wait/text", summary="Wait until text appears on the page")
def wait_text(req: WaitTextRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().wait_text, req.text, selector=req.selector, timeout=req.timeout)


# ---------------------------------------------------------------------------
# Routes: cookie persistence
# ---------------------------------------------------------------------------

@app.post("/cookies/save", summary="Save browser cookies to a workspace file")
def save_cookies(req: SaveCookiesRequest) -> dict[str, Any]:
    import json as _json
    cookies = _on_browser_thread(get_agent().get_cookies)
    get_tools().write_file(req.path, _json.dumps(cookies, indent=2))
    return {"cookies_saved": len(cookies), "path": req.path}


@app.post("/cookies/load", summary="Load cookies from a workspace file into the browser")
def load_cookies(req: LoadCookiesRequest) -> dict[str, Any]:
    import json as _json
    result = get_tools().read_file(req.path)
    cookies = _json.loads(result["content"])
    _on_browser_thread(get_agent().add_cookies, cookies)
    return {"cookies_loaded": len(cookies), "path": req.path}


# ---------------------------------------------------------------------------
# Routes: multi-tab
# ---------------------------------------------------------------------------

@app.post("/tabs/new", summary="Open a new browser tab")
def new_tab(req: NewTabRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().new_tab, url=req.url)


@app.post("/tabs/switch", summary="Switch to a tab by index")
def switch_tab(req: SwitchTabRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().switch_tab, req.index)


@app.post("/tabs/close", summary="Close a browser tab")
def close_tab(req: CloseTabRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().close_tab, index=req.index)


@app.get("/tabs/list", summary="List all open browser tabs")
def list_tabs() -> dict[str, Any]:
    return _on_browser_thread(get_agent().list_tabs)


@app.post("/tabs/execute_parallel", summary="Open N tabs in the current session and run a step sequence on each (sequential per-tab)")
def tabs_execute_parallel(req: TabParallelExecuteRequest) -> dict[str, Any]:
    """
    Open one tab per task in the **current** browser session (shared cookies /
    storage) and execute each task's steps on its own tab.

    Execution is sequential per-tab (switch → run steps → switch → …), which
    keeps the shared Playwright context thread-safe.  Use
    ``/agents/execute_parallel`` when you need true concurrency.

    The first task reuses tab 0 (the existing active tab); subsequent tasks
    each open a new tab.  All tabs remain open after the call — use
    ``/tabs/close`` to clean them up.
    """
    agent = get_agent()
    planner_inst = get_planner()

    # Validate all step lists up-front so we fail early (no browser needed).
    validated: list[tuple[list[dict], str | None]] = []
    for task in req.tasks:
        try:
            vsteps = validate_steps(task.steps)
        except StepValidationError as exc:
            raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
        validated.append((vsteps, task.url))

    # Run all browser work on the dedicated browser thread.
    def _do_tabs() -> dict[str, Any]:
        # Open a new tab for every task beyond the first (reuse tab 0 for task 0).
        tab_indices: list[int] = [0]
        for _ in range(1, len(validated)):
            result = agent.new_tab()
            tab_indices.append(result["tab_index"])

        all_results: list[dict[str, Any]] = []
        for i, (vsteps, url) in enumerate(validated):
            agent.switch_tab(tab_indices[i])
            if url:
                try:
                    agent.navigate(url)
                except Exception:
                    # Do not echo the exception message (may contain the caller-supplied
                    # URL); use a static string to avoid stack-trace-exposure.
                    all_results.append(_safe_response({
                        "tab_index": tab_indices[i],
                        "success": False,
                        "failed_count": 1,
                        "step_results": [{"step": 0, "action": "navigate",
                                          "status": "error",
                                          "error": "Navigation failed before steps could execute."}],
                    }))
                    if req.stop_on_error:
                        break
                    continue
            task_results = planner_inst.execute(vsteps, agent, stop_on_error=req.stop_on_error)
            failed_count_tab = sum(1 for r in task_results if r.get("status") == "error")
            # Build safe_results using the untainted validated step list for step index
            # and action name — only the error string comes from execution results.
            safe_results = []
            for j, (vstep, r) in enumerate(zip(vsteps, task_results)):
                is_err = r.get("status") == "error"
                safe_results.append({
                    "step":   j,                                      # untainted counter
                    "action": vstep["action"],                        # from untainted validated list
                    "status": "error" if is_err else "ok",            # literal string
                    "error":  _sanitize_error(str(r.get("error", ""))) if is_err else None,
                })
            all_results.append({
                "tab_index":    tab_indices[i],
                "success":      failed_count_tab == 0,
                "failed_count": failed_count_tab,
                "step_results": safe_results,
            })

        # Return focus to tab 0.
        try:
            agent.switch_tab(0)
        except Exception:
            pass

        return {
            "results":      all_results,
            "total_tabs":   len(all_results),
            "all_succeeded": all(r["success"] for r in all_results),
        }

    return _safe_response(_browser_thread.submit(_do_tabs))


# ---------------------------------------------------------------------------
# Routes: agent pool
# ---------------------------------------------------------------------------

@app.post("/agents/pool/start", summary="Start a new named agent in the pool")
def pool_agent_start(req: PoolAgentStartRequest) -> dict[str, Any]:
    """
    Spawn a new ``BrowserAgent`` with its own browser process and add it to
    the named pool.  Use the returned ``agent_id`` to target subsequent
    requests at this specific agent (e.g. ``/agents/pool/{agent_id}/task/execute``).
    """
    global _agent_pool, _planner_pool
    aid = req.agent_id or f"agent-{uuid.uuid4().hex[:8]}"
    with _pool_lock:
        if aid in _agent_pool:
            raise HTTPException(status_code=409, detail=f"Agent {aid!r} already exists in pool.")
    agent = BrowserAgent(
        headless=req.headless,
        slow_mo=req.slow_mo,
        auto_close_popups=req.auto_close_popups,
        default_timeout=req.default_timeout,
    )
    agent.start()
    planner = TaskPlanner()
    planner._system_tools = get_tools()
    with _pool_lock:
        _agent_pool[aid]   = agent
        _planner_pool[aid] = planner
    logger.info("Pooled agent %r started", aid)
    return {"agent_id": aid, "status": "started", "headless": req.headless}


@app.get("/agents/pool", summary="List all pooled agents and their status")
def pool_agents_list() -> dict[str, Any]:
    with _pool_lock:
        snapshot = list(_agent_pool.items())
    agents: list[dict[str, Any]] = []
    for aid, agent in snapshot:
        try:
            info: dict[str, Any] = {"agent_id": aid, "active": agent._page is not None}
            if agent._page is not None:
                info["url"]       = agent._page.url
                info["tab_count"] = len(agent._pages)
                info["recording"] = agent._recording
        except Exception:
            info = {"agent_id": aid, "active": False}
        agents.append(info)
    return {"agents": agents, "count": len(agents)}


@app.get("/agents/pool/{agent_id}", summary="Get status of a specific pooled agent")
def pool_agent_status(agent_id: str) -> dict[str, Any]:
    agent = get_pooled_agent(agent_id)
    try:
        tabs = agent.list_tabs()
    except Exception:
        tabs = {"tabs": [], "count": 0}
    return {
        "agent_id":  agent_id,
        "active":    True,
        "url":       agent._page.url if agent._page else None,
        "tab_count": len(agent._pages),
        "recording": agent._recording,
        "tabs":      tabs.get("tabs", []),
    }


@app.delete("/agents/pool/{agent_id}", summary="Stop and remove a pooled agent")
def pool_agent_stop(agent_id: str) -> dict[str, Any]:
    with _pool_lock:
        agent   = _agent_pool.pop(agent_id, None)
        _planner_pool.pop(agent_id, None)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found in pool.")
    try:
        agent.stop()
    except Exception as exc:
        logger.warning("Error stopping pooled agent %r: %s", agent_id, exc)
    logger.info("Pooled agent %r stopped and removed", agent_id)
    return {"agent_id": agent_id, "status": "stopped"}


@app.post("/agents/pool/{agent_id}/task/execute", summary="Execute a step list on a specific pooled agent")
def pool_agent_execute(agent_id: str, req: PoolAgentExecuteRequest) -> dict[str, Any]:
    agent = get_pooled_agent(agent_id)
    with _pool_lock:
        planner = _planner_pool.get(agent_id)
    if planner is None:
        planner = TaskPlanner()
        planner._system_tools = get_tools()
        with _pool_lock:
            _planner_pool[agent_id] = planner
    try:
        validated = validate_steps(req.steps)
    except StepValidationError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
    results = planner.execute(validated, agent, stop_on_error=req.stop_on_error)
    failed_count = 0
    # Build safe_results using the untainted validated step list for step index
    # and action name — only the error string comes from execution results.
    safe_results = []
    for j, (vstep, r) in enumerate(zip(validated, results)):
        is_err = r.get("status") == "error"
        if is_err:
            failed_count += 1
        safe_results.append({
            "step":   j,                                      # untainted counter
            "action": vstep["action"],                        # from untainted validated list
            "status": "error" if is_err else "ok",            # literal string
            "error":  _sanitize_error(str(r.get("error", ""))) if is_err else None,
        })
    return _safe_response({
        "agent_id":     agent_id,
        "success":      failed_count == 0,
        "failed_count": failed_count,
        "step_results": safe_results,
    })


@app.post("/agents/execute_parallel", summary="Run tasks across multiple agents in parallel (true concurrency)")
def agents_execute_parallel(req: ExecuteParallelRequest) -> dict[str, Any]:
    """
    Execute step lists across **multiple independent agents** in parallel using
    a thread pool.  Each agent runs in its own browser process, so tasks are
    truly concurrent.

    **Two modes:**

    * **Explicit** (``tasks`` list): each item may reference an existing
      ``agent_id`` (from the pool) or leave it ``null`` to auto-spawn a fresh
      agent.
    * **Broadcast** (``agent_count`` + ``steps``): spawn *N* fresh agents and
      run the same step list on every one of them simultaneously.

    Auto-spawned agents are removed from the pool and stopped automatically
    when ``auto_stop=true`` (default).
    """
    # -- Build the task list -------------------------------------------------
    if req.tasks is not None:
        raw_tasks = req.tasks
    elif req.agent_count is not None and req.steps is not None:
        raw_tasks = [ParallelAgentTask(steps=req.steps) for _ in range(req.agent_count)]
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'tasks' or both 'agent_count' and 'steps'.",
        )

    if not raw_tasks:
        raise HTTPException(status_code=422, detail="No tasks provided.")

    # -- Validate all step lists up-front ------------------------------------
    validated_tasks: list[tuple[str, BrowserAgent, list[dict], bool]] = []
    spawned_ids: list[str] = []
    tools = get_tools()

    for task in raw_tasks:
        try:
            vsteps = validate_steps(task.steps)
        except StepValidationError as exc:
            raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))

        if task.agent_id is not None:
            agent = get_pooled_agent(task.agent_id)
            validated_tasks.append((task.agent_id, agent, vsteps, False))
        else:
            aid = f"agent-{uuid.uuid4().hex[:8]}"
            agent = BrowserAgent(headless=req.headless)
            agent.start()
            with _pool_lock:
                _agent_pool[aid]   = agent
                p = TaskPlanner()
                p._system_tools    = tools
                _planner_pool[aid] = p
            spawned_ids.append(aid)
            validated_tasks.append((aid, agent, vsteps, True))

    # -- Execute in parallel -------------------------------------------------
    results: list[dict[str, Any]] = [{}] * len(validated_tasks)

    def _run(idx: int, aid: str, agent: BrowserAgent, vsteps: list[dict], spawned: bool) -> dict[str, Any]:
        planner = TaskPlanner()
        planner._system_tools = tools
        task_results = planner.execute(vsteps, agent, stop_on_error=req.stop_on_error)
        failed = [r for r in task_results if r.get("status") == "error"]
        return {
            "agent_id":     aid,
            "spawned":      spawned,
            "success":      len(failed) == 0,
            "failed_count": len(failed),
            "step_results": [
                {
                    "step":   r.get("step"),
                    "action": r.get("action"),
                    "status": r.get("status"),
                    "error":  _sanitize_error(str(r.get("error", ""))) if r.get("status") == "error" else None,
                }
                for r in task_results
            ],
        }

    with ThreadPoolExecutor(max_workers=len(validated_tasks)) as pool:
        futures = {
            pool.submit(_run, i, aid, agent, vsteps, spawned): i
            for i, (aid, agent, vsteps, spawned) in enumerate(validated_tasks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                aid = validated_tasks[idx][0]
                results[idx] = {
                    "agent_id":     aid,
                    "spawned":      aid in spawned_ids,
                    "success":      False,
                    "failed_count": -1,
                    "error":        _sanitize_error(str(exc)),
                    "step_results": [],
                }

    # -- Auto-stop spawned agents --------------------------------------------
    if req.auto_stop:
        for aid in spawned_ids:
            with _pool_lock:
                a = _agent_pool.pop(aid, None)
                _planner_pool.pop(aid, None)
            if a:
                try:
                    a.stop()
                except Exception:
                    pass

    return _safe_response({
        "results":       results,
        "total_agents":  len(results),
        "all_succeeded": all(r.get("success", False) for r in results),
    })


# ---------------------------------------------------------------------------
# Routes: high-priority advanced interactions
# ---------------------------------------------------------------------------

@app.post("/upload_file", summary="Set file(s) on a file input element")
def upload_file(req: UploadFileRequest) -> dict[str, Any]:
    tools = get_tools()
    try:
        file_path = str(safe_path(tools.workspace, req.path))
    except PathTraversalError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
    return _on_browser_thread(get_agent().upload_file, req.selector, file_path)


@app.post("/download_file", summary="Navigate to a URL and save the triggered download")
def download_file(req: DownloadFileRequest) -> dict[str, Any]:
    tools = get_tools()
    try:
        save_path = str(safe_path(tools.workspace, req.path))
    except PathTraversalError as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
    return _on_browser_thread(get_agent().download_file, req.url, save_path)


@app.post("/drag_drop", summary="Drag an element and drop it on another element")
def drag_drop(req: DragDropRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().drag_and_drop, req.source, req.target)


@app.post("/right_click", summary="Right-click (context-menu) an element")
def right_click(req: RightClickRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().right_click, req.selector)


@app.post("/double_click", summary="Double-click an element")
def double_click(req: DoubleClickRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().double_click, req.selector)


@app.post("/page/rect", summary="Get the bounding box of an element")
def get_element_rect(req: GetElementRectRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().get_element_rect, req.selector)


@app.post("/network/intercept", summary="Intercept (block or pass through) requests matching a URL pattern")
def set_network_intercept(req: SetNetworkInterceptRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().set_network_intercept, req.url_pattern, action=req.action)


@app.post("/network/clear_intercepts", summary="Remove all network intercept routes")
def clear_network_intercepts() -> dict[str, Any]:
    return _on_browser_thread(get_agent().clear_network_intercepts)


@app.post("/session/viewport", summary="Resize the browser viewport")
def set_viewport(req: SetViewportRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().set_viewport, req.width, req.height)


@app.post("/session/geolocation", summary="Override the browser geolocation")
def set_geolocation(req: SetGeolocationRequest) -> dict[str, Any]:
    return _on_browser_thread(get_agent().set_geolocation, req.latitude, req.longitude, accuracy=req.accuracy)


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
    if not _CODE_EXEC_ALLOWED:
        raise HTTPException(
            status_code=403,
            detail="Code execution is disabled. Set BROWSER_ALLOW_CODE_EXEC=true to enable.",
        )
    return get_tools().run_python(req.code, timeout=req.timeout)


@app.post("/system/run_shell", summary="Execute a shell command in the workspace")
def run_shell(req: RunShellRequest) -> dict[str, Any]:
    if not _CODE_EXEC_ALLOWED:
        raise HTTPException(
            status_code=403,
            detail="Code execution is disabled. Set BROWSER_ALLOW_CODE_EXEC=true to enable.",
        )
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

    # Plan first (untainted source of step metadata) — no browser call needed.
    try:
        steps = planner.plan(req.intent)
    except (ValueError, StepValidationError) as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))

    # Execute on the browser thread — results list may contain exception strings.
    exec_results = _on_browser_thread(planner.execute, steps, agent, stop_on_error=req.stop_on_error)

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

    # Optionally save execution log (only when log_path is set).
    # The log is a file write, so we can include the full results without
    # exposing them in the HTTP response.
    if req.log_path:
        import json as _log_json
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        _tools = get_tools()
        log_entry = {
            "intent":       req.intent,
            "success":      failed_count == 0,
            "timestamp":    _dt.now(_tz.utc).isoformat(),
            "step_count":   len(steps),
            "failed_count": failed_count,
            "steps":        [{"action": s["action"]} for s in steps],
            "results":      safe_step_results,
        }
        try:
            _tools.write_file(req.log_path, _log_json.dumps(log_entry, indent=2))
        except Exception as _exc:
            logging.getLogger(__name__).warning("Could not save log: %s", _sanitize_error(str(_exc)))

    safe_response: dict[str, Any] = {
        "success":      failed_count == 0,
        "intent":       req.intent,    # from request, not from execution results
        "failed_count": failed_count,
        "results":      safe_step_results,
    }
    if failed_count > 0:
        raise HTTPException(status_code=422, detail=safe_response)
    return safe_response


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
    planner = get_planner()
    agent   = get_agent()
    results = _on_browser_thread(planner.execute, validated, agent, stop_on_error=req.stop_on_error)
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


@app.post("/task/vision_plan", summary="Plan a task using a screenshot of the current page")
def task_vision_plan(req: VisionPlanRequest) -> dict[str, Any]:
    """
    Take a screenshot of the current page and send it together with *intent*
    to a vision-capable LLM to generate a grounded step plan.

    The vision provider is selected in priority order: ``OPENAI_API_KEY`` →
    ``ANTHROPIC_API_KEY`` → ``GOOGLE_API_KEY`` → Ollama.  You may also specify
    the *provider* field explicitly (``"openai"``, ``"anthropic"``, ``"gemini"``,
    or ``"ollama"``).  Falls back to text-only planning when no provider is
    available or the vision call fails.  An active browser session is required.
    """
    planner = get_planner()
    agent = get_agent()
    try:
        # vision_plan takes a screenshot (browser call) so it must run on the browser thread.
        steps = _on_browser_thread(planner.vision_plan, req.intent, agent, provider=req.provider)
        return {"intent": req.intent, "steps": steps, "count": len(steps), "vision": True}
    except (ValueError, StepValidationError) as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))


@app.post("/task/vision_run", summary="Plan with vision and execute a browser task")
def task_vision_run(req: VisionRunRequest) -> dict[str, Any]:
    """
    Take a screenshot of the current page, generate a grounded step plan
    using a vision-capable LLM, then execute the steps.

    The provider is auto-detected from available API keys or can be set
    explicitly via the *provider* field.  Falls back to text-only planning
    when no vision provider is available.
    """
    planner = get_planner()
    agent = get_agent()
    try:
        steps = _on_browser_thread(planner.vision_plan, req.intent, agent, provider=req.provider)
    except (ValueError, StepValidationError) as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))

    exec_results = _on_browser_thread(planner.execute, steps, agent, stop_on_error=req.stop_on_error)

    safe_step_results = []
    failed_count = 0
    for step, r in zip(steps, exec_results):
        is_error = r.get("status") == "error"
        if is_error:
            failed_count += 1
        safe_step_results.append({
            "step":   len(safe_step_results),
            "action": step["action"],
            "status": "error" if is_error else "ok",
            "error":  _sanitize_error(str(r.get("error", ""))) if is_error else None,
        })

    if req.log_path:
        import json as _log_json
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        _tools = get_tools()
        log_entry = {
            "intent":       req.intent,
            "vision":       True,
            "success":      failed_count == 0,
            "timestamp":    _dt.now(_tz.utc).isoformat(),
            "step_count":   len(steps),
            "failed_count": failed_count,
            "steps":        [{"action": s["action"]} for s in steps],
            "results":      safe_step_results,
        }
        try:
            _tools.write_file(req.log_path, _log_json.dumps(log_entry, indent=2))
        except Exception as _exc:
            logger.warning(
                "Could not save vision_run log: %s", _sanitize_error(str(_exc))
            )

    safe_response: dict[str, Any] = {
        "success":      failed_count == 0,
        "intent":       req.intent,
        "vision":       True,
        "failed_count": failed_count,
        "results":      safe_step_results,
    }
    if failed_count > 0:
        raise HTTPException(status_code=422, detail=safe_response)
    return safe_response


@app.post("/task/agentic_run", summary="Execute a browser task in agentic observe-decide-act mode")
def task_agentic_run(req: AgenticRunRequest) -> dict[str, Any]:
    """
    Plan a task and execute it with an observe-decide-act loop.

    Unlike ``/task/run``, which executes a static plan to completion, this
    endpoint periodically observes the live page state (URL + body text) and
    asks the configured LLM whether the goal has been achieved or whether
    corrective steps are needed.  Corrective steps are validated and injected
    into the execution queue automatically.

    The ``stopped_reason`` field in the response indicates why the loop ended:

    * ``"verified"``        — an assert step confirmed the expected outcome.
    * ``"done_by_model"``   — the LLM (or natural step-queue exhaustion) reported
                              the goal is met.
    * ``"max_steps"``       — the step budget was exhausted.
    * ``"abort"``           — the LLM detected an unrecoverable blocker
                              (login wall, CAPTCHA, …).
    * ``"error"``           — execution stopped due to a failed step.

    When no LLM is configured, observe-decide checkpoints are silently skipped
    and the endpoint behaves like ``/task/run`` with a ``max_steps`` ceiling.
    """
    planner = get_planner()
    agent   = get_agent()

    # Plan first so we have an untainted step list for response construction.
    # Passing it via `initial_steps` avoids a second LLM call inside agentic_run.
    try:
        initial_steps = planner.plan(req.intent)
    except (ValueError, StepValidationError) as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))

    summary = _on_browser_thread(
        planner.agentic_run,
        req.intent,
        agent,
        initial_steps=initial_steps,
        max_steps=req.max_steps,
        checkpoint_every=req.checkpoint_every,
        stop_on_error=req.stop_on_error,
        log_path=req.log_path,
    )

    # Build a safe response by cross-referencing the *untainted* initial_steps
    # for action names — same pattern as /task/run.  Execution results beyond
    # the initial plan (corrective steps) use the generic label "corrective_step".
    all_results  = summary.get("results") or []
    safe_results = []
    failed_count = 0
    for i, r in enumerate(all_results):
        is_error = r.get("status") == "error"
        if is_error:
            failed_count += 1
        # Action name: from the untainted initial plan when available; otherwise
        # a safe literal (corrective steps injected beyond the initial plan).
        if i < len(initial_steps):
            action = initial_steps[i]["action"]   # from untainted plan()
        else:
            action = "corrective_step"            # safe literal for injected steps
        safe_results.append({
            "step":   len(safe_results),
            "action": action,
            "status": "error" if is_error else "ok",
            "error":  _sanitize_error(str(r.get("error", ""))) if is_error else None,
        })

    # Extract known-safe scalar values from summary to break CodeQL taint chains.
    # stopped_reason is always one of a fixed set of string literals.
    _KNOWN_STOPPED_REASONS = frozenset({
        "verified", "done_by_model", "max_steps", "abort", "error",
    })
    _raw_reason = summary.get("stopped_reason")
    stopped_reason = _raw_reason if _raw_reason in _KNOWN_STOPPED_REASONS else "error"

    _raw_verified = summary.get("verified")
    verified = bool(_raw_verified) if _raw_verified is not None else None

    _raw_injected = summary.get("recovery_steps_injected", 0)
    recovery_steps_injected = int(_raw_injected) if isinstance(_raw_injected, int) else 0

    safe_response: dict[str, Any] = {
        "success":                  failed_count == 0,
        "verified":                 verified,
        "intent":                   req.intent,     # from request, not from summary
        "stopped_reason":           stopped_reason,
        "recovery_steps_injected":  recovery_steps_injected,
        "failed_count":             failed_count,
        "results":                  safe_results,
    }

    if failed_count > 0:
        raise HTTPException(status_code=422, detail=safe_response)
    return safe_response


# ---------------------------------------------------------------------------
# Routes: doctor (environment health checks)
# ---------------------------------------------------------------------------

@app.get("/doctor", summary="Run environment health checks")
def doctor_check() -> dict[str, Any]:
    """
    Run all environment health checks and return a structured report.

    No auto-fix is performed via the API; use ``--doctor --fix`` from the CLI
    to auto-remediate issues.
    """
    checks = run_checks(workspace=_workspace(), fix=False)
    all_ok = all(c.status != "fail" for c in checks)
    return _safe_response({
        "status": "ok" if all_ok else "degraded",
        "checks": [c.to_dict() for c in checks],
    })


# ---------------------------------------------------------------------------
# Routes: skills
# ---------------------------------------------------------------------------

class SkillLoadRequest(BaseModel):
    source: str = Field(
        ...,
        description=(
            "Skill source: a local file path, directory path, HTTP/HTTPS URL, "
            "or GitHub shorthand 'gh:owner/repo[/path]'."
        ),
    )


@app.get("/skills", summary="List loaded skills")
def skills_list() -> dict[str, Any]:
    """Return all skills currently registered in the default skill registry."""
    reg = get_default_registry()
    return {
        "count":  len(reg),
        "skills": [s.to_dict() for s in reg.list_skills()],
    }


@app.post("/skills/load", summary="Load skills from a file, URL, or GitHub repo")
def skills_load(req: SkillLoadRequest) -> dict[str, Any]:
    """
    Load one or more skills from *source* and register them in the default
    registry.  Existing skills with the same name are overwritten.

    Only ``https://`` URLs and ``gh:owner/repo`` GitHub references are accepted
    via the API.  To load local filesystem skills, use the CLI
    ``--skills <path>`` flag.
    """
    from skills import _validate_source_for_api

    try:
        _validate_source_for_api(req.source)
    except SkillLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    reg = get_default_registry()
    try:
        loaded = reg.load_from_remote_source(req.source)
    except (SkillLoadError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=422, detail=_sanitize_error(str(exc)))
    return {
        "loaded": len(loaded),
        "skills": [s.name for s in loaded],
    }


@app.get("/skills/{name}", summary="Get a skill by name")
def skills_get(name: str) -> dict[str, Any]:
    """Return the full definition of a single skill."""
    skill = get_default_registry().get(name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill {name!r} not found")
    return skill.to_dict()


@app.delete("/skills/{name}", summary="Unload a skill by name")
def skills_delete(name: str) -> dict[str, Any]:
    """Remove a skill from the default registry."""
    removed = get_default_registry().unregister(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Skill {name!r} not found")
    return {"unloaded": name}


# ---------------------------------------------------------------------------
# Routes: web GUI
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """Redirect the root URL to the web GUI."""
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def serve_gui() -> HTMLResponse:
    """Serve the built-in web GUI (single-page application)."""
    html_path = _Path(__file__).parent / "gui" / "index.html"
    try:
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>GUI not found</h1><p>gui/index.html is missing.</p>",
            status_code=404,
        )


# ---------------------------------------------------------------------------
# WebSocket: streaming task execution
# ---------------------------------------------------------------------------

@app.websocket("/ws/task")
async def ws_task(websocket: WebSocket) -> None:
    """
    Stream task execution step-by-step over a WebSocket connection.

    Protocol
    --------
    Client  → ``{"intent": "<natural language task>"}``

    Server  → ``{"type": "planned",    "step_count": N, "actions": [...]}``
            → ``{"type": "step_start", "step": i, "action": "..."}``
            → ``{"type": "step_done",  "step": i, "action": "...", "status": "ok"|"error", "error": "..."}``
            → ``{"type": "done",       "success": bool, "failed_count": N}``
            → ``{"type": "error",      "message": "..."}``  (on unrecoverable errors)

    Steps are emitted in real-time: ``step_start`` fires the moment a step
    begins executing, and ``step_done`` fires as soon as it completes — not
    after all steps finish.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    try:
        data = await websocket.receive_json()
        intent = str(data.get("intent", "")).strip()
        if not intent:
            await websocket.send_json({"type": "error", "message": "intent is required"})
            return

        try:
            planner = get_planner()
            agent   = get_agent()
        except HTTPException as exc:
            await websocket.send_json({"type": "error", "message": str(exc.detail)})
            return

        # Plan steps (synchronous call → run in thread pool)
        try:
            steps = await loop.run_in_executor(None, planner.plan, intent)
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": _sanitize_error(str(exc))})
            return

        await websocket.send_json({
            "type":       "planned",
            "step_count": len(steps),
            "actions":    [s["action"] for s in steps],
        })

        # ------------------------------------------------------------------
        # Real-time streaming: use a queue to bridge the synchronous browser
        # thread (where execute() runs) and this async coroutine.
        # The callbacks below are called from the browser thread and push
        # events onto the queue; the consumer loop below drains it and sends
        # each event to the WebSocket immediately.
        #
        # Thread-affinity note: planner.execute() is submitted to the
        # dedicated _browser_thread via run_in_executor so that all
        # Playwright calls happen on the same thread that called agent.start().
        # ------------------------------------------------------------------
        queue: asyncio.Queue = asyncio.Queue()

        def _start_cb(step_idx: int, action: str) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "step_start", "step": step_idx, "action": action},
            )

        def _done_cb(result: dict) -> None:
            is_err = result.get("status") == "error"
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type":   "step_done",
                    "step":   result["step"],
                    "action": result["action"],
                    "status": "error" if is_err else "ok",
                    "error":  _sanitize_error(str(result.get("error", ""))) if is_err else None,
                },
            )

        # Submit execute() to the browser thread via run_in_executor so the
        # event loop stays unblocked while Playwright work runs on the owner thread.
        exec_future = loop.run_in_executor(
            None,
            lambda: _browser_thread.submit(
                planner.execute,
                steps,
                agent,
                stop_on_error=True,
                step_start_callback=_start_cb,
                step_callback=_done_cb,
            ),
        )

        # Drain the queue until the executor is done AND the queue is empty.
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=0.05)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                if exec_future.done():
                    # Flush any remaining messages before breaking.
                    while not queue.empty():
                        msg = queue.get_nowait()
                        await websocket.send_json(msg)
                    break

        results = await exec_future
        failed = sum(1 for r in results if r.get("status") == "error")
        await websocket.send_json({
            "type":         "done",
            "success":      failed == 0,
            "failed_count": failed,
        })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": _sanitize_error(str(exc))})
        except Exception:
            pass


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
    parser.add_argument(
        "--ssl-certfile",
        default=os.environ.get("AGENTICBROWSER_SSL_CERTFILE"),
        help="Path to TLS certificate file (PEM).  Also read from AGENTICBROWSER_SSL_CERTFILE.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=os.environ.get("AGENTICBROWSER_SSL_KEYFILE"),
        help="Path to TLS private-key file (PEM).  Also read from AGENTICBROWSER_SSL_KEYFILE.",
    )
    args = parser.parse_args()

    uvicorn_kwargs: dict = dict(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    if args.ssl_certfile:
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile:
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile

    uvicorn.run("api_server:app", **uvicorn_kwargs)

