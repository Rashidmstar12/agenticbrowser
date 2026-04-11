"""
API Server for the Agentic Browser.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Or use the helper:
    python api_server.py
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from browser_agent import BrowserAgent
from task_planner import TaskPlanner, StepValidationError, validate_steps, STEP_SCHEMA

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global agent instance (one per server process)
# ---------------------------------------------------------------------------
_agent: BrowserAgent | None = None
_planner: TaskPlanner | None = None


def get_agent() -> BrowserAgent:
    if _agent is None or _agent._page is None:
        raise HTTPException(status_code=400, detail="Browser session is not active. POST /session/start first.")
    return _agent


def get_planner() -> TaskPlanner:
    if _planner is None:
        raise HTTPException(status_code=400, detail="Browser session is not active. POST /session/start first.")
    return _planner


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
        "and natural-language task planning with zero hallucination."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    headless: bool = Field(True, description="Run browser without a visible window")
    slow_mo: int = Field(0, description="Slow down operations by this many ms (for debugging)")
    auto_close_popups: bool = Field(True, description="Auto-dismiss common popups on page load")
    default_timeout: int = Field(30_000, description="Default timeout in milliseconds")


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


class TaskRunRequest(BaseModel):
    intent: str = Field(..., description="Natural-language task, e.g. 'go to google and search python'")
    stop_on_error: bool = Field(True, description="Stop execution on the first failed step")


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
    global _agent, _planner
    if _agent is not None and _agent._page is not None:
        return {"status": "already_running", "headless": _agent.headless}
    _agent = BrowserAgent(
        headless=req.headless,
        slow_mo=req.slow_mo,
        auto_close_popups=req.auto_close_popups,
        default_timeout=req.default_timeout,
    )
    _agent.start()
    _planner = TaskPlanner()
    return {"status": "started", "headless": req.headless}


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
# Routes: task planner
# ---------------------------------------------------------------------------

@app.post("/task/run", summary="Plan and execute a natural-language browser task")
def task_run(req: TaskRunRequest) -> dict[str, Any]:
    """
    Convert *intent* to a step plan and execute it in one call.

    This is the primary endpoint for agentic use.  Common tasks (Google search,
    navigation, YouTube search, etc.) are resolved via deterministic templates
    — no LLM call, no hallucination.  For unknown intents the configured LLM
    backend is used with a constrained prompt.
    """
    result = get_planner().run(req.intent, get_agent(), stop_on_error=req.stop_on_error)
    if not result["success"]:
        raise HTTPException(status_code=422, detail=result)
    return result


@app.post("/task/plan", summary="Convert a natural-language intent to a step list (dry run)")
def task_plan(req: TaskPlanRequest) -> dict[str, Any]:
    """
    Return the planned steps WITHOUT executing them.
    Useful for previewing what would happen before running.
    """
    try:
        steps = get_planner().plan(req.intent)
        return {"intent": req.intent, "steps": steps, "count": len(steps)}
    except (ValueError, StepValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/task/execute", summary="Execute a pre-built step list")
def task_execute(req: TaskExecuteRequest) -> dict[str, Any]:
    """
    Run an already-built list of step dicts directly.
    Steps are re-validated before execution.
    """
    try:
        validated = validate_steps(req.steps)
    except StepValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    results = get_planner().execute(validated, get_agent(), stop_on_error=req.stop_on_error)
    failed  = [r for r in results if r["status"] == "error"]
    return {"success": len(failed) == 0, "results": results, "failed_count": len(failed)}


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
