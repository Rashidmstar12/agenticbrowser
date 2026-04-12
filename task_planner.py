"""
TaskPlanner: intent → validated workflow steps → execution.

Solves the hallucination problem in existing browser agents by:

1. Template matching  – common intents (Google search, navigate, login …) are
   resolved to hardcoded, verified step sequences.  No LLM needed.
2. Constrained LLM planning – for unknown intents the LLM receives a strict
   prompt that allows ONLY a JSON array of known action types.  Free-form text
   is rejected at the validation layer.
3. Schema validation  – every generated step is checked against STEP_SCHEMA
   before any browser interaction happens.
4. Execution with state verification – the result of each step is checked so
   that a wrong action cannot silently cascade into subsequent steps.

LLM backends (both optional, controlled by environment variables):
  - OpenAI  :  set OPENAI_API_KEY  (and optionally OPENAI_MODEL, default gpt-4o-mini)
  - Ollama  :  set OLLAMA_HOST     (default http://localhost:11434)
                and optionally OLLAMA_MODEL (default llama3)
  If neither is configured, only template matching is available.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step schema – the only allowed actions and their required / optional keys
# ---------------------------------------------------------------------------

STEP_SCHEMA: dict[str, dict[str, Any]] = {
    "navigate": {
        "required": ["url"],
        "optional": {"wait_until": "domcontentloaded"},
        "description": "Navigate the browser to a URL.",
    },
    "click": {
        "required": ["selector"],
        "optional": {"timeout": None},
        "description": "Click the element matching selector.",
    },
    "fill": {
        "required": ["selector", "value"],
        "optional": {},
        "description": "Fill an input field with a value (fast, no key events).",
    },
    "type": {
        "required": ["selector", "text"],
        "optional": {"clear_first": True},
        "description": "Type text into an input field (fires key events).",
    },
    "press": {
        "required": ["key"],
        "optional": {},
        "description": "Press a keyboard key (e.g. Enter, Tab, Escape).",
    },
    "wait_selector": {
        "required": ["selector"],
        "optional": {"timeout": None},
        "description": "Wait until the element matching selector is visible.",
    },
    "wait_state": {
        "required": [],
        "optional": {"state": "networkidle"},
        "description": "Wait for a page load state: load | domcontentloaded | networkidle.",
    },
    "close_popups": {
        "required": [],
        "optional": {},
        "description": "Dismiss common cookie banners, modals, and overlays.",
    },
    "scroll": {
        "required": [],
        "optional": {"x": 0, "y": 500},
        "description": "Scroll the page by x/y pixels.",
    },
    "scroll_to_element": {
        "required": ["selector"],
        "optional": {},
        "description": "Scroll an element into the visible viewport.",
    },
    "screenshot": {
        "required": [],
        "optional": {"path": None, "full_page": False},
        "description": "Capture a screenshot.",
    },
    "hover": {
        "required": ["selector"],
        "optional": {},
        "description": "Move the mouse over the element matching selector.",
    },
    "select_option": {
        "required": ["selector", "value"],
        "optional": {},
        "description": "Select an <option> in a <select> element.",
    },
    "evaluate": {
        "required": ["script"],
        "optional": {},
        "description": "Run arbitrary JavaScript in the page context.",
    },
    # ---- System actions (file I/O + code execution) ----
    "get_text": {
        "required": [],
        "optional": {"selector": "body"},
        "description": "Get the inner text of a page element and store it as {{last}}.",
    },
    "write_file": {
        "required": ["path", "content"],
        "optional": {"mode": "w"},
        "description": "Write content to a file in the workspace. Use {{last}} to reference the previous step's result.",
    },
    "append_file": {
        "required": ["path", "content"],
        "optional": {},
        "description": "Append content to a file in the workspace.",
    },
    "read_file": {
        "required": ["path"],
        "optional": {},
        "description": "Read a file from the workspace.",
    },
    "list_dir": {
        "required": [],
        "optional": {"path": "."},
        "description": "List files in the workspace directory.",
    },
    "run_python": {
        "required": ["code"],
        "optional": {"timeout": 30},
        "description": "Execute a Python code snippet. {{last}} is injected as 'last_result' variable.",
    },
    "run_shell": {
        "required": ["command"],
        "optional": {"timeout": 30},
        "description": "Execute a shell command in the workspace directory.",
    },
    # ---- Smart extraction actions ----
    "extract_links": {
        "required": [],
        "optional": {"selector": "a", "limit": 100},
        "description": "Extract all hyperlinks from the page. Returns list of {text, href}. Stored as {{last}} (JSON).",
    },
    "extract_table": {
        "required": [],
        "optional": {"selector": "table", "table_index": 0},
        "description": "Extract an HTML table as a list of row dicts keyed by header text.",
    },
    # ---- Assertion actions ----
    "assert_text": {
        "required": ["text"],
        "optional": {"selector": "body", "case_sensitive": False},
        "description": "Fail the task if the given text is NOT found in the element. Use to verify page state.",
    },
    "assert_url": {
        "required": ["pattern"],
        "optional": {},
        "description": "Fail the task if the current URL does not contain the literal pattern substring.",
    },
    # ---- Wait for dynamic content ----
    "wait_text": {
        "required": ["text"],
        "optional": {"selector": "body", "timeout": None},
        "description": "Wait until the given text appears in the element (polls until timeout).",
    },
    # ---- Cookie persistence ----
    "save_cookies": {
        "required": ["path"],
        "optional": {},
        "description": "Save all browser cookies to a JSON file in the workspace.",
    },
    "load_cookies": {
        "required": ["path"],
        "optional": {},
        "description": "Load cookies from a workspace JSON file into the browser context.",
    },
    # ---- Multi-tab actions ----
    "new_tab": {
        "required": [],
        "optional": {"url": None},
        "description": "Open a new browser tab (optionally navigate to url). Makes the new tab active.",
    },
    "switch_tab": {
        "required": ["index"],
        "optional": {},
        "description": "Switch the active browser tab to the one at the given index.",
    },
    "close_tab": {
        "required": [],
        "optional": {"index": None},
        "description": "Close a browser tab by index (default: current active tab).",
    },
    "list_tabs": {
        "required": [],
        "optional": {},
        "description": "Return info about all open tabs (index, url, title, active).",
    },
    # ---- New browser interactions (Category 1) ----
    "drag_drop": {
        "required": ["source", "target"],
        "optional": {},
        "description": "Drag the element matching source and drop it onto target.",
    },
    "right_click": {
        "required": ["selector"],
        "optional": {},
        "description": "Right-click an element to open its context menu.",
    },
    "double_click": {
        "required": ["selector"],
        "optional": {},
        "description": "Double-click an element.",
    },
    "upload_file": {
        "required": ["selector", "path"],
        "optional": {},
        "description": "Attach a local file to an <input type='file'> element.",
    },
    "set_viewport": {
        "required": ["width", "height"],
        "optional": {},
        "description": "Resize the browser viewport to the given width and height in pixels.",
    },
    "block_resource": {
        "required": [],
        "optional": {"types": ["image", "stylesheet", "font"]},
        "description": "Block requests for the given resource types (image, stylesheet, font, script, media).",
    },
    "iframe_switch": {
        "required": ["selector"],
        "optional": {},
        "description": "Switch the interaction context to the <iframe> matching selector.",
    },
    "iframe_exit": {
        "required": [],
        "optional": {},
        "description": "Exit the current iframe context and return to the top-level page.",
    },
    # ---- Data extraction (Category 2) ----
    "extract_json_ld": {
        "required": [],
        "optional": {},
        "description": "Extract all Schema.org JSON-LD metadata blocks from the page.",
    },
    "extract_headings": {
        "required": [],
        "optional": {},
        "description": "Extract all headings (h1–h6) as a structured outline.",
    },
    "extract_images": {
        "required": [],
        "optional": {"selector": "img", "limit": 100},
        "description": "Extract all images from the page (src, alt, width, height).",
    },
    "extract_form_fields": {
        "required": [],
        "optional": {"selector": "form"},
        "description": "Describe all interactive form fields on the page.",
    },
    "extract_meta": {
        "required": [],
        "optional": {},
        "description": "Extract <meta> tags including title, description, og:* and twitter:* tags.",
    },
    # ---- Authentication & Session (Category 3) ----
    "set_extra_headers": {
        "required": ["headers"],
        "optional": {},
        "description": "Inject extra HTTP request headers (dict) for all subsequent requests.",
    },
    "http_auth": {
        "required": ["username", "password"],
        "optional": {},
        "description": "Set HTTP Basic Auth credentials for all subsequent requests.",
    },
    "local_storage_set": {
        "required": ["key", "value"],
        "optional": {},
        "description": "Write a key-value pair to the page's localStorage.",
    },
    "local_storage_get": {
        "required": ["key"],
        "optional": {},
        "description": "Read a value from the page's localStorage. Stored as {{last}}.",
    },
    "session_storage_set": {
        "required": ["key", "value"],
        "optional": {},
        "description": "Write a key-value pair to the page's sessionStorage.",
    },
    "session_storage_get": {
        "required": ["key"],
        "optional": {},
        "description": "Read a value from the page's sessionStorage. Stored as {{last}}.",
    },
    # ---- Assertions & Verification (Category 4) ----
    "assert_element_count": {
        "required": ["selector", "count"],
        "optional": {"operator": "eq"},
        "description": "Assert the number of elements matching selector. operator: eq|gte|lte|gt|lt.",
    },
    "assert_attribute": {
        "required": ["selector", "attribute", "value"],
        "optional": {"case_sensitive": True},
        "description": "Assert that an element's HTML attribute equals the expected value.",
    },
    "assert_title": {
        "required": ["pattern"],
        "optional": {"case_sensitive": False},
        "description": "Assert that the page title contains pattern as a substring.",
    },
    "assert_visible": {
        "required": ["selector"],
        "optional": {},
        "description": "Assert that the element matching selector is visible on the page.",
    },
    "assert_hidden": {
        "required": ["selector"],
        "optional": {},
        "description": "Assert that the element matching selector is NOT visible (or absent).",
    },
}

# ---------------------------------------------------------------------------
# Well-known selectors for popular sites
# (avoids LLM guessing wrong selectors for common sites)
# ---------------------------------------------------------------------------

_KNOWN_SELECTORS: dict[str, dict[str, str]] = {
    "google.com": {
        "search_input":  "textarea[name='q']",
        "search_button": "input[name='btnK']",
        "results":       "#search",
    },
    "bing.com": {
        "search_input":  "input#sb_form_q",
        "search_button": "input#search_icon",
        "results":       "#b_results",
    },
    "duckduckgo.com": {
        "search_input":  "input[name='q']",
        "search_button": "button[type='submit']",
        "results":       "#links",
    },
    "youtube.com": {
        "search_input":  "input#search",
        "search_button": "button#search-icon-legacy",
        "results":       "ytd-video-renderer",
    },
    "wikipedia.org": {
        "search_input":  "input#searchInput",
        "search_button": "button[type='submit']",
        "results":       "#mw-content-text",
    },
}


def _selectors_for(host: str) -> dict[str, str]:
    for domain, sels in _KNOWN_SELECTORS.items():
        if domain in host:
            return sels
    return {}


# ---------------------------------------------------------------------------
# Workflow templates
# Each template is (regex_pattern, builder_fn(match, intent) -> list[step])
# Templates are tried in order; first match wins.
# ---------------------------------------------------------------------------

def _google_search_steps(query: str) -> list[dict[str, Any]]:
    sels = _KNOWN_SELECTORS["google.com"]
    return [
        {"action": "navigate",      "url": "https://www.google.com", "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "wait_selector", "selector": sels["search_input"]},
        {"action": "fill",          "selector": sels["search_input"], "value": query},
        {"action": "press",         "key": "Enter"},
        {"action": "wait_state",    "state": "networkidle"},
    ]


def _bing_search_steps(query: str) -> list[dict[str, Any]]:
    sels = _KNOWN_SELECTORS["bing.com"]
    return [
        {"action": "navigate",      "url": "https://www.bing.com", "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "wait_selector", "selector": sels["search_input"]},
        {"action": "fill",          "selector": sels["search_input"], "value": query},
        {"action": "press",         "key": "Enter"},
        {"action": "wait_state",    "state": "networkidle"},
    ]


def _ddg_search_steps(query: str) -> list[dict[str, Any]]:
    sels = _KNOWN_SELECTORS["duckduckgo.com"]
    return [
        {"action": "navigate",      "url": "https://duckduckgo.com", "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "fill",          "selector": sels["search_input"], "value": query},
        {"action": "press",         "key": "Enter"},
        {"action": "wait_state",    "state": "networkidle"},
    ]


def _youtube_search_steps(query: str) -> list[dict[str, Any]]:
    sels = _KNOWN_SELECTORS["youtube.com"]
    return [
        {"action": "navigate",      "url": "https://www.youtube.com", "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "wait_selector", "selector": sels["search_input"]},
        {"action": "fill",          "selector": sels["search_input"], "value": query},
        {"action": "press",         "key": "Enter"},
        {"action": "wait_state",    "state": "networkidle"},
    ]


def _wikipedia_search_steps(query: str) -> list[dict[str, Any]]:
    sels = _KNOWN_SELECTORS["wikipedia.org"]
    return [
        {"action": "navigate",      "url": "https://en.wikipedia.org", "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "fill",          "selector": sels["search_input"], "value": query},
        {"action": "press",         "key": "Enter"},
        {"action": "wait_state",    "state": "networkidle"},
    ]


def _navigate_steps(url: str) -> list[dict[str, Any]]:
    if not re.match(r"https?://", url):
        url = "https://" + url
    return [
        {"action": "navigate",   "url": url, "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "wait_state", "state": "networkidle"},
    ]


# Pattern → builder mapping.
# Each entry: (compiled_regex, fn(match) -> list[step])
_TEMPLATES: list[tuple[re.Pattern[str], Any]] = [
    # "search <query> on google" / "google search <query>"
    (
        re.compile(
            r"(?:search\s+(.+?)\s+on\s+google|google\s+(?:search\s+)?(.+)|"
            r"(?:go\s+to\s+google\s+and\s+)?search\s+(?:for\s+)?(.+?)\s+(?:on|using|with|in)\s+google"
            r"|(?:use\s+google\s+to\s+)?(?:look\s+up|find|search\s+for?)\s+(.+?)(?:\s+on\s+google)?$)",
            re.IGNORECASE,
        ),
        lambda m: _google_search_steps(next(g for g in m.groups() if g)),
    ),
    # Standalone "go to google and search ..." / "open google and search ..."
    (
        re.compile(
            r"(?:go\s+to|open|visit)\s+google\s+and\s+search\s+(?:for\s+)?(.+)",
            re.IGNORECASE,
        ),
        lambda m: _google_search_steps(m.group(1)),
    ),
    # Bing search
    (
        re.compile(
            r"(?:search\s+(.+?)\s+on\s+bing|bing\s+(?:search\s+)?(.+)|"
            r"(?:go\s+to\s+bing\s+and\s+)?search\s+(?:for\s+)?(.+?)\s+on\s+bing)",
            re.IGNORECASE,
        ),
        lambda m: _bing_search_steps(next(g for g in m.groups() if g)),
    ),
    # DuckDuckGo search
    (
        re.compile(
            r"(?:search\s+(.+?)\s+on\s+duckduckgo|duckduckgo\s+(?:search\s+)?(.+)|"
            r"(?:go\s+to\s+duckduckgo\s+and\s+)?search\s+(?:for\s+)?(.+?)\s+on\s+duckduckgo)",
            re.IGNORECASE,
        ),
        lambda m: _ddg_search_steps(next(g for g in m.groups() if g)),
    ),
    # YouTube search
    (
        re.compile(
            r"(?:search\s+(.+?)\s+on\s+youtube|youtube\s+(?:search\s+)?(.+)|"
            r"(?:go\s+to\s+youtube\s+and\s+)?search\s+(?:for\s+)?(.+?)\s+on\s+youtube|"
            r"(?:find|look\s+up)\s+(.+?)\s+(?:on\s+)?youtube)",
            re.IGNORECASE,
        ),
        lambda m: _youtube_search_steps(next(g for g in m.groups() if g)),
    ),
    # Wikipedia search
    (
        re.compile(
            r"(?:search\s+(.+?)\s+on\s+wikipedia|wikipedia\s+(?:search\s+)?(.+)|"
            r"(?:go\s+to\s+wikipedia\s+and\s+)?search\s+(?:for\s+)?(.+?)\s+on\s+wikipedia|"
            r"(?:look\s+up|find)\s+(.+?)\s+(?:on\s+)?wikipedia)",
            re.IGNORECASE,
        ),
        lambda m: _wikipedia_search_steps(next(g for g in m.groups() if g)),
    ),
    # Generic "open / go to / navigate to / visit <url or site>"
    (
        re.compile(
            r"(?:open|go\s+to|navigate\s+to|visit|load)\s+(https?://\S+|\S+\.\S+)",
            re.IGNORECASE,
        ),
        lambda m: _navigate_steps(m.group(1)),
    ),
    # "collect/scrape/get/grab text from <url> and save to <file>"
    (
        re.compile(
            r"(?:collect|scrape|get|grab|fetch|extract)\s+(?:the\s+)?(?:text|content|info|information|data)\s+"
            r"(?:from\s+)?(https?://\S+|\S+\.\S+)\s+and\s+(?:save|write|store)\s+(?:it\s+)?(?:to|in)\s+(\S+)",
            re.IGNORECASE,
        ),
        lambda m: [
            {"action": "navigate",   "url": m.group(1) if re.match(r"https?://", m.group(1)) else "https://" + m.group(1),
             "wait_until": "domcontentloaded"},
            {"action": "close_popups"},
            {"action": "wait_state", "state": "networkidle"},
            {"action": "get_text",   "selector": "body"},
            {"action": "write_file", "path": m.group(2), "content": "{{last}}"},
        ],
    ),
    # "open <url>, collect page text, save to <file>"
    (
        re.compile(
            r"(?:open|go\s+to|navigate\s+to|visit)\s+(https?://\S+|\S+\.\S+)"
            r".*?(?:collect|save|store|write).*?(?:to|in)\s+(\S+\.(?:txt|csv|json|md|html))",
            re.IGNORECASE | re.DOTALL,
        ),
        lambda m: [
            {"action": "navigate",   "url": m.group(1) if re.match(r"https?://", m.group(1)) else "https://" + m.group(1),
             "wait_until": "domcontentloaded"},
            {"action": "close_popups"},
            {"action": "wait_state", "state": "networkidle"},
            {"action": "get_text",   "selector": "body"},
            {"action": "write_file", "path": m.group(2), "content": "{{last}}"},
        ],
    ),
]

# ---------------------------------------------------------------------------
# LLM prompt for unknown intents
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a browser automation planner. Your ONLY job is to convert a
natural-language browser task into a minimal JSON step array.

RULES (MUST follow all):
1. Output ONLY a valid JSON array. No prose, no markdown, no explanation.
2. Each element must have an "action" key matching one of the allowed actions below.
3. Use the fewest steps possible. Never add unnecessary steps.
4. Maximum 12 steps. If you cannot do the task in 12 steps, return an error step.
5. For well-known sites (Google, Bing, YouTube, Wikipedia, DuckDuckGo) use the
   exact selectors listed below — never invent selectors.
6. After navigate always add close_popups.
7. Never add steps to "verify" or "confirm" — just do the task.

ALLOWED ACTIONS (schema):
""" + json.dumps(STEP_SCHEMA, indent=2) + """

KNOWN SELECTORS:
""" + json.dumps(_KNOWN_SELECTORS, indent=2) + """

FEW-SHOT EXAMPLES:

Task: "go to google and search python tutorials"
Output:
[
  {"action": "navigate", "url": "https://www.google.com", "wait_until": "domcontentloaded"},
  {"action": "close_popups"},
  {"action": "wait_selector", "selector": "textarea[name='q']"},
  {"action": "fill", "selector": "textarea[name='q']", "value": "python tutorials"},
  {"action": "press", "key": "Enter"},
  {"action": "wait_state", "state": "networkidle"}
]

Task: "open https://news.ycombinator.com"
Output:
[
  {"action": "navigate", "url": "https://news.ycombinator.com", "wait_until": "domcontentloaded"},
  {"action": "close_popups"},
  {"action": "wait_state", "state": "networkidle"}
]

Task: "search for machine learning on YouTube"
Output:
[
  {"action": "navigate", "url": "https://www.youtube.com", "wait_until": "domcontentloaded"},
  {"action": "close_popups"},
  {"action": "wait_selector", "selector": "input#search"},
  {"action": "fill", "selector": "input#search", "value": "machine learning"},
  {"action": "press", "key": "Enter"},
  {"action": "wait_state", "state": "networkidle"}
]

Now output ONLY the JSON array for the task below. Nothing else.
"""


# ---------------------------------------------------------------------------
# Step validation
# ---------------------------------------------------------------------------

class StepValidationError(ValueError):
    pass


# Universal optional keys that apply to every step.
# They are preserved by validate_steps and consumed by execute().
_UNIVERSAL_STEP_KEYS: dict[str, Any] = {
    "retry":       0,    # number of extra attempts after a failure
    "retry_delay": 1.0,  # seconds to wait between retry attempts
}


def validate_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate and normalise a list of step dicts against STEP_SCHEMA.

    Raises StepValidationError if a step is invalid.
    Returns the normalised (with defaults applied) step list.
    """
    if not isinstance(steps, list):
        raise StepValidationError("Plan must be a JSON array of steps.")
    if len(steps) == 0:
        raise StepValidationError("Plan must contain at least one step.")
    if len(steps) > 20:
        raise StepValidationError(f"Plan has {len(steps)} steps; maximum is 20 to prevent runaway execution.")

    validated: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise StepValidationError(f"Step {i} is not an object: {step!r}")
        action = step.get("action")
        if not action:
            raise StepValidationError(f"Step {i} is missing the 'action' key.")
        if action not in STEP_SCHEMA:
            raise StepValidationError(
                f"Step {i}: unknown action {action!r}. Allowed: {list(STEP_SCHEMA)}"
            )
        schema = STEP_SCHEMA[action]
        for req in schema["required"]:
            if req not in step:
                raise StepValidationError(
                    f"Step {i} ({action!r}): missing required key {req!r}."
                )
        # Build normalised step with defaults
        normalised: dict[str, Any] = {"action": action}
        for key in schema["required"]:
            normalised[key] = step[key]
        for key, default in schema["optional"].items():
            normalised[key] = step.get(key, default)
        # Preserve universal keys (retry, retry_delay)
        for key, default in _UNIVERSAL_STEP_KEYS.items():
            normalised[key] = step.get(key, default)
        validated.append(normalised)

    return validated


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_openai(intent: str) -> list[dict[str, Any]]:
    """Call OpenAI chat completions API and return validated steps."""
    import openai  # type: ignore[import]

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": intent},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""
    logger.debug("OpenAI raw response: %s", raw)

    # The response_format=json_object wrapper may wrap the array in {"steps": [...]}
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return validate_steps(parsed)
    for key in ("steps", "plan", "actions", "workflow"):
        if key in parsed and isinstance(parsed[key], list):
            return validate_steps(parsed[key])
    raise StepValidationError(f"OpenAI returned unexpected JSON shape: {list(parsed)}")


def _call_ollama(intent: str) -> list[dict[str, Any]]:
    """Call a local Ollama instance and return validated steps."""
    import urllib.request

    host  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3")

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": intent},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode()

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    raw = body.get("message", {}).get("content", "")
    logger.debug("Ollama raw response: %s", raw)

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Extract the first JSON array from the response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise StepValidationError(f"Ollama response contains no JSON array: {raw[:300]}")

    steps = json.loads(match.group())
    return validate_steps(steps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate_last(step: dict[str, Any], last: str) -> dict[str, Any]:
    """
    Return a copy of *step* with the token ``{{last}}`` replaced by *last*
    in every string value.

    This lets steps reference the text output of a previous step without
    the LLM having to hard-code values.  For example::

        {"action": "write_file", "path": "out.txt", "content": "{{last}}"}

    writes the text extracted by the preceding ``get_text`` step.
    """
    return {
        k: v.replace("{{last}}", last) if isinstance(v, str) else v
        for k, v in step.items()
    }


# ---------------------------------------------------------------------------
# TaskPlanner
# ---------------------------------------------------------------------------

class TaskPlanner:
    """
    Convert a natural-language intent into a validated, executable step list
    and run it against a BrowserAgent.

    Parameters
    ----------
    llm : str | None
        LLM backend to use for intents that don't match any template.
        ``"openai"`` – requires ``OPENAI_API_KEY`` env var.
        ``"ollama"`` – requires a running Ollama instance (``OLLAMA_HOST``).
        ``None``     – auto-detect from environment; templates only if no LLM.
    """

    def __init__(self, llm: str | None = None, skill_registry: Any = None) -> None:
        self.llm = llm or self._detect_llm()
        self._system_tools: Any = None  # lazy-created on first system action
        # skill_registry is a SkillRegistry instance (or None to use the default).
        # Typed as Any to avoid a hard import cycle at module load time.
        self._skill_registry: Any = skill_registry

    @staticmethod
    def _detect_llm() -> str | None:
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("OLLAMA_HOST") or _ollama_running():
            return "ollama"
        return None

    def _get_system_tools(self) -> Any:
        """Return (and lazily create) the shared SystemTools instance."""
        if self._system_tools is None:
            from system_tools import SystemTools  # local import avoids hard dep
            self._system_tools = SystemTools()
        return self._system_tools

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, intent: str) -> list[dict[str, Any]]:
        """
        Convert *intent* to a validated list of browser steps.

        Template matching is tried first.  If no template matches, the
        configured LLM is called.  If no LLM is available, a ValueError
        is raised with a helpful message.
        """
        intent = intent.strip()
        logger.info("Planning intent: %r", intent)

        # 0. Skill registry (user-loaded external skills take highest priority)
        registry = self._skill_registry
        if registry is None:
            try:
                from skills import get_default_registry
                registry = get_default_registry()
            except ImportError:
                registry = None
        if registry is not None:
            match = registry.match(intent)
            if match is not None:
                skill, params = match
                try:
                    from skills import resolve_steps
                    raw_steps = resolve_steps(skill, params)
                    steps = validate_steps(raw_steps)
                    logger.info("Matched skill %r → %d steps", skill.name, len(steps))
                    return steps
                except Exception as exc:
                    logger.warning("Skill %r builder failed: %s", skill.name, exc)

        # 1. Template matching (no LLM, no hallucination)
        steps = self._match_template(intent)
        if steps is not None:
            logger.info("Matched template → %d steps", len(steps))
            return steps

        # 2. LLM planning
        if self.llm == "openai":
            logger.info("Using OpenAI planner")
            return _call_openai(intent)
        if self.llm == "ollama":
            logger.info("Using Ollama planner")
            return _call_ollama(intent)

        raise ValueError(
            f"No template matched for intent {intent!r} and no LLM is configured.\n"
            "Set OPENAI_API_KEY for OpenAI or OLLAMA_HOST for a local Ollama instance.\n"
            "Alternatively, use one of the supported built-in intents:\n"
            "  • go to google and search <query>\n"
            "  • search <query> on bing / duckduckgo / youtube / wikipedia\n"
            "  • open / go to / navigate to <url>"
        )

    def execute(
        self,
        steps: list[dict[str, Any]],
        agent: Any,  # BrowserAgent (avoid circular import)
        *,
        stop_on_error: bool = True,
        step_callback: Any = None,
        step_start_callback: Any = None,
    ) -> list[dict[str, Any]]:
        """
        Execute *steps* against *agent*, returning a result record for each step.

        Parameters
        ----------
        stop_on_error:
            If ``True`` (default), execution stops on the first failed step.
            If ``False``, failures are recorded but execution continues.
        step_start_callback:
            Optional callable ``(step_index: int, action: str) -> None`` invoked
            immediately *before* each step begins.  Useful for real-time progress
            reporting (e.g. WebSocket streaming).
        step_callback:
            Optional callable ``(result: dict) -> None`` invoked immediately
            *after* each step finishes (both ok and error).  The dict has the
            same shape as the elements of the returned list.

        Notes
        -----
        The special token ``{{last}}`` in any string field of a step is
        replaced with the text output of the most recent step that produces
        one (``get_text``, ``read_file``, ``run_python``, ``run_shell``,
        ``extract_links``, ``extract_table``).

        Each step may include the universal keys ``retry`` (int, default 0)
        and ``retry_delay`` (float seconds, default 1.0) to automatically
        re-attempt failed steps before marking them as errors.
        """
        import time as _time

        results: list[dict[str, Any]] = []
        last: str = ""
        for i, step in enumerate(steps):
            action = step["action"]
            logger.info("Step %d/%d: %s", i + 1, len(steps), action)
            # Notify caller that this step is about to start.
            if step_start_callback is not None:
                step_start_callback(i, action)
            # Substitute {{last}} in all string values of this step.
            if last:
                step = _interpolate_last(step, last)

            retry_count = int(step.get("retry", 0))
            retry_delay = float(step.get("retry_delay", 1.0))
            last_exc: Exception | None = None

            for attempt in range(retry_count + 1):
                try:
                    result = self._execute_step(agent, step, last=last)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < retry_count:
                        logger.warning(
                            "Step %d (%s) attempt %d/%d failed (%s); retrying in %.1fs",
                            i, action, attempt + 1, retry_count + 1, exc, retry_delay,
                        )
                        _time.sleep(retry_delay)

            if last_exc is not None:
                logger.error("Step %d (%s) failed after %d attempt(s): %s", i, action, retry_count + 1, last_exc)
                err_record: dict[str, Any] = {"step": i, "action": action, "status": "error", "error": str(last_exc)}
                results.append(err_record)
                if step_callback is not None:
                    step_callback(err_record)
                if stop_on_error:
                    break
                continue

            # Update {{last}} from steps that produce text output.
            if action == "get_text":
                last = result if isinstance(result, str) else str(result)
            elif action == "read_file" and isinstance(result, dict):
                last = result.get("content", "")
            elif action in ("run_python", "run_shell") and isinstance(result, dict):
                last = result.get("stdout", "")
            elif action in (
                "extract_links", "extract_table", "extract_json_ld",
                "extract_headings", "extract_images", "extract_form_fields",
                "extract_meta",
            ) and isinstance(result, dict):
                import json as _json
                last = _json.dumps(result, ensure_ascii=False)
            elif action in ("local_storage_get", "session_storage_get") and isinstance(result, dict):
                last = result.get("value") or ""
            ok_record: dict[str, Any] = {"step": i, "action": action, "status": "ok", "result": result}
            results.append(ok_record)
            if step_callback is not None:
                step_callback(ok_record)
        return results

    def run(
        self,
        intent: str,
        agent: Any,
        *,
        stop_on_error: bool = True,
        log_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Plan *intent* and execute the resulting steps against *agent*.

        Returns a summary dict with ``steps``, ``results``, and ``success`` flag.

        Parameters
        ----------
        log_path:
            Optional workspace-relative path (e.g. ``"logs/run.json"``) where
            the full execution log (steps + results + timestamp) is saved.
        """
        from datetime import datetime as _dt

        try:
            steps = self.plan(intent)
        except (ValueError, StepValidationError) as exc:
            return {"success": False, "error": str(exc), "steps": [], "results": []}

        results = self.execute(steps, agent, stop_on_error=stop_on_error)
        failed  = [r for r in results if r["status"] == "error"]
        summary = {
            "success":      len(failed) == 0,
            "intent":       intent,
            "steps":        steps,
            "results":      results,
            "failed_count": len(failed),
        }

        if log_path:
            import json as _json
            from datetime import timezone as _tz
            st = self._get_system_tools()
            log_entry = {**summary, "timestamp": _dt.now(_tz.utc).isoformat()}
            try:
                st.write_file(log_path, _json.dumps(log_entry, indent=2, default=str))
                logger.info("Execution log saved to %s", log_path)
            except Exception as exc:
                logger.warning("Could not save execution log to %r: %s", log_path, exc)

        return summary

    # ------------------------------------------------------------------
    # Template matching
    # ------------------------------------------------------------------

    def _match_template(self, intent: str) -> list[dict[str, Any]] | None:
        for pattern, builder in _TEMPLATES:
            m = pattern.fullmatch(intent.strip()) or pattern.search(intent.strip())
            if m:
                try:
                    steps = builder(m)
                    return validate_steps(steps)
                except Exception as exc:
                    logger.warning("Template builder failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Step executor
    # ------------------------------------------------------------------

    def _execute_step(self, agent: Any, step: dict[str, Any], *, last: str = "") -> Any:
        action = step["action"]

        if action == "navigate":
            return agent.navigate(step["url"], wait_until=step.get("wait_until", "domcontentloaded"))

        if action == "click":
            return agent.click(step["selector"], timeout=step.get("timeout"))

        if action == "fill":
            return agent.fill(step["selector"], step["value"])

        if action == "type":
            return agent.type_text(
                step["selector"],
                step["text"],
                clear_first=step.get("clear_first", True),
            )

        if action == "press":
            return agent.press_key(step["key"])

        if action == "wait_selector":
            return agent.wait_for_selector(step["selector"], timeout=step.get("timeout"))

        if action == "wait_state":
            return agent.wait_for_load_state(step.get("state", "networkidle"))

        if action == "close_popups":
            return agent.close_popups()

        if action == "scroll":
            return agent.scroll(x=step.get("x", 0), y=step.get("y", 500))

        if action == "scroll_to_element":
            return agent.scroll_to_element(step["selector"])

        if action == "screenshot":
            return agent.screenshot(
                path=step.get("path"),
                full_page=step.get("full_page", False),
                as_base64=step.get("path") is None,
            )

        if action == "hover":
            return agent.hover(step["selector"])

        if action == "select_option":
            return agent.select_option(step["selector"], step["value"])

        if action == "evaluate":
            return agent.evaluate(step["script"])

        if action == "get_text":
            return agent.get_text(step.get("selector", "body"))

        # ---- Smart extraction ----

        if action == "extract_links":
            return agent.extract_links(
                selector=step.get("selector", "a"),
                limit=step.get("limit", 100),
            )

        if action == "extract_table":
            return agent.extract_table(
                selector=step.get("selector", "table"),
                table_index=step.get("table_index", 0),
            )

        # ---- Assertions ----

        if action == "assert_text":
            return agent.assert_text(
                step["text"],
                selector=step.get("selector", "body"),
                case_sensitive=step.get("case_sensitive", False),
            )

        if action == "assert_url":
            return agent.assert_url(step["pattern"])

        # ---- Wait for dynamic content ----

        if action == "wait_text":
            return agent.wait_text(
                step["text"],
                selector=step.get("selector", "body"),
                timeout=step.get("timeout"),
            )

        # ---- Multi-tab ----

        if action == "new_tab":
            return agent.new_tab(url=step.get("url"))

        if action == "switch_tab":
            return agent.switch_tab(step["index"])

        if action == "close_tab":
            return agent.close_tab(index=step.get("index"))

        if action == "list_tabs":
            return agent.list_tabs()

        # ---- System actions (file I/O + code execution) ----

        st = self._get_system_tools()

        if action == "write_file":
            return st.write_file(step["path"], step["content"], mode=step.get("mode", "w"))

        if action == "append_file":
            return st.append_file(step["path"], step["content"])

        if action == "read_file":
            return st.read_file(step["path"])

        if action == "list_dir":
            return st.list_dir(step.get("path", "."))

        if action == "run_python":
            # Inject the previous step's text output as `last_result` variable.
            extra: dict[str, Any] = {"last_result": last} if last else {}
            return st.run_python(step["code"], timeout=step.get("timeout"), extra_vars=extra or None)

        if action == "run_shell":
            return st.run_shell(step["command"], timeout=step.get("timeout"))

        # ---- Cookie persistence ----

        if action == "save_cookies":
            import json as _json
            cookies = agent.get_cookies()
            st.write_file(step["path"], _json.dumps(cookies, indent=2))
            return {"cookies_saved": len(cookies), "path": step["path"]}

        if action == "load_cookies":
            import json as _json
            result = st.read_file(step["path"])
            cookies = _json.loads(result["content"])
            agent.add_cookies(cookies)
            return {"cookies_loaded": len(cookies), "path": step["path"]}

        # ---- New browser interactions (Category 1) ----

        if action == "drag_drop":
            return agent.drag_drop(step["source"], step["target"])

        if action == "right_click":
            return agent.right_click(step["selector"])

        if action == "double_click":
            return agent.double_click(step["selector"])

        if action == "upload_file":
            return agent.upload_file(step["selector"], step["path"])

        if action == "set_viewport":
            return agent.set_viewport(step["width"], step["height"])

        if action == "block_resource":
            return agent.block_resource(types=step.get("types"))

        if action == "iframe_switch":
            return agent.iframe_switch(step["selector"])

        if action == "iframe_exit":
            return agent.iframe_exit()

        # ---- Data extraction (Category 2) ----

        if action == "extract_json_ld":
            return agent.extract_json_ld()

        if action == "extract_headings":
            return agent.extract_headings()

        if action == "extract_images":
            return agent.extract_images(
                selector=step.get("selector", "img"),
                limit=step.get("limit", 100),
            )

        if action == "extract_form_fields":
            return agent.extract_form_fields(selector=step.get("selector", "form"))

        if action == "extract_meta":
            return agent.extract_meta()

        # ---- Authentication & Session (Category 3) ----

        if action == "set_extra_headers":
            return agent.set_extra_headers(step["headers"])

        if action == "http_auth":
            return agent.http_auth(step["username"], step["password"])

        if action == "local_storage_set":
            return agent.local_storage_set(step["key"], step["value"])

        if action == "local_storage_get":
            return agent.local_storage_get(step["key"])

        if action == "session_storage_set":
            return agent.session_storage_set(step["key"], step["value"])

        if action == "session_storage_get":
            return agent.session_storage_get(step["key"])

        # ---- Assertions & Verification (Category 4) ----

        if action == "assert_element_count":
            return agent.assert_element_count(
                step["selector"],
                step["count"],
                operator=step.get("operator", "eq"),
            )

        if action == "assert_attribute":
            return agent.assert_attribute(
                step["selector"],
                step["attribute"],
                step["value"],
                case_sensitive=step.get("case_sensitive", True),
            )

        if action == "assert_title":
            return agent.assert_title(
                step["pattern"],
                case_sensitive=step.get("case_sensitive", False),
            )

        if action == "assert_visible":
            return agent.assert_visible(step["selector"])

        if action == "assert_hidden":
            return agent.assert_hidden(step["selector"])

        raise ValueError(f"Unknown action: {action!r}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ollama_running() -> bool:
    """Return True if a local Ollama server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False
