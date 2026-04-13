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
        "optional": {"framework_safe": True},
        "description": (
            "Fill an input field. By default fires key events via type_text "
            "(safe for React/Vue/Angular). Set framework_safe: false to use raw "
            "Playwright fill for plain server-rendered HTML forms."
        ),
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
    "start_recording": {
        "required": ["path"],
        "optional": {},
        "description": "Start recording the browser session as a WebM video. path is the workspace-relative output file (e.g. 'recording.webm').",
    },
    "stop_recording": {
        "required": [],
        "optional": {},
        "description": "Stop the video recording started by start_recording and save the file.",
    },
    "record_gif": {
        "required": ["path"],
        "optional": {"duration": 3.0, "fps": 2},
        "description": "Capture an animated GIF of the current page. path is workspace-relative. duration: seconds to record (default 3.0). fps: frames per second (default 2).",
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
        "optional": {"selector": "a", "limit": 100, "wait_for_ready": True, "timeout": 5000},
        "description": "Extract all hyperlinks from the page. Returns list of {text, href}. Stored as {{last}} (JSON). When selector is not the default 'a', waits for the selector to be present before extracting (wait_for_ready: true by default).",
    },
    "extract_table": {
        "required": [],
        "optional": {"selector": "table", "table_index": 0, "wait_for_ready": True, "timeout": 5000},
        "description": "Extract an HTML table as a list of row dicts keyed by header text. When selector is not the default 'table', waits for the selector to be present before extracting (wait_for_ready: true by default).",
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
    # ---- High-priority advanced interactions ----
    "upload_file": {
        "required": ["selector", "path"],
        "optional": {},
        "description": "Set file(s) on a <input type='file'> element. path is workspace-relative. Separate multiple files with '|'.",
    },
    "download_file": {
        "required": ["url", "path"],
        "optional": {},
        "description": "Navigate to url and save the triggered download to a workspace-relative path.",
    },
    "drag_drop": {
        "required": ["source", "target"],
        "optional": {},
        "description": "Drag the element matching source selector and drop it onto the target selector.",
    },
    "right_click": {
        "required": ["selector"],
        "optional": {},
        "description": "Right-click (context-menu click) on the element matching selector.",
    },
    "double_click": {
        "required": ["selector"],
        "optional": {},
        "description": "Double-click on the element matching selector.",
    },
    "get_rect": {
        "required": ["selector"],
        "optional": {},
        "description": "Return the bounding box (x, y, width, height) of the first element matching selector.",
    },
    "set_network_intercept": {
        "required": ["url_pattern"],
        "optional": {"intercept_action": "abort"},
        "description": "Intercept requests matching url_pattern. intercept_action: 'abort' (block) or 'continue' (pass through).",
    },
    "clear_network_intercepts": {
        "required": [],
        "optional": {},
        "description": "Remove all network intercept routes set by set_network_intercept.",
    },
    "set_viewport": {
        "required": ["width", "height"],
        "optional": {},
        "description": "Resize the browser viewport to width x height pixels.",
    },
    "set_geolocation": {
        "required": ["latitude", "longitude"],
        "optional": {"accuracy": 10.0},
        "description": "Override the browser geolocation. latitude/longitude in decimal degrees.",
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
        {"action": "wait_selector", "selector": sels["results"]},
        {"action": "assert_text",   "text": query, "selector": sels["results"]},
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
        {"action": "wait_selector", "selector": sels["results"]},
        {"action": "assert_text",   "text": query, "selector": sels["results"]},
    ]


def _ddg_search_steps(query: str) -> list[dict[str, Any]]:
    sels = _KNOWN_SELECTORS["duckduckgo.com"]
    return [
        {"action": "navigate",      "url": "https://duckduckgo.com", "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "fill",          "selector": sels["search_input"], "value": query},
        {"action": "press",         "key": "Enter"},
        {"action": "wait_state",    "state": "networkidle"},
        {"action": "wait_selector", "selector": sels["results"]},
        {"action": "assert_text",   "text": query, "selector": sels["results"]},
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
        {"action": "wait_selector", "selector": sels["results"]},
        {"action": "assert_text",   "text": query, "selector": sels["results"]},
    ]


def _wikipedia_search_steps(query: str) -> list[dict[str, Any]]:
    sels = _KNOWN_SELECTORS["wikipedia.org"]
    return [
        {"action": "navigate",      "url": "https://en.wikipedia.org", "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "fill",          "selector": sels["search_input"], "value": query},
        {"action": "press",         "key": "Enter"},
        {"action": "wait_state",    "state": "networkidle"},
        {"action": "wait_selector", "selector": sels["results"]},
        {"action": "assert_text",   "text": query, "selector": sels["results"]},
    ]


def _navigate_steps(url: str) -> list[dict[str, Any]]:
    import urllib.parse as _urlparse
    if not re.match(r"https?://", url):
        url = "https://" + url
    domain = _urlparse.urlparse(url).netloc
    return [
        {"action": "navigate",    "url": url, "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
        {"action": "wait_state",  "state": "networkidle"},
        {"action": "assert_url",  "pattern": domain},
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

# Operating rules applied to every planning prompt.  Kept as a separate
# constant so callers (e.g. the REPL, custom skills, voice pipelines) can
# embed them in their own contexts without duplicating the text.
_AGENT_OPERATING_RULES = """\
You are a careful, goal-driven browser agent.

Your job is to complete web tasks accurately, safely, and efficiently.

Operating rules:
1. First identify the user's goal, required inputs, constraints, and completion criteria.
2. Break the task into small verifiable steps.
3. Before each important action, confirm that the current page matches your expectation.
4. Prefer robust selectors, visible labels, stable text, and semantic page structure over fragile assumptions.
5. If multiple similar elements exist, pause and disambiguate using nearby text, section headings, button labels, or form context.
6. After every click, form submission, or navigation, verify the result before proceeding.
7. Detect common blockers: popups, cookie banners, captchas, sign-in walls, permission dialogs, slow loads, hidden menus, expired sessions, and validation errors.
8. If blocked, attempt safe recovery:
   - close or dismiss overlays
   - refresh once if needed
   - navigate back if taken off task
   - retry with an alternative path
   - report clearly if user intervention is required
9. Never submit the same form twice unless you confirm the first submission failed.
10. Never perform destructive, financial, legal, or irreversible actions without explicit confirmation.
11. Extract important information in structured form whenever useful.
12. Keep track of completed steps and avoid repeating work.
13. If the task cannot be completed, explain exactly where it failed, why, and what the user should provide next.
14. When the task succeeds, provide a concise completion summary with relevant outputs.

Behavior expectations:
- Be precise, not fast-and-loose.
- Do not assume; inspect.
- Do not hallucinate missing page content.
- Do not invent success.
- Use recovery strategies before giving up.
- Minimize unnecessary navigation.
- Preserve context across tabs, page changes, and multi-step workflows.

Completion standard:
A task is complete only when the requested action is verified on the page or the requested information is extracted and checked for relevance.\
"""

_SYSTEM_PROMPT = _AGENT_OPERATING_RULES + """

---

Your ONLY output must be a valid JSON step array (no prose, no markdown, no explanation).

PLANNING RULES (MUST follow all):
1. Output ONLY a valid JSON array. No prose, no markdown, no explanation.
2. Each element must have an "action" key matching one of the allowed actions below.
3. Use the fewest steps possible. Never add unnecessary steps.
4. Maximum 20 steps. If you cannot do the task in 20 steps, return an error step.
5. For well-known sites (Google, Bing, YouTube, Wikipedia, DuckDuckGo) use the
   exact selectors listed below — never invent selectors.
6. After navigate always add close_popups.
7. MANDATORY VERIFICATION: Any plan that performs a form submission, login,
   checkout, or navigation to an authenticated page MUST end with an assert_text
   or assert_url step confirming the expected outcome.
   - assert_text: check for a confirmation heading, success message, or next-step
     indicator that would only be present on successful completion.
   - assert_url: check for a URL path fragment expected after the action
     (e.g. "/dashboard", "/search?q=").
   - If you cannot identify a specific success indicator, use assert_url with the
     most specific URL fragment you can determine from the task description.
   - Never omit this step when the task has a verifiable outcome.
8. FORM INPUT: Use "type" instead of "fill" for all form inputs. The "fill" action
   sets the field value directly without firing JavaScript onChange/onInput events,
   which leaves React, Vue, and Angular forms in an invalid state (submit button
   stays disabled, field may clear on blur). Use "fill" only when you are certain
   the target is a plain server-rendered HTML form and you explicitly set
   framework_safe to false.

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
  {"action": "type", "selector": "textarea[name='q']", "text": "python tutorials"},
  {"action": "press", "key": "Enter"},
  {"action": "wait_selector", "selector": "#search"},
  {"action": "assert_text", "text": "python", "selector": "#search"}
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
  {"action": "type", "selector": "input#search", "text": "machine learning"},
  {"action": "press", "key": "Enter"},
  {"action": "wait_selector", "selector": "ytd-video-renderer"},
  {"action": "assert_text", "text": "machine learning", "selector": "body"}
]

Now output ONLY the JSON array for the task below. Nothing else.
"""

# Vision-capable system prompt — identical in spirit but notes that a
# screenshot is available so the model should ground selectors in what it sees.
_VISION_SYSTEM_PROMPT = _AGENT_OPERATING_RULES + """

---

You also have access to a SCREENSHOT of the current page.
Use it to understand the current page state and generate accurate selectors.
Prefer semantic selectors (text=, label=, placeholder=, role=) that are grounded in the
visible content of the screenshot, rather than brittle CSS class names.

Your ONLY output must be a valid JSON step array (no prose, no markdown, no explanation).

PLANNING RULES (MUST follow all):
1. Output ONLY a valid JSON array. No prose, no markdown, no explanation.
2. Each element must have an "action" key matching one of the allowed actions below.
3. Use the fewest steps possible. Never add unnecessary steps.
4. Maximum 20 steps. If you cannot do the task in 20 steps, return an error step.
5. After navigate always add close_popups.
6. MANDATORY VERIFICATION: Any plan that performs a form submission, login,
   checkout, or navigation to an authenticated page MUST end with an assert_text
   or assert_url step confirming the expected outcome.
   - assert_text: check for a confirmation heading, success message, or next-step
     indicator present only on successful completion.
   - assert_url: check for a URL path fragment expected after the action.
   - Never omit this step when the task has a verifiable outcome.
7. FORM INPUT: Use "type" instead of "fill" for all form inputs. The "fill" action
   bypasses JavaScript onChange/onInput events, leaving React/Vue/Angular forms
   in an invalid state. Use "fill" only for plain server-rendered HTML forms.
8. Prefer text=, label=, placeholder=, role= selectors over CSS when possible.

ALLOWED ACTIONS (schema):
""" + json.dumps(STEP_SCHEMA, indent=2) + """

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


def _call_vision_openai(intent: str, screenshot_b64: str) -> list[dict[str, Any]]:
    """Call GPT-4V (OpenAI vision) with a screenshot + intent and return validated steps.

    The screenshot is sent as an inline base-64 PNG.  The model is asked to
    ground its selector choices in the visible page content.

    Parameters
    ----------
    intent:
        Natural-language task description.
    screenshot_b64:
        Base-64 encoded PNG screenshot of the current browser page.

    Returns
    -------
    list[dict]
        Validated step list.

    Raises
    ------
    StepValidationError
        When the response cannot be parsed or validated.
    """
    import openai  # type: ignore[import]

    model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": intent},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content or ""
    logger.debug("GPT-4V raw response: %s", raw)

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Extract the first JSON array from the response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise StepValidationError(f"GPT-4V response contains no JSON array: {raw[:300]}")

    parsed = json.loads(match.group())
    if isinstance(parsed, list):
        return validate_steps(parsed)
    raise StepValidationError(f"GPT-4V returned unexpected JSON shape: {type(parsed)}")


def _call_vision_anthropic(intent: str, screenshot_b64: str) -> list[dict[str, Any]]:
    """Call Anthropic Claude vision with a screenshot + intent and return validated steps.

    Parameters
    ----------
    intent:
        Natural-language task description.
    screenshot_b64:
        Base-64 encoded PNG screenshot of the current browser page.

    Returns
    -------
    list[dict]
        Validated step list.

    Raises
    ------
    StepValidationError
        When the response cannot be parsed or validated.
    """
    import anthropic  # type: ignore[import]

    model = os.environ.get("ANTHROPIC_VISION_MODEL", "claude-3-5-sonnet-20241022")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_VISION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": intent},
                ],
            }
        ],
    )

    raw = message.content[0].text if message.content else ""
    logger.debug("Claude vision raw response: %s", raw)

    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise StepValidationError(f"Claude vision response contains no JSON array: {raw[:300]}")

    parsed = json.loads(match.group())
    if isinstance(parsed, list):
        return validate_steps(parsed)
    raise StepValidationError(f"Claude vision returned unexpected JSON shape: {type(parsed)}")


def _call_vision_gemini(intent: str, screenshot_b64: str) -> list[dict[str, Any]]:
    """Call Google Gemini vision with a screenshot + intent and return validated steps.

    Parameters
    ----------
    intent:
        Natural-language task description.
    screenshot_b64:
        Base-64 encoded PNG screenshot of the current browser page.

    Returns
    -------
    list[dict]
        Validated step list.

    Raises
    ------
    StepValidationError
        When the response cannot be parsed or validated.
    """
    import base64 as _base64

    import google.generativeai as genai  # type: ignore[import]

    model_name = os.environ.get("GEMINI_VISION_MODEL", "gemini-1.5-pro")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=_VISION_SYSTEM_PROMPT,
    )

    image_bytes = _base64.b64decode(screenshot_b64)
    image_part = {"mime_type": "image/png", "data": image_bytes}

    response = model.generate_content(
        [image_part, intent],
        generation_config={"temperature": 0.0, "max_output_tokens": 1024},
    )

    raw = response.text if hasattr(response, "text") else ""
    logger.debug("Gemini vision raw response: %s", raw)

    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise StepValidationError(f"Gemini vision response contains no JSON array: {raw[:300]}")

    parsed = json.loads(match.group())
    if isinstance(parsed, list):
        return validate_steps(parsed)
    raise StepValidationError(f"Gemini vision returned unexpected JSON shape: {type(parsed)}")


def _call_vision_ollama(intent: str, screenshot_b64: str) -> list[dict[str, Any]]:
    """Call a local Ollama multimodal model with a screenshot + intent and return validated steps.

    Uses the ``/api/generate`` endpoint with the ``images`` field, which is
    supported by Ollama multimodal models such as ``llava`` and ``bakllava``.

    Parameters
    ----------
    intent:
        Natural-language task description.
    screenshot_b64:
        Base-64 encoded PNG screenshot of the current browser page.

    Returns
    -------
    list[dict]
        Validated step list.

    Raises
    ------
    StepValidationError
        When the response cannot be parsed or validated.
    """
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_VISION_MODEL", "llava")

    prompt = f"{_VISION_SYSTEM_PROMPT}\n\nTask: {intent}"

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [screenshot_b64],
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode()

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    raw = body.get("response", "")
    logger.debug("Ollama vision raw response: %s", raw)

    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise StepValidationError(f"Ollama vision response contains no JSON array: {raw[:300]}")

    steps = json.loads(match.group())
    return validate_steps(steps)


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

    @staticmethod
    def _detect_vision_provider() -> str | None:
        """Return the vision provider to use based on available environment variables.

        Priority order:
        1. ``OPENAI_API_KEY``   → ``"openai"``
        2. ``ANTHROPIC_API_KEY`` → ``"anthropic"``
        3. ``GOOGLE_API_KEY``   → ``"gemini"``
        4. Ollama running       → ``"ollama"``
        5. None (fall back to text-only)
        """
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        if os.environ.get("OLLAMA_HOST") or _ollama_running():
            return "ollama"
        return None

    def vision_plan(
        self, intent: str, agent: Any, *, provider: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert *intent* to a validated step list using a live screenshot.

        Takes a screenshot of the current browser page and sends it together
        with the intent to a vision-capable LLM so the model can ground its
        selector choices in the actual visible content.

        The vision provider is selected in the following order:

        1. The explicit *provider* argument when supplied (``"openai"``,
           ``"anthropic"``, ``"gemini"``, or ``"ollama"``).
        2. Auto-detection based on available environment variables
           (``OPENAI_API_KEY`` → OpenAI, ``ANTHROPIC_API_KEY`` → Anthropic,
           ``GOOGLE_API_KEY`` → Gemini, Ollama running → Ollama).
        3. Falls back to text-only :meth:`plan` when no vision provider is
           available or when the vision LLM call fails.

        Parameters
        ----------
        intent:
            Natural-language description of the task to perform on the page.
        agent:
            A ``BrowserAgent`` instance used to capture the screenshot.
        provider:
            Explicit vision provider override (``"openai"``, ``"anthropic"``,
            ``"gemini"``, or ``"ollama"``).  When ``None`` the provider is
            auto-detected from the environment.

        Returns
        -------
        list[dict]
            Validated step list.
        """
        intent = intent.strip()
        logger.info("Vision planning intent: %r", intent)

        resolved_provider = provider or self._detect_vision_provider()

        if not resolved_provider:
            logger.info("No vision provider available; falling back to text-only planning")
            return self.plan(intent)

        try:
            screenshot_result = agent.screenshot(as_base64=True)
            screenshot_b64: str = screenshot_result.get("base64", "")
        except Exception as exc:
            logger.warning(
                "Screenshot failed during vision_plan (%s); falling back to text-only", exc
            )
            return self.plan(intent)

        if not screenshot_b64:
            logger.warning("Empty screenshot in vision_plan; falling back to text-only")
            return self.plan(intent)

        _vision_callers = {
            "openai": _call_vision_openai,
            "anthropic": _call_vision_anthropic,
            "gemini": _call_vision_gemini,
            "ollama": _call_vision_ollama,
        }
        vision_fn = _vision_callers.get(resolved_provider)
        if vision_fn is None:
            logger.warning("Unknown vision provider %r; falling back to text-only", resolved_provider)
            return self.plan(intent)

        logger.info("Using vision provider: %s", resolved_provider)
        try:
            return vision_fn(intent, screenshot_b64)
        except Exception as exc:
            logger.warning(
                "Vision LLM call failed (%s); falling back to text-only planning", exc
            )
            return self.plan(intent)

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
            elif action in ("extract_links", "extract_table") and isinstance(result, dict):
                import json as _json
                last = _json.dumps(result, ensure_ascii=False)
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

        Returns a summary dict with the following keys:

        ``success``
            ``True`` when no step raised an exception (unchanged semantics).
        ``verified``
            ``True`` when the last ``assert_text`` or ``assert_url`` step
            confirmed the expected outcome (``found``/``matched`` == True).
            ``False`` when an assertion step ran but the check failed.
            ``None`` when no assertion steps exist in the plan — meaning
            verification was not attempted; ``success`` is still meaningful.
        ``steps``
            The planned step list.
        ``results``
            Per-step result records.
        ``failed_count``
            Number of steps with status ``"error"``.

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

        # Compute verified: True/False when an assertion step ran, None when none did.
        assertion_results = [
            r for r in results
            if r.get("action") in ("assert_text", "assert_url") and r.get("status") == "ok"
        ]
        if not assertion_results:
            verified: bool | None = None
        else:
            last_assert_result = assertion_results[-1].get("result") or {}
            verified = bool(
                last_assert_result.get("found") or last_assert_result.get("matched")
            )

        summary = {
            "success":      len(failed) == 0,
            "verified":     verified,
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
            result = agent.navigate(step["url"], wait_until=step.get("wait_until", "domcontentloaded"))
            # Login-wall detection: if we intended to navigate to a non-auth URL
            # but the browser landed on an auth/login URL, execution must stop.
            # Continuing would silently run all subsequent steps against the
            # wrong page (the login form), producing misleading results.
            _AUTH_TOKENS = (
                "login", "signin", "sign-in", "auth", "sso",
                "oauth", "session/new", "account/login",
            )
            target = step["url"].lower()
            actual = agent.page.url.lower()
            _target_is_auth = any(t in target for t in _AUTH_TOKENS)
            _actual_is_auth = any(t in actual for t in _AUTH_TOKENS)
            if not _target_is_auth and _actual_is_auth:
                raise ValueError(
                    f"Login wall detected: navigating to {step['url']!r} redirected to "
                    f"{agent.page.url!r}. Task cannot proceed without authentication."
                )
            return result

        if action == "click":
            return agent.click(step["selector"], timeout=step.get("timeout"))

        if action == "fill":
            # framework_safe=True (default): route through type_text so that
            # JavaScript onChange/onInput events fire and React/Vue/Angular
            # component state stays in sync with the field value.
            # framework_safe=False: use raw Playwright fill (no key events) —
            # faster, but only correct for plain server-rendered HTML forms.
            if step.get("framework_safe", True):
                return agent.type_text(
                    step["selector"],
                    step["value"],
                    clear_first=True,
                )
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

        if action == "start_recording":
            return agent.start_video_recording(step["path"])

        if action == "stop_recording":
            return agent.stop_video_recording()

        if action == "record_gif":
            return agent.record_gif(
                path=step["path"],
                duration=float(step.get("duration", 3.0)),
                fps=int(step.get("fps", 2)),
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

        # ---- Smart extraction ----

        if action == "extract_links":
            sel = step.get("selector", "a")
            if step.get("wait_for_ready", True) and sel != "a":
                try:
                    agent.wait_for_selector(sel, timeout=step.get("timeout", 5_000))
                except Exception:
                    pass  # proceed — extraction may still succeed
            return agent.extract_links(
                selector=sel,
                limit=step.get("limit", 100),
            )

        if action == "extract_table":
            sel = step.get("selector", "table")
            if step.get("wait_for_ready", True) and sel != "table":
                try:
                    agent.wait_for_selector(sel, timeout=step.get("timeout", 5_000))
                except Exception:
                    pass  # proceed — extraction may still succeed
            return agent.extract_table(
                selector=sel,
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

        # ---- High-priority advanced interactions ----

        if action == "upload_file":
            st = self._get_system_tools()
            file_path = str(st.workspace / step["path"])
            return agent.upload_file(step["selector"], file_path)

        if action == "download_file":
            st = self._get_system_tools()
            save_path = str(st.workspace / step["path"])
            return agent.download_file(step["url"], save_path)

        if action == "drag_drop":
            return agent.drag_and_drop(step["source"], step["target"])

        if action == "right_click":
            return agent.right_click(step["selector"])

        if action == "double_click":
            return agent.double_click(step["selector"])

        if action == "get_rect":
            return agent.get_element_rect(step["selector"])

        if action == "set_network_intercept":
            return agent.set_network_intercept(
                step["url_pattern"],
                action=step.get("intercept_action", "abort"),
            )

        if action == "clear_network_intercepts":
            return agent.clear_network_intercepts()

        if action == "set_viewport":
            return agent.set_viewport(int(step["width"]), int(step["height"]))

        if action == "set_geolocation":
            return agent.set_geolocation(
                float(step["latitude"]),
                float(step["longitude"]),
                accuracy=float(step.get("accuracy", 10.0)),
            )

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
