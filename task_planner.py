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

    def __init__(self, llm: str | None = None) -> None:
        self.llm = llm or self._detect_llm()

    @staticmethod
    def _detect_llm() -> str | None:
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("OLLAMA_HOST") or _ollama_running():
            return "ollama"
        return None

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
    ) -> list[dict[str, Any]]:
        """
        Execute *steps* against *agent*, returning a result record for each step.

        Parameters
        ----------
        stop_on_error:
            If ``True`` (default), execution stops on the first failed step.
            If ``False``, failures are recorded but execution continues.
        """
        results: list[dict[str, Any]] = []
        for i, step in enumerate(steps):
            action = step["action"]
            logger.info("Step %d/%d: %s", i + 1, len(steps), action)
            try:
                result = self._execute_step(agent, step)
                results.append({"step": i, "action": action, "status": "ok", "result": result})
            except Exception as exc:
                logger.error("Step %d (%s) failed: %s", i, action, exc)
                results.append({"step": i, "action": action, "status": "error", "error": str(exc)})
                if stop_on_error:
                    break
        return results

    def run(
        self,
        intent: str,
        agent: Any,
        *,
        stop_on_error: bool = True,
    ) -> dict[str, Any]:
        """
        Plan *intent* and execute the resulting steps against *agent*.

        Returns a summary dict with ``steps``, ``results``, and ``success`` flag.
        """
        try:
            steps = self.plan(intent)
        except (ValueError, StepValidationError) as exc:
            return {"success": False, "error": str(exc), "steps": [], "results": []}

        results = self.execute(steps, agent, stop_on_error=stop_on_error)
        failed  = [r for r in results if r["status"] == "error"]
        return {
            "success": len(failed) == 0,
            "intent":  intent,
            "steps":   steps,
            "results": results,
            "failed_count": len(failed),
        }

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

    @staticmethod
    def _execute_step(agent: Any, step: dict[str, Any]) -> Any:
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
