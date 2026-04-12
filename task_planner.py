import re
from typing import Any

from pydantic import BaseModel

STEP_SCHEMA = [
    "navigate", "click", "fill", "type_text", "clear", "press_key",
    "scroll", "scroll_to_element", "screenshot", "get_text", "get_attribute",
    "get_url", "get_title", "hover", "double_click", "right_click",
    "wait_for_selector", "wait_text", "select_option", "check", "uncheck",
    "upload_file", "extract_links", "extract_table", "assert_text", "assert_url",
    "save_cookies", "load_cookies", "new_tab", "switch_tab", "close_tab", "list_tabs",
]


class Step(BaseModel):
    action: str
    params: dict[str, Any] = {}


class Task(BaseModel):
    name: str
    steps: list[Step]


class TaskResult(BaseModel):
    name: str
    success: bool
    results: list[dict]
    error: str | None = None


def validate_step(step: dict) -> bool:
    """Return True if step has a valid 'action' in STEP_SCHEMA."""
    return isinstance(step.get("action"), str) and step["action"] in STEP_SCHEMA


def validate_task(task: dict) -> list[str]:
    """Validate task dict. Return list of error strings (empty = valid)."""
    errors = []
    if not task.get("name"):
        errors.append("Task must have a name")
    steps = task.get("steps", [])
    if not steps:
        errors.append("Task must have at least one step")
    for i, step in enumerate(steps):
        if not validate_step(step):
            action = step.get("action", "<missing>")
            errors.append(f"Step {i}: invalid action '{action}'")
    return errors


def plan_task(name: str, goal: str) -> Task:
    """Create a simple task with a navigate step from a goal string."""
    url_match = re.search(r'https?://\S+', goal)
    url = url_match.group(0) if url_match else "https://example.com"
    return Task(name=name, steps=[Step(action="navigate", params={"url": url})])


def summarize_results(results: list[dict]) -> dict:
    """Return {"total": n, "success": n, "failed": n, "errors": [...]}"""
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    failed = total - successes
    errors = [r["error"] for r in results if not r.get("success") and "error" in r]
    return {"total": total, "success": successes, "failed": failed, "errors": errors}


def merge_tasks(*tasks: Task) -> Task:
    """Merge multiple tasks into one, combining all steps."""
    if not tasks:
        return Task(name="merged", steps=[])
    name = " + ".join(t.name for t in tasks)
    steps = [step for task in tasks for step in task.steps]
    return Task(name=name, steps=steps)
