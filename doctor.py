"""
doctor.py — Environment health checks and failure analysis for agenticbrowser.

Two public surfaces:
  1. run_checks(workspace, fix) — environment diagnostics (used by --doctor CLI flag
     and the GET /doctor API route).
  2. explain_failure(error, ...)  — analyse a step/command failure and suggest fixes
     (used by the CLI after a failed task/command).
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str   # "ok" | "warn" | "fail"
    message: str
    fixed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":    self.name,
            "status":  self.status,
            "message": self.message,
            "fixed":   self.fixed,
        }


@dataclass
class FailureReport:
    """Structured failure analysis returned by :func:`explain_failure`."""

    reason:      str
    suggestions: list[str] = field(default_factory=list)
    rerun_hint:  str = ""


# ---------------------------------------------------------------------------
# Required-package registry  (import_name, pip_install_name)
# ---------------------------------------------------------------------------

_REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("playwright", "playwright"),
    ("fastapi",    "fastapi"),
    ("uvicorn",    "uvicorn"),
    ("pydantic",   "pydantic"),
    ("PIL",        "pillow"),
    ("httpx",      "httpx"),
]


# ---------------------------------------------------------------------------
# Individual environment checks
# ---------------------------------------------------------------------------

def _check_python_version() -> CheckResult:
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}"
    if (major, minor) >= (3, 10):
        return CheckResult("python_version", "ok", f"Python {ver}")
    return CheckResult(
        "python_version", "warn",
        f"Python {ver} — 3.10+ recommended",
    )


def _check_packages(fix: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    for import_name, install_name in _REQUIRED_PACKAGES:
        if importlib.util.find_spec(import_name) is not None:
            results.append(
                CheckResult(f"package:{install_name}", "ok", f"{install_name} is installed")
            )
            continue
        if fix:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", install_name],
                    check=True, capture_output=True, timeout=120,
                )
                results.append(CheckResult(
                    f"package:{install_name}", "ok",
                    f"{install_name} installed (fixed)", fixed=True,
                ))
            except Exception as exc:
                results.append(CheckResult(
                    f"package:{install_name}", "fail",
                    f"Failed to install {install_name}: {exc}",
                ))
        else:
            results.append(CheckResult(
                f"package:{install_name}", "fail",
                f"{install_name} is not installed — run: pip install {install_name}",
            ))
    return results


def _check_chromium(fix: bool = False) -> CheckResult:
    """Check whether Playwright's Chromium browser binary is present."""
    if importlib.util.find_spec("playwright") is None:
        return CheckResult(
            "chromium", "fail",
            "playwright package not installed — run: pip install playwright",
        )
    try:
        r = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--list"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and "chromium" in r.stdout.lower():
            return CheckResult("chromium", "ok", "Playwright Chromium browser is installed")
    except Exception as exc:
        return CheckResult("chromium", "fail", f"Cannot query playwright browsers: {exc}")

    if fix:
        try:
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=True, capture_output=True, timeout=300,
            )
            return CheckResult(
                "chromium", "ok",
                "Playwright Chromium installed (fixed)", fixed=True,
            )
        except Exception as exc:
            return CheckResult("chromium", "fail", f"playwright install chromium failed: {exc}")

    return CheckResult(
        "chromium", "fail",
        "Playwright Chromium not installed — run: playwright install chromium",
    )


def _check_workspace(workspace: str, fix: bool = False) -> CheckResult:
    path = Path(workspace)
    if path.is_dir():
        return CheckResult("workspace", "ok", f"Workspace directory exists: {workspace}")
    if fix:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return CheckResult(
                "workspace", "ok",
                f"Workspace directory created: {workspace}", fixed=True,
            )
        except Exception as exc:
            return CheckResult("workspace", "fail", f"Cannot create workspace '{workspace}': {exc}")
    return CheckResult("workspace", "fail", f"Workspace directory missing: {workspace}")


def _check_openai() -> CheckResult:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return CheckResult("openai_api_key", "ok", "OPENAI_API_KEY is set")
    return CheckResult(
        "openai_api_key", "warn",
        "OPENAI_API_KEY not set (optional — LLM planning disabled)",
    )


def _check_ollama() -> CheckResult:
    host = os.environ.get("OLLAMA_HOST", "")
    if not host:
        return CheckResult("ollama", "warn", "OLLAMA_HOST not set (optional — Ollama LLM disabled)")
    try:
        import urllib.request
        url = host.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
            if resp.status == 200:
                return CheckResult("ollama", "ok", f"Ollama reachable at {host}")
    except Exception as exc:
        return CheckResult("ollama", "fail", f"Ollama unreachable at {host}: {exc}")
    return CheckResult("ollama", "fail", f"Ollama returned unexpected response at {host}")


def _check_port(port: int = 8000) -> CheckResult:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        in_use = sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()
    if in_use:
        return CheckResult(
            "port_8000", "warn",
            f"Port {port} is already in use (API server may fail to start)",
        )
    return CheckResult("port_8000", "ok", f"Port {port} is free")


# ---------------------------------------------------------------------------
# Public: run all checks
# ---------------------------------------------------------------------------

def run_checks(workspace: str = "workspace", fix: bool = False) -> list[CheckResult]:
    """
    Run all environment checks, optionally auto-fixing remediable issues.

    Parameters
    ----------
    workspace:
        Path to the workspace directory (used for the workspace-exists check).
    fix:
        When ``True``, attempt to auto-fix failures where possible:
        install missing packages via ``pip`` and run ``playwright install chromium``.
    """
    checks: list[CheckResult] = [_check_python_version()]
    checks.extend(_check_packages(fix=fix))
    checks.append(_check_chromium(fix=fix))
    checks.append(_check_workspace(workspace=workspace, fix=fix))
    checks.append(_check_openai())
    checks.append(_check_ollama())
    checks.append(_check_port())
    return checks


# ---------------------------------------------------------------------------
# Public: failure analysis
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[tuple[str, str, list[str]]] = [
    (
        "TimeoutError",
        "A page element was not found within the timeout period.",
        [
            "Inspect available elements:  query <selector>",
            "Try a semantic selector:  text=<visible text>  or  label=<label>",
            "Wait for the page to settle:  wait_state networkidle",
            "Debug in headed mode:  python local_runner.py --no-headless",
        ],
    ),
    (
        "net::ERR_",
        "Navigation failed — the page or URL could not be reached.",
        [
            "Verify the URL is correct and reachable from this machine",
            "Check network or proxy settings",
            "Confirm manually in the REPL:  navigate <url>",
        ],
    ),
    (
        "ERR_NAME_NOT_RESOLVED",
        "DNS resolution failed — the hostname could not be resolved.",
        [
            "Check the URL hostname is spelled correctly",
            "Verify your network/DNS configuration",
        ],
    ),
    (
        "selector",
        "A CSS/text selector did not match any element on the page.",
        [
            "Inspect elements:  query <selector>",
            "Try semantic selectors:  text=<text>  or  label=<label>  or  placeholder=<text>",
            "Take a screenshot to see the current page state:  screenshot",
        ],
    ),
    (
        "Element is not visible",
        "The target element exists but is hidden or off-screen.",
        [
            "Scroll the element into view first:  scroll_to <selector>",
            "Dismiss overlapping popups:  close_popups",
        ],
    ),
    (
        "OPENAI_API_KEY",
        "OpenAI API key is missing — LLM-based task planning is unavailable.",
        [
            "Set the environment variable:  export OPENAI_API_KEY=<your-key>",
            "Or use Ollama:  export OLLAMA_HOST=http://localhost:11434",
            "Run doctor:  python local_runner.py --doctor",
        ],
    ),
    (
        "ollama",
        "Ollama LLM backend is unreachable.",
        [
            "Start Ollama:  ollama serve",
            "Check OLLAMA_HOST is correct:  echo $OLLAMA_HOST",
            "Or use OpenAI:  export OPENAI_API_KEY=<your-key>",
        ],
    ),
    (
        "StepValidationError",
        "A task step failed schema validation.",
        [
            "Check valid actions:  GET /task/schema  or  task_plan help",
            "Ensure all required fields for the action are present",
        ],
    ),
    (
        "No module named",
        "A required Python package is not installed.",
        [
            "Auto-fix:  python local_runner.py --doctor --fix",
            "Or manually:  pip install -r requirements.txt",
        ],
    ),
    (
        "Browser session is not active",
        "No browser session is running — start one first.",
        [
            "Start a session via the API:  POST /session/start",
            "Or use the CLI:  python local_runner.py",
            "Or via the GUI:  click 'Start Session' in the sidebar",
        ],
    ),
    (
        "PermissionError",
        "A filesystem permission error occurred.",
        [
            "Check the workspace directory is writable",
            "Run environment doctor:  python local_runner.py --doctor",
        ],
    ),
    (
        "FileNotFoundError",
        "A required file was not found.",
        [
            "Verify the file path is correct",
            "List workspace contents:  list_dir .",
        ],
    ),
]

_GENERIC_SUGGESTIONS: list[str] = [
    "Run environment diagnostics:  python local_runner.py --doctor",
    "Enable debug logging:  LOG_LEVEL=DEBUG python local_runner.py ...",
    "Capture the current page:  screenshot",
]


def explain_failure(
    error: str,
    *,
    action: str | None = None,
    step_index: int | None = None,
    intent: str | None = None,
    rerun_cmd: str | None = None,
) -> FailureReport:
    """
    Analyse *error* and return a structured :class:`FailureReport`.

    Parameters
    ----------
    error:
        The raw error/exception message string.
    action:
        The browser action that failed (e.g. ``"click"``).
    step_index:
        Which step in the task failed (0-based).
    intent:
        The natural-language intent that was being executed.
    rerun_cmd:
        Shell command the user can run to retry the same task.
    """
    reason = "An unexpected error occurred."
    suggestions: list[str] = []

    for pattern, pat_reason, pat_suggestions in _ERROR_PATTERNS:
        if pattern.lower() in error.lower():
            reason = pat_reason
            suggestions = list(pat_suggestions)
            break

    if not suggestions:
        suggestions = list(_GENERIC_SUGGESTIONS)

    if action:
        prefix = f"Failed action: {action!r}"
        if step_index is not None:
            prefix = f"Failed at step {step_index} ({action!r})"
        reason = f"{prefix}. {reason}"

    if intent:
        suggestions.insert(0, f"Retry the same task:  task {intent}")

    return FailureReport(
        reason=reason,
        suggestions=suggestions,
        rerun_hint=rerun_cmd or "",
    )
