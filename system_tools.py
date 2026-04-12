"""
SystemTools: file system access and code execution for the Agentic Browser.

All file operations are confined to a configurable ``workspace`` directory
(default: ``./workspace`` relative to the current working directory) to prevent
path-traversal attacks.

Code execution runs in a subprocess with a configurable timeout so a runaway
script cannot hang the agent.

Classes
-------
SystemTools
    The main class.  Instantiate once and pass to ``TaskPlanner.execute()``.

Helpers
-------
safe_path(workspace, user_path)
    Resolve ``user_path`` inside ``workspace`` and verify it does not escape.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default timeout (seconds) for subprocess code/command execution.
_DEFAULT_EXEC_TIMEOUT = 30

# Maximum characters captured from stdout/stderr of a subprocess.
# Prevents runaway scripts from filling memory with gigabytes of output.
_MAX_OUTPUT_CHARS = 1_000_000  # ≈ 1 MB of ASCII text


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

class PathTraversalError(ValueError):
    """Raised when a user-supplied path attempts to escape the workspace."""


def safe_path(workspace: Path, user_path: str) -> Path:
    """
    Resolve *user_path* inside *workspace* and raise ``PathTraversalError`` if
    the result escapes the workspace directory.

    Parameters
    ----------
    workspace:
        The allowed root directory (must be absolute).
    user_path:
        The user-supplied relative or absolute path.

    Returns
    -------
    Path
        Absolute, resolved path guaranteed to be inside *workspace*.
    """
    workspace = workspace.resolve()
    candidate = (workspace / user_path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        raise PathTraversalError(
            f"Path {user_path!r} escapes the workspace {workspace}. "
            "Only paths inside the workspace are allowed."
        )
    return candidate


# ---------------------------------------------------------------------------
# SystemTools
# ---------------------------------------------------------------------------

class SystemTools:
    """
    File-system read/write and code execution tools.

    Parameters
    ----------
    workspace : str | Path | None
        Root directory for all file operations.  Defaults to ``./workspace``
        (created automatically if it does not exist).
    exec_timeout : int
        Maximum seconds allowed for ``run_python`` / ``run_shell`` calls.
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        exec_timeout: int = _DEFAULT_EXEC_TIMEOUT,
    ) -> None:
        self.workspace = Path(workspace or "workspace").resolve()
        self.exec_timeout = exec_timeout
        self.workspace.mkdir(parents=True, exist_ok=True)
        logger.info("SystemTools workspace: %s", self.workspace)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def write_file(
        self,
        path: str,
        content: str,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """
        Write *content* to *path* inside the workspace.

        Parameters
        ----------
        path:
            Relative path inside the workspace (e.g. ``"results/output.txt"``).
        content:
            Text content to write.
        mode:
            ``"w"`` to overwrite (default) or ``"a"`` to append.
        encoding:
            File encoding (default ``"utf-8"``).

        Returns
        -------
        dict
            ``{"path": absolute_path, "bytes_written": N}``
        """
        target = safe_path(self.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "w":
            target.write_text(content, encoding=encoding)
        else:
            with target.open("a", encoding=encoding) as fh:
                fh.write(content)
        size = len(content.encode(encoding))
        logger.info("write_file: %s (%d bytes, mode=%s)", target, size, mode)
        # bytes_written counts only the newly-written bytes, not the total file size.
        return {"path": str(target), "bytes_written": size, "mode": mode}

    def append_file(self, path: str, content: str, encoding: str = "utf-8") -> dict[str, Any]:
        """
        Append *content* to *path* (creates the file if it does not exist).
        """
        return self.write_file(path, content, mode="a", encoding=encoding)

    def read_file(self, path: str, encoding: str = "utf-8") -> dict[str, Any]:
        """
        Read and return the text content of *path*.

        Returns
        -------
        dict
            ``{"path": ..., "content": ..., "size": N}``
        """
        target = safe_path(self.workspace, path)
        if not target.exists():
            raise FileNotFoundError(f"File not found in workspace: {path!r}")
        content = target.read_text(encoding=encoding)
        logger.info("read_file: %s (%d chars)", target, len(content))
        return {"path": str(target), "content": content, "size": len(content)}

    def list_dir(self, path: str = ".") -> dict[str, Any]:
        """
        List the contents of *path* inside the workspace.

        Returns
        -------
        dict
            ``{"path": ..., "entries": [{"name": ..., "type": "file"|"dir", "size": N}]}``
        """
        target = safe_path(self.workspace, path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found in workspace: {path!r}")
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path!r}")
        entries = []
        for entry in sorted(target.iterdir()):
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })
        return {"path": str(target), "entries": entries, "count": len(entries)}

    def make_dir(self, path: str) -> dict[str, Any]:
        """Create *path* (and any missing parents) inside the workspace."""
        target = safe_path(self.workspace, path)
        target.mkdir(parents=True, exist_ok=True)
        return {"path": str(target), "created": True}

    def delete_file(self, path: str) -> dict[str, Any]:
        """
        Delete *path* inside the workspace.

        Works for both files and directories (directories are removed
        recursively).
        """
        target = safe_path(self.workspace, path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found in workspace: {path!r}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        logger.info("delete_file: %s", target)
        return {"path": str(target), "deleted": True}

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def run_python(
        self,
        code: str,
        timeout: int | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a Python code snippet in a subprocess and return the output.

        The snippet runs with ``sys.stdout`` / ``sys.stderr`` captured.
        Any ``extra_vars`` are injected as top-level variables via a generated
        preamble (values must be JSON-serialisable).

        Parameters
        ----------
        code:
            Python source code to execute.
        timeout:
            Override the instance ``exec_timeout`` (seconds).
        extra_vars:
            Optional dict of variable name → value to inject into the snippet's
            scope.  Example: ``{"page_text": "..."}`` makes ``page_text``
            available in the snippet.

        Returns
        -------
        dict
            ``{"stdout": ..., "stderr": ..., "exit_code": N, "success": bool}``
        """
        import json as _json

        preamble = ""
        if extra_vars:
            for var, val in extra_vars.items():
                preamble += f"{var} = {_json.dumps(val)}\n"

        full_code = preamble + textwrap.dedent(code)
        timeout = timeout or self.exec_timeout

        logger.info("run_python: %d chars of code (timeout=%ds)", len(full_code), timeout)

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
            )
            return {
                "stdout":    result.stdout[:_MAX_OUTPUT_CHARS],
                "stderr":    result.stderr[:_MAX_OUTPUT_CHARS],
                "exit_code": result.returncode,
                "success":   result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout":    "",
                "stderr":    f"Execution timed out after {timeout}s",
                "exit_code": -1,
                "success":   False,
            }

    def run_shell(
        self,
        command: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a shell command and return its output.

        The command is split with :func:`shlex.split` and executed directly
        (``shell=False``) — no shell is invoked and shell metacharacters such
        as pipes, redirects, or semicolons have no special meaning.  This is
        the safe default.

        For commands that genuinely require a shell (pipes, redirects, etc.),
        use :meth:`run_python` with ``subprocess`` instead::

            import subprocess, shlex
            r = subprocess.run("ls | grep foo", shell=True, capture_output=True, text=True)
            print(r.stdout)

        Parameters
        ----------
        command:
            Command string to execute (e.g. ``"ls -la"`` or ``"cat output.txt"``).
        timeout:
            Override the instance ``exec_timeout`` (seconds).

        Returns
        -------
        dict
            ``{"stdout": ..., "stderr": ..., "exit_code": N, "success": bool}``
        """
        import shlex as _shlex

        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string.")

        try:
            cmd = _shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"Invalid command syntax: {exc}") from None

        timeout = timeout or self.exec_timeout
        logger.info("run_shell: %r (timeout=%ds)", command, timeout)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                cwd=str(self.workspace),
            )
            return {
                "stdout":    result.stdout[:_MAX_OUTPUT_CHARS],
                "stderr":    result.stderr[:_MAX_OUTPUT_CHARS],
                "exit_code": result.returncode,
                "success":   result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout":    "",
                "stderr":    f"Command timed out after {timeout}s",
                "exit_code": -1,
                "success":   False,
            }

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Return a summary of the workspace and Python executable."""
        entries = self.list_dir(".")
        return {
            "workspace":  str(self.workspace),
            "python":     sys.executable,
            "file_count": sum(1 for e in entries["entries"] if e["type"] == "file"),
            "dir_count":  sum(1 for e in entries["entries"] if e["type"] == "dir"),
        }
